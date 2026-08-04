# CUDA 12 / RTX 50 NVEncC 調査引き継ぎ

## セッション

- Codex session ID: `019f8353-44ae-7bb0-a00c-75d4f18ccdb8`
- 作成日: `2026-07-23`
- 調査元: `/home/oomugi413/git/KonomiTV`
- 調査先: `/home/oomugi413/git/NVEnc`
- KonomiTV 側実装コミット: `0f0314938323a5bcea7d1f6ae2856aa163d5f70d`
- コミットタイトル: `Build NVEncC 9.25 with configurable CUDA`
- NVEnc 9.25 source revision: `8c873e4d15aefb93dd50396e5c70fffb842f7d22`
- build_scripts revision: `1fda04abb39f897ab54d98a44d28da50de7237b2`
- GPU: NVIDIA GeForce RTX 5070 Ti
- OS前提: Ubuntu 24.04

## 現在までの結論

1. KonomiTV が以前ダウンロードしていた公式 NVEncC 9.18/9.25 の Ubuntu パッケージは CUDA 11.2 / GCC 9.4 系であり、実録画をリアルタイム入力した30秒試験では NVEncC の CPU 使用率が約9%だった。
2. NVEncC 9.25 を CUDA 12.8 / GCC 13.3 / Ubuntu 24.04 でビルドしたものは、同一入力・同一NVEncCオプションで約98%、ほぼCPU 1コアを消費する。
3. rigaya upstream の Fedora 39 / CUDA 12.4 方式に近い公式 RPM 版 NVEncC 9.25 も約96%だった。このため KonomiTV独自Dockerfile、Ubuntu 24.04、FFmpeg 8.1.2、sm_120追加だけが原因ではない。
4. NVEnc 9.25自体が原因ではない。公式 CUDA 11.2版の9.25は約9%。
5. `--cuda-schedule sync`、`--output-thread 1`、perf monitor関連の切り分けでは改善しなかった。
6. `NVEncCore/NVEncPipeline.h` のパイプライン用CUDAイベント6個を `cudaEventDefault` から `cudaEventBlockingSync` に変更して再ビルドしたが、95%で改善しなかった。変更したイベント由来の voluntary context switch は増えたためパッチは動作したが、主なビジーウェイトは別経路。
7. CUDA 12系で別の `cuda*Synchronize`、Driver API、NVDEC/NVENC、NPP/VPP、またはCUDAコンテキストスケジューリングがCPUをスピンさせている可能性が高い。次は高CPUスレッドの複数時点スタック、`perf`、Nsight Systems等でホットスポットを直接特定する。

## 発端

- KonomiTVの旧 `server/thirdparty/NVEncC/NVEncC.elf` は正常動作。
- 手動配置した新しいNVEncCでは「エンコードを開始しています」から進まない現象があった。
- RTX 50向けの `sm_120` ネイティブcubinを含む、依存関係を静的リンクしたNVEncC 9.25 / CUDA 12.8版を作成した。
- 作成版はRTX 5070 TiでエンコードおよびKonomiTV再生に成功した。
- その後、旧9.18時代より `NVEncC.elf` のCPU使用率が高いことをユーザーが発見し、CUDA世代差の調査へ移行した。

## KonomiTV側で変更したファイル

コミット `0f0314938323a5bcea7d1f6ae2856aa163d5f70d`:

```text
.dockerignore
.github/workflows/build_thirdparty.yaml
.github/workflows/docker/nvencc-ubuntu24.04.Dockerfile
.github/workflows/scripts/build_nvencc_local.sh
.github/workflows/scripts/verify_nvencc.sh
.gitignore
```

確認:

```bash
cd /home/oomugi413/git/KonomiTV
git show --stat 0f0314938323a5bcea7d1f6ae2856aa163d5f70d
git show 0f0314938323a5bcea7d1f6ae2856aa163d5f70d -- \
  .github/workflows/build_thirdparty.yaml \
  .github/workflows/docker/nvencc-ubuntu24.04.Dockerfile \
  .github/workflows/scripts/build_nvencc_local.sh \
  .github/workflows/scripts/verify_nvencc.sh
```

### build_thirdparty.yaml

主要変数:

```yaml
env:
  NVENCC_VERSION: '9.25'
  NVENCC_CUDA_VERSION: '12.8.1'
  NVENCC_FFMPEG_VERSION: '8.1.2'
```

Linux amd64ジョブでは、従来の `nvencc_${NVENCC_VERSION}_amd64.deb` ダウンロードを廃止し、Thirdparty配置・圧縮前にDocker Buildxでソースビルド:

```yaml
- name: Build NVEncC with configured CUDA (for x64)
  if: matrix.arch == 'amd64'
  uses: docker/build-push-action@v5
  with:
    context: .
    file: .github/workflows/docker/nvencc-ubuntu24.04.Dockerfile
    build-args: |
      NVENCC_VERSION=${{ env.NVENCC_VERSION }}
      NVENCC_CUDA_VERSION=${{ env.NVENCC_CUDA_VERSION }}
      NVENCC_FFMPEG_VERSION=${{ env.NVENCC_FFMPEG_VERSION }}
    target: artifact
    outputs: type=local,dest=nvencc-build
    cache-from: type=gha,scope=nvencc-ubuntu24.04-cuda-${{ env.NVENCC_CUDA_VERSION }}
    cache-to: type=gha,scope=nvencc-ubuntu24.04-cuda-${{ env.NVENCC_CUDA_VERSION }},mode=max
```

配置:

```bash
mkdir thirdparty/NVEncC/
cp nvencc-build/NVEncC.elf thirdparty/NVEncC/NVEncC.elf
cp nvencc-build/License.txt thirdparty/NVEncC/License.txt
cp nvencc-build/BuildInfo.txt thirdparty/NVEncC/BuildInfo.txt
chmod a+x thirdparty/NVEncC/NVEncC.elf
patchelf --set-rpath '$ORIGIN:$ORIGIN/../lib:$ORIGIN/../Library' thirdparty/NVEncC/NVEncC.elf
```

### Ubuntu 24.04 / CUDA 12.8 Dockerfile

ベース:

```dockerfile
ARG NVENCC_CUDA_VERSION=12.8.1
FROM nvidia/cuda:${NVENCC_CUDA_VERSION}-devel-ubuntu24.04 AS builder
```

主要な構成:

- CUDAバージョンは `NVENCC_CUDA_VERSION` 一か所で変更可能。12.8以降を想定し、厳密な特定バージョン比較は行わない。
- NVEncCは数値タグ `9.25` で取得。コミットID固定は使用しない。
- FFmpegは `8.1.2`。
- SVT-AV1の個別バージョン指定は行わず、FFmpeg/build_scripts側に任せる。ビルド時はSVT-AV1 4.1.0。
- `rigaya/build_scripts` の `build_ffmpeg_dll.sh --skip-src-archive --disable-pgo` を使用。
- FFmpeg、libplacebo、libvmaf、Vulkan Loader、NPP/CUDA runtime等をNVEncCへ静的リンク。
- build_scriptsのLinux Vulkan Loaderが共有ライブラリになっていたため、CMake定義を `SHARED` から `STATIC` に補正。
- AviSynthPlus v3.7.5、VapourSynth R72、Vship v5.0.0の公開ヘッダーを配置。
- Meson release + LTO。
- `sm_120` cubin、NVEncCバージョン、共有ライブラリ依存、`check_options.py`をビルド内で検査。

NVEnc取得:

```dockerfile
RUN git clone --branch "${NVENCC_VERSION}" --depth 1 --recurse-submodules --shallow-submodules \
      https://github.com/rigaya/NVEnc.git /opt/nvencc \
    && source_version="$(sed -n 's/^#define VER_STR_FILEVERSION[[:space:]]*"\([^"]*\)".*$/\1/p' /opt/nvencc/NVEncCore/rgy_version.h | tr -d '\r\n')" \
    && test "${source_version}" = "${NVENCC_VERSION}"
```

NVEncビルド:

```bash
VMAFPKG='/opt/build_scripts/ffmpeg_dll/x64/build/lib/pkgconfig'
FFPKG='/opt/build_scripts/ffmpeg_dll/build_dll/x64/build/lib/pkgconfig:/opt/build_scripts/ffmpeg_dll/build_dll/x64/build/lib/x86_64-linux-gnu/pkgconfig'
export PKG_CONFIG_PATH="${VMAFPKG}:${FFPKG}"

meson setup ./build . \
  --buildtype=release \
  -Db_lto=true \
  -Denable_libplacebo=enabled \
  -Dlibplacebo_static=true \
  -Denable_vmaf=enabled \
  -Dlibvmaf_static=true \
  -Dcpp_args="['-I${AVISYNTH_HEADER_INC}','-I${VAPOURSYNTH_HEADER_INC}','-I${VSHIP_HEADER_INC}']"

grep -F '#define LIBVMAF_STATIC_CUDA_LINK 1' ./build/rgy_config.h
grep -F '#define ENABLE_LIBVSHIP 1' ./build/rgy_config.h
meson compile -C ./build -j"$(nproc)"
```

CUDA 12.8を検出したNVEnc 9.25のMeson構成により `compute_120` / `sm_120` が追加される。生成物では `cuobjdump --list-elf` により `sm_120` cubinを84個検出した。

### ローカル通常ビルド

```bash
cd /home/oomugi413/git/KonomiTV
bash .github/workflows/scripts/build_nvencc_local.sh
```

スクリプトは `.github/workflows/build_thirdparty.yaml` から3バージョンを取得し、Buildx `docker-container` builder `konomitv-nvencc` と `.build/NVEncC/cache` を再利用する。

等価な主要Buildxコマンド:

```bash
docker buildx build \
  --builder konomitv-nvencc \
  --progress plain \
  --target artifact \
  --build-arg NVENCC_VERSION=9.25 \
  --build-arg NVENCC_CUDA_VERSION=12.8.1 \
  --build-arg NVENCC_FFMPEG_VERSION=8.1.2 \
  --cache-from type=local,src=.build/NVEncC/cache \
  --cache-to type=local,dest=.build/NVEncC/cache-next,mode=max \
  --output type=local,dest=.build/NVEncC/output-next \
  --file .github/workflows/docker/nvencc-ubuntu24.04.Dockerfile \
  .
```

通常版BuildInfo:

```text
NVEnc (x64) 9.25 (r1) by rigaya, Jul 21 2026 12:41:09 (gcc 13.3.0/Linux)
  [NVENC API v13.0, CUDA 12.8]
CUDA source version: 12.8.1
FFmpeg source version: 8.1.2
NVEnc source revision: 8c873e4d15aefb93dd50396e5c70fffb842f7d22
build_scripts source revision: 1fda04abb39f897ab54d98a44d28da50de7237b2
e91027d1a81af9aad267da8c6acc25938030d235666982fac444bcc74b0235dd
```

## CUDA 12向けテスト

### 機能テスト

- NVEncC 9.25 / CUDA 12.8 / FFmpeg 8.1.2 通しビルド成功。
- `sm_120` cubin 84個。
- `check_options.py` 成功。
- `ldd` / `readelf` により、配布時に含められない `libav*`, `libplacebo`, `libvmaf`, `libvulkan`, `libnpp`, `libcudart` の動的依存がないことを確認。
- RTX 5070 Tiで実録画TSを入力した高速smoke test: 5,570 frames、約9秒、568.19fps、exit 0。
- ユーザーによるKonomiTV再生試験成功。

### CPU比較試験

入力:

```text
/mnt/recording/20260721_孤独のグルメ　Season２＃９＜全１２話＞.hevc.ts
```

入力をKonomiTV相当のリアルタイム速度に制限するため、KonomiTV同梱FFmpegで `-re`、30秒、映像・音声copy、MPEG-TS stdout:

```bash
server/thirdparty/FFmpeg/ffmpeg.elf \
  -hide_banner -loglevel error -re \
  -i '/mnt/recording/20260721_孤独のグルメ　Season２＃９＜全１２話＞.hevc.ts' \
  -map 0:v:0 -map '0:a:0?' -c copy -t 30 -f mpegts -
```

NVEncCオプション:

```bash
NVEncC.elf \
  --input-format mpegts \
  --input-probesize 2000K \
  --input-analyze 1.1 \
  --fps 30000/1001 \
  --input - \
  --avhw \
  --audio-stream '1?:stereo' \
  --audio-stream '2?:stereo' \
  --data-copy timed_id3 \
  -m avioflags:direct \
  -m fflags:nobuffer+flush_packets \
  -m flush_packets:1 \
  -m max_delay:250000 \
  -m max_interleave_delta:700K \
  --output-thread 0 \
  --lowlatency \
  --disable-nvml 1 \
  --disable-dx11 \
  --disable-vulkan \
  --log-level debug \
  --codec hevc \
  --video-tag hvc1 \
  --vbr 3000K \
  --max-bitrate 4500K \
  --qp-min 23:26:30 \
  --lookahead 16 \
  --multipass 2pass-full \
  --bref-mode middle \
  --aq \
  --aq-temporal \
  --repeat-headers \
  --preset default \
  --profile main \
  --dar 16:9 \
  --vpp-deband \
  --output-depth 10 \
  --fallback-bitdepth \
  --interlace tff \
  --vpp-afs preset=default \
  --avsync vfr \
  --gop-len 15 \
  --output-res 1440x1080 \
  --audio-codec aac:aac_coder=twoloop \
  --audio-bitrate 192K \
  --audio-samplerate 48000 \
  --audio-filter volume=2.0 \
  --audio-ignore-decode-error 30 \
  --output-format mpegts \
  --output /tmp/output.ts
```

計測:

```bash
/usr/bin/time \
  -f 'real=%e user=%U sys=%S cpu=%P maxrss_kb=%M voluntary_cs=%w involuntary_cs=%c'
```

ウォームアップ8秒後に30秒計測。すべて884 frames。

| バイナリ | real | user | sys | CPU | voluntary_cs | involuntary_cs |
|---|---:|---:|---:|---:|---:|---:|
| upstream公式 9.18 / CUDA 11.2 / GCC 9.4 | 30.17 | 1.57 | 1.28 | 9% | 554392 | 167 |
| upstream公式 9.25 / CUDA 11.2 / GCC 9.4 | 30.37 | 1.79 | 1.10 | 9% | 551508 | 165 |
| KonomiTV custom 9.25 / CUDA 12.8 / GCC 13.3 | 29.94 | 28.49 | 0.85 | 98% | 8289 | 340 |
| upstream Fedora39方式 9.25 / CUDA 12.4 / GCC 13.3 / FFmpeg 8.0 RPM | 29.93 | 28.46 | 0.56 | 96% | 8777 | 441 |
| custom + `--cuda-schedule sync` | 29.93 | 28.46 | 0.98 | 98% | 9355 | 434 |
| custom + `--output-thread 1` | 29.90 | 28.79 | 0.91 | 99% | 34891 | 433 |
| custom perf monitor切り分け | 29.88 | 28.48 | 0.67 | 97% | 8047 | 405 |
| custom + pipeline events `cudaEventBlockingSync` | 30.07 | 28.01 | 0.76 | 95% | 33291 | 360 |

比較ログはKonomiTVリポジトリ内:

```text
/home/oomugi413/git/KonomiTV/.build/NVEncC/cpu-comparison/
```

主要ファイル:

```text
measured-9.18.time
measured-9.25-official.time
measured-9.25.time
measured-9.25-cuda12.4-rpm.time
measured-9.25-custom-sync.time
measured-9.25-custom-output1.time
measured-9.25-custom-idleoff.time
measured-9.25-cuda12.8-blocking-sync.time
```

### cudaEventBlockingSync比較

NVEnc 9.25 `NVEncCore/NVEncPipeline.h`:

```text
line 640:  cudaEventSynchronize(*cuevent.get());
line 1282: cudaEventCreateWithFlags(event, cudaEventDefault)
line 2553: cudaEventCreateWithFlags(event, cudaEventDefault)
line 3351: cudaEventCreateWithFlags(&m_eventDefaultToFilter, cudaEventDefault)
line 3512: cudaEventCreateWithFlags(event, cudaEventDefault)
line 3649: cudaEventCreateWithFlags(event, cudaEventDefault)
line 3686: cudaEventCreateWithFlags(event, cudaEventDefault)
```

比較用Dockerfileに一時追加したパッチ:

```dockerfile
ARG NVENCC_CUDA_EVENT_BLOCKING_SYNC=false

RUN if [[ "${NVENCC_CUDA_EVENT_BLOCKING_SYNC}" == 'true' ]]; then \
      event_count="$(grep -c 'cudaEventCreateWithFlags(.*cudaEventDefault)' /opt/nvencc/NVEncCore/NVEncPipeline.h)"; \
      test "${event_count}" -eq 6; \
      sed -i 's/cudaEventDefault)/cudaEventBlockingSync)/g' /opt/nvencc/NVEncCore/NVEncPipeline.h; \
      test "$(grep -c 'cudaEventCreateWithFlags(.*cudaEventBlockingSync)' /opt/nvencc/NVEncCore/NVEncPipeline.h)" -eq 6; \
      ! grep -q 'cudaEventCreateWithFlags(.*cudaEventDefault)' /opt/nvencc/NVEncCore/NVEncPipeline.h; \
    fi
```

ビルド:

```bash
cd /home/oomugi413/git/KonomiTV
docker buildx build \
  --builder konomitv-nvencc \
  --progress plain \
  --target artifact \
  --build-arg NVENCC_VERSION=9.25 \
  --build-arg NVENCC_CUDA_VERSION=12.8.1 \
  --build-arg NVENCC_FFMPEG_VERSION=8.1.2 \
  --build-arg NVENCC_CUDA_EVENT_BLOCKING_SYNC=true \
  --cache-from type=local,src=.build/NVEncC/cache \
  --cache-to type=local,dest=.build/NVEncC/cache-blocking-sync,mode=max \
  --output type=local,dest=.build/NVEncC/blocking-sync-output \
  --file .github/workflows/docker/nvencc-ubuntu24.04.Dockerfile \
  .
```

生成物:

```text
/home/oomugi413/git/KonomiTV/.build/NVEncC/blocking-sync-output/NVEncC.elf
SHA-256: 91f181de2063d2701309cc138ef032d6f8b0dd5ca0c376fde897d2e3d3930c2e
size: 644 MiB
NVEnc 9.25 / CUDA 12.8 / gcc 13.3.0
sm_120 cubins: 84
```

ログ:

```text
/home/oomugi413/git/KonomiTV/.build/NVEncC/logs/build-cuda-event-blocking-sync.log
/home/oomugi413/git/KonomiTV/.build/NVEncC/cpu-comparison/measured-9.25-cuda12.8-blocking-sync.log
/home/oomugi413/git/KonomiTV/.build/NVEncC/cpu-comparison/measured-9.25-cuda12.8-blocking-sync.time
```

出力:

```text
encoded 884 frames, 30.42 fps, 2932.75 kbps, 10.31 MB
HEVC Main10, 1440x1080, yuv420p10le
AAC audio
duration=29.529500
size=11992708
```

結果:

```text
real=30.07 user=28.01 sys=0.76 cpu=95%
maxrss_kb=476428 voluntary_cs=33291 involuntary_cs=360
```

BlockingSync化前:

```text
real=29.94 user=28.49 sys=0.85 cpu=98%
maxrss_kb=472624 voluntary_cs=8289 involuntary_cs=340
```

イベントのblocking化によって voluntary context switchは約4倍になったが、user CPU時間は28.49秒から28.01秒にしか減らなかった。`PipelineTaskOutputSurf::depend_clear()` が待つ6イベントは主因ではない。

比較パッチは試験後にKonomiTV Dockerfileから除去済み。比較バイナリとログだけ `.build/NVEncC/` に保存。本番 `server/thirdparty/NVEncC/NVEncC.elf` はBlockingSync版へ置換していない。

なお `NVEncCore/rgy_cuda_util.h` の `CUFrameBufPair` にも次の既定イベント生成がある。今回の6イベント置換対象外:

```cpp
cudaEventCreate(&event);
```

NVEnc 9.25では同ファイル付近に2か所。使用経路と同期方法を次の調査で確認する。

## 比較したupstreamビルド定義

```text
https://github.com/rigaya/NVEnc/blob/master/.github/workflows/build_packages.yml
https://github.com/rigaya/NVEnc/blob/master/docker/docker_ubuntu2004_cuda11
https://github.com/rigaya/NVEnc/blob/master/docker/docker_ubuntu2404_cuda12
https://github.com/rigaya/NVEnc/blob/master/docker/docker_fedora39_cuda12
```

- upstream Ubuntu 20.04配布版: CUDA 11.2。
- upstream Ubuntu 24.04 CUDA 12定義も参照。
- upstream Fedora39 CUDA 12.4系の生成物でもCPU高負荷を再現。
- FFmpeg差よりCUDA世代差を優先して調査する。

## 次の調査候補

1. CUDA 12.4/12.8版を実行し、CPUを消費しているTIDを `top -H` / `ps -L` で特定。
2. 同一TIDのスタックを時間をずらして複数回取得。
3. debug symbols付き、LTOなし、最適化を必要に応じて下げたNVEncCを作成。
4. `perf record -g`, `perf top`, Nsight SystemsでユーザーCPUホットスポットを特定。
5. 次を全ソース検索し、CUDA 11とCUDA 12で待機動作を比較:

```bash
rg -n \
  'cuda(Event|Stream|Device).*Synchronize|cu(Event|Stream|Ctx).*Synchronize|cudaEventQuery|cuEventQuery|NV_ENC_LOCK_BITSTREAM|cuvidMapVideoFrame|cuvidUnmapVideoFrame' \
  .
```

6. `cudaEventCreate()` を使用する `CUFrameBufPair` の利用箇所と同期を確認。
7. CUDA Runtimeのコンテキスト作成前に `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)` が有効かを独立比較。ただし `--cuda-schedule sync` は改善しなかったため、既存オプションの実装位置・Runtime初期化順を確認する。
8. VPPを段階的に外す: `--vpp-afs`, `--vpp-deband`, 10bit変換、resize、avhw decode。どのパイプライン段階で1コア消費が発生するか二分探索。
9. CUDA 11.2ビルドとCUDA 12.xビルドに同一のNVEncソース、コンパイラ、FFmpeg依存を可能な限り揃え、CUDA Toolkitだけを変える。
10. CUDA 12.x内で12.4/12.8/13.xを比較。現時点で12.4と12.8はいずれも再現。
11. Driver versionとCUDA Runtime versionの組合せも記録する。

## 注意事項

- KonomiTVのサーバープロセスはユーザー管理。追加試験で進行中エンコードを止める必要がある場合は必ずユーザーへ依頼する。
- CPU比較はFFmpeg feederを `-re` にしてリアルタイム入力する。ファイルを無制限速度で処理したsmoke testのCPU%とは比較しない。
- 試験前に既存NVEncCプロセスを確認する。
- 成果物・ログ・BuildKitキャッシュはプロジェクト内の無視対象ビルドフォルダへ保存し、依存ビルドキャッシュを再利用する。
- 全工程の通しビルドは修正完了後に行う。
- `/home/oomugi413/git/KonomiTV/server/debug_dump_nal.bin` はユーザー所有の既存変更であり、触らない。
