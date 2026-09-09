# RTX 50 / CUDA 13.3 配布ビルド対応まとめ

## セッション

- Codex session ID: `019f8d9c-bbb9-7fe1-8fa4-fcd28130a9db`
- 作業日: `2026-07-23`
- 対象リポジトリ: `/home/oomugi413/git/NVEnc`
- GitHubリポジトリ: `Oomugi413/NVEnc`
- 元のKonomiTV対応セッション: `019f8353-44ae-7bb0-a00c-75d4f18ccdb8`
- 元のKonomiTV実装コミット: `0f0314938323a5bcea7d1f6ae2856aa163d5f70d`
- NVEnc実装コミット:
  - `e4ab71f5` Build Ubuntu 24.04 packages with CUDA 13.3
  - `4af240ee` Fix GHCR image naming and build ordering

## 目的

KonomiTV側で個別に行っていたRTX 50シリーズ向けNVEncCビルドを、`rigaya/NVEnc`が使用する配布パッケージ用GitHub Actionsへ移植した。

主な要件:

- Ubuntu 24.04 / CUDA 13.3でNVEncCをビルドする。
- RTX 50シリーズ向けの`sm_120`ネイティブcubinを含める。
- `docker_ubuntu2004_cuda11`で有効だったAviSynthPlus、VapourSynth、Vship用ヘッダーを維持する。
- `Oomugi413/NVEnc`のGitHub ActionsとGHCRで動作させる。
- Ubuntu 24.04 / CUDA 13.3を既定の配布対象とし、旧ターゲットは必要時のみ再有効化できるようにする。

## 変更内容

### `docker/docker_ubuntu2404_cuda13`

新しい配布ビルド用Dockerfileを追加した。

- ベースイメージ:

  ```dockerfile
  nvidia/cuda:13.3.0-devel-ubuntu24.04
  ```

- FFmpegはKonomiTV側で動作確認した`8.1.2`へ固定。
- `rigaya/build_scripts`はGitHub Actionsで解決したコミットIDを取得して使用。
- `build_ffmpeg_dll.sh --skip-src-archive --disable-pgo`で静的依存ライブラリを構築。
- Linux側Vulkan Loaderを`SHARED`から`STATIC`へ補正し、`libvulkan.a`として再構築。
- libvmafの`enable_float=true`を維持。
- Rustの`cargo-c`はKonomiTV側で動作確認した`0.10.18+cargo-0.92.0`へ固定。
- 次のNVEncC用公開ヘッダーを配置:
  - AviSynthPlus 3.7.5
  - VapourSynth R72
  - Vship 5.0.0
- FFmpeg、libplacebo、libvmaf、Vulkan Loader、NPP、CUDA RuntimeをNVEncCへ静的リンクできるベース環境を作成。

### `.github/workflows/build_base_images.yml`

- 既定のベースイメージを`ubuntu2404_cuda13`のみに変更。
- `ubuntu2004_cuda11`と`fedora39_cuda12`はDockerfileを残したまま、マトリクス上でコメントアウト。
- `workflow_call`を追加し、Linuxパッケージワークフローから先行実行できるようにした。
- GitHubの所有者名`Oomugi413`をDockerが許可する小文字へ変換し、次のGHCRイメージを発行:

  ```text
  ghcr.io/oomugi413/nvenc-build:ubuntu2404_cuda13
  ```

- `GITHUB_TOKEN`の`packages: write`権限でGHCRへログインするため、個人アクセストークンの追加発行は不要。

初回実装では`ghcr.io/Oomugi413/...`を使用したため、Dockerの次の制約で失敗した。

```text
repository name must be lowercase
```

コミット`4af240ee`で所有者名を小文字化して修正した。

### `.github/workflows/build_packages.yml`

- 既定の配布対象を次の1種類へ変更:

  ```yaml
  dockerimg: ubuntu2404_cuda13
  pkgtype: deb
  arch: x86_64
  ```

- Linuxパッケージビルドの前に`build_base_images.yml`を呼び、ベースイメージの発行成功後にだけパッケージビルドを開始するよう依存関係を追加。
- GHCRからprivateイメージも取得できるよう、`GITHUB_TOKEN`でログイン。
- NVEncCビルド後、`cuobjdump --list-elf`で`sm_120`を検査。1個も含まれなければ配布を失敗させる。
- `ldd`と`readelf`で欠落ライブラリおよびlibplacebo/libvmafの予期しない動的依存を検査。
- `check_options.py`と`NVEncC --version`をGPUなしのrunner上で実行。
- GitHub Releaseへファイルを追加できるよう`contents: write`を明示。

### 旧パッケージのファイル名

既定のCUDA 13.3版は従来どおり次の名前になる。

```text
nvencc_9.25_amd64.deb
```

旧マトリクスをコメント解除した場合は、`package_suffix`によってファイル名を区別できる。

```text
nvencc_9.25_ubuntu2004_cuda11_amd64.deb
```

Fedora RPMにも`fedora39_cuda12`サフィックスを付けられる。パッケージ内部の名前とインストール後のコマンド名は`nvencc`のまま変更しない。

## CUDAコード生成対象

CUDA 13.3では現行`meson.build`の条件により、次のネイティブコードを生成する。

```text
sm_75
sm_86
sm_120
```

KonomiTV側のCUDA 12.8ビルドに含まれていた`sm_50`と`sm_61`は、CUDA 13ではソース側の条件により対象外になる。

## GitHub Actions結果

- Workflow: `Build Linux Packages`
- Run ID: `29985819310`
- Run URL: `https://github.com/Oomugi413/NVEnc/actions/runs/29985819310`
- Source commit: `4af240ee72ff78aca8ce76f25e95b74f26dc0e2f`
- Base image job: success
- Ubuntu 24.04 / CUDA 13.3 deb job: success
- Artifact:
  - ID: `8556455657`
  - Name: `nvencc_ubuntu2404_cuda13_deb`
  - ZIP size: 116,212,705 bytes
  - File: `nvencc_9.25_amd64.deb`

## 生成物検証

GitHub ActionsのArtifactをダウンロードして展開した。

### deb

```text
Package: nvencc
Version: 9.25
Architecture: amd64
Depends: libc6(>=2.31)
SHA-256: 26abb89add1fd85e27dfb1cdd33a8c9aac610d74cd1b572bbb2d4f6868d51f9d
```

### NVEncCバイナリ

```text
NVEnc (x64) 9.25 (r3931)
gcc 13.3.0/Linux
NVENC API v13.0
CUDA 13.3
SHA-256: 07752e6725970b19daefa321656bd0025a41f3fd7bee2f47a021308a4c2550f6
sm_120 cubins: 84
```

`readelf`で確認した動的依存:

```text
libgcc_s.so.1
libcuda.so.1
libm.so.6
libmvec.so.1
libc.so.6
ld-linux-x86-64.so.2
```

`libav*`、`libplacebo`、`libvmaf`、`libvulkan`、`libnpp`、`libcudart`の動的依存はない。

## RTX 5070 Ti実機エンコード試験

### 環境

```text
GPU: NVIDIA GeForce RTX 5070 Ti
Driver: 610.43.02
Compute capability: 12.0
OS: Ubuntu 24.04
```

入力:

```text
/mnt/recording/20260721_孤独のグルメ　Season２＃９＜全１２話＞.hevc.ts
```

KonomiTV同梱FFmpegを使い、`-re`で30秒間の映像・音声をMPEG-TS標準出力へ送った。GitHub Actions生成NVEncCへパイプし、KonomiTVで停止していた条件を再現した。

主要条件:

- MPEG-TS標準入力
- `--avhw`
- HEVC / Main 10
- 1440x1080
- `--vpp-deband`
- `--vpp-afs preset=default`
- `--interlace tff`
- `--lookahead 16`
- `--multipass 2pass-full`
- AAC 192K / 48kHz / stereo
- MPEG-TS出力
- low latency / output thread無効

### 結果

```text
FFmpeg exit code: 0
NVEncC exit code: 0
encoded 884 frames, 30.47 fps, 2933.03 kbps, 10.31 MB
RunEncode2: finished.
real=30.07
user=28.00
sys=1.09
cpu=96%
maxrss_kb=472344
voluntary_cs=32255
involuntary_cs=320
```

出力:

```text
/tmp/nvencc-cuda13-output.ts
size: 11,991,016 bytes
duration: 29.5295 seconds
container: MPEG-TS
video: HEVC Main 10, yuv420p10le, 1440x1080, 30000/1001 fps
audio: AAC LC, 48000 Hz, stereo
```

NVDEC入力終了、NVENC flush、出力writer終了、CUDA context破棄まで正常に完了した。元のKonomiTVで発生していた「エンコードを開始しています」から進まず完了しない現象は再現しなかった。

ログ中の次のメッセージは`--disable-nvml 1`使用時の性能監視に関するもので、エンコード結果には影響しない。

```text
perf monitor: Failed to start NVML Monitoring
```

## 結論と注意点

- GitHub Actions配布ビルドでNVEncC 9.25 / CUDA 13.3 / Ubuntu 24.04パッケージを生成できた。
- ArtifactにはRTX 50向け`sm_120`ネイティブcubinが84個含まれる。
- KonomiTVの旧失敗条件を含む30秒リアルタイム実機試験は終了コード0で完走した。
- 出力MPEG-TSのHEVC Main 10映像とAAC音声をffprobeで確認した。
- CUDA 12系ビルドで確認済みだった約1 CPU coreの高使用率はCUDA 13.3でも96%として残る。今回の変更はエンコード停止の解消とRTX 50対応配布物の生成を達成するが、高CPU使用率そのものは解消していない。
- GitHub ActionsとGHCRは自動`GITHUB_TOKEN`で動作するため、現構成ではユーザーによるPAT発行は不要。
- この文書と`.codex/`は調査記録であり、Gitへ追加・コミットしない。
