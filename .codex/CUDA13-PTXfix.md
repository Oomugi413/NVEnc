# CUDA 13 / NVRTC 12.9 PTXエラー調査・変更まとめ

## 概要

2026-08-05に、次のエンコードログで発生したCUDA PTXエラーを調査した。

```text
/mnt/recording/20260805_[4K]運転席からの風景　ＪＲ鶴見線-enc.log
```

結論は次のとおり。

1. `--vpp-colorspace` の失敗原因は、NVRTC 13.3が生成したPTXを実行時のCUDAドライバが受け付けない `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` だった。
2. NVEncCが実行時に動的ロードするNVRTCを12.9系へ切り替えると、同じcolorspace処理が成功した。検証時のパッケージは12.9.86だった。
3. CUDA 13.3のコンパイラはRTX 50シリーズ向け `sm_120` 生成に必要なため、ビルドツールチェーンは維持し、NVRTCランタイムだけ12.9系にする構成とした。
4. `--vpp-libplacebo-tonemapping` はNVRTCではなくVulkan/shaderc経路であり、NVRTC 12.9/13.3のどちらでも動作した。ただし、配備済みNVEncC 9.29.1では別のSPIR-V不整合が発生するため、9.30.1以降が必要になる。

## 発生したエラー

ログ中のNVEncC実行条件は、概略次のとおり。

```text
NVEncC 9.29.1
--codec hevc --profile main10 --output-depth 10
--vpp-resize libplacebo-ewa-lanczos
--vpp-colorspace matrix=bt2020nc:bt709,
                 colorprim=bt2020:bt709,
                 transfer=bt2020-10:bt709,
                 range=limited:limited
```

失敗箇所は `--vpp-colorspace` のCUDAカーネル生成である。

```text
colorspace_conv: failed to instantiate program source.
colorspace_conv: CUDA_ERROR_UNSUPPORTED_PTX_VERSION
colorspace: failed to setup custom filter: error in cuda..
Unsupported vpp filter type.
```

実行環境は次のとおり。

```text
GPU: NVIDIA GeForce RTX 5070 Ti
Driver: 595.84
NVENC API: 13.0
CUDA runtime reported by NVEncC: 13.2
NVRTC: 13.3
```

## 原因の切り分け

`NVEncCore/rgy_nvrtc.cpp` はNVRTCをリンク時に固定せず、実行時に `libnvrtc.so` をロードして関数を解決する実装になっている。このため、同じNVEncCバイナリに対して `LD_LIBRARY_PATH` の優先順位を変更し、NVRTCランタイムだけを差し替えられる。

### `--vpp-colorspace` の対照試験

同じNVEncC、同じ入力、同じcolorspace指定で、NVRTCだけを切り替えた。

テスト入力は、640x360・59.94fps・10bit YUV・HLG/BT.2020の短いY4Mを使用した。

| NVEncC | NVRTC | 結果 |
|---|---|---|
| 9.29.1 | 13.3 | `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`で失敗 |
| 9.29.1 | 12.9.86 | 60/60フレーム完走 |
| 9.30.1（CUDA 13.3でビルド） | 13.3 | 同じPTXエラーで失敗 |
| 9.30.1（CUDA 13.3でビルド） | 12.9.86 | 60/60フレーム完走、約193fps |

NVRTC 12.9.86で成功した出力は、`ffprobe`でも次を確認した。

```text
codec_name=hevc
profile=Main 10
width=640
height=360
pix_fmt=yuv420p10le
color_range=tv
color_space=bt709
color_transfer=bt709
color_primaries=bt709
nb_read_frames=60
```

この比較により、PTXエラーはNVEncCのCUDAフィルター実装そのものより、NVRTC 13.3が生成するPTXと実行側ドライバの互換性に依存していると判断した。

## `--vpp-libplacebo-tonemapping` の試験

NVEncC 9.30.1をCUDA 13.3でビルドし、HLGからSDRへのトーンマッピングを試験した。

```text
--vpp-libplacebo-tonemapping src_csp=hlg,dst_csp=sdr
```

9.30.1では、NVRTC 12.9.86/13.3の両方で60/60フレーム完走した。ログには次が出力された。

```text
libplacebo v7.360.1
shaderc SPIR-V version 1.6
Vulkan driver 595.84
```

さらに、元の設定に近い次の併用も成功した。

```text
--vpp-libplacebo-tonemapping src_csp=hlg,dst_csp=sdr
--vpp-resize libplacebo-ewa-lanczos
--output-res 1920x1080
```

結果は1920x1080、60/60フレーム、約69.7fpsだった。

一方、配備済みのNVEncC 9.29.1ではNVRTC 12.9.86に切り替えても失敗した。

```text
libplacebo: shaderc output:
shaderc: internal error: compilation succeeded but failed to optimize:
Invalid SPIR-V binary version 1.6 for target environment SPIR-V 1.0
```

これはNVRTC/PTXとは別の問題で、9.29.1に含まれるlibplacebo 7.351.0とshaderc/SPIR-Vターゲットの組み合わせに起因する。libplaceboトーンマッピングを使用する場合は、9.30.1以降を使用する必要がある。

## 9.26.1のエンコードが成功した理由

次のエンコードログについて、NVEncC 9.26.1が成功した理由をNVEncのGit履歴と対照試験結果から整理した。

```text
/mnt/recording/20260804_＜アニメギルド＞最強出涸らし皇子の暗躍帝位争い　#5-enc.log
```

このコマンドラインには`--vpp-colorspace`が含まれておらず、使用しているのは次のlibplacebo系フィルタだった。

```text
--vpp-resize libplacebo-ewa-lanczos
--vpp-libplacebo-tonemapping src_csp=hlg,dst_csp=sdr
```

ログにも次が記録されており、NVRTCではなくVulkan/libplacebo経路で処理されたことが確認できる。

```text
libplacebo: Initialized libplacebo v7.360.1
libplacebo: shaderc SPIR-V version 1.6
libplacebo: Driver info: 595.84
encoded 43633 frames
```

NVRTCは`--vpp-colorspace`のCUDAカスタムフィルタ生成時に使用される。一方、libplaceboフィルタはVulkanデバイス上でshadercによりSPIR-Vを生成する。このため、このエンコードの成功はNVRTC 13.3のPTX問題を解決したことを意味しない。

バイナリに組み込まれた依存ライブラリにも差があった。

| 構成 | libplacebo | shaderc/SPIR-V | 結果 |
|---|---:|---:|---|
| NVEncC 9.26.1の実エンコード | v7.360.1 | 1.6 | 成功 |
| 配備済みNVEncC 9.29.1の対照試験 | v7.351.0 | 1.6 | SPIR-Vターゲット不整合で失敗 |
| NVEncC 9.30.1の対照試験 | v7.360.1 | 1.6 | 成功 |

shadercについては、ログ上のSPIR-Vバージョンは両方とも1.6であり、shadercのバージョン番号そのものが異なるとは断定できない。正確には、libplaceboとshadercを含むビルド済みツールチェーンの組み合わせが異なり、9.29.1ではSPIR-V 1.6を生成した後にSPIR-V 1.0向けとして検証される不整合が発生した。

Git履歴上、9.26.1から9.29.1の間で`NVEncFilterLibplacebo`と`rgy_libplacebo`に差分はなく、9.29.1から9.30.1でも関連するNVEncソースとDockerfileに差分はない。公式Linuxビルドはlibplaceboをstatic linkし、外部の`rigaya/build_scripts`を`master`から取得してビルドするため、libplaceboの具体的なバージョンはNVEncのGitタグだけでは再現できない。

## リポジトリに加えた変更

### `docker/docker_ubuntu2404_cuda13`

CUDA 13.3のツールチェーンを維持したまま、NVRTC 12.9系を導入するように変更した。APTのパッケージ名だけを指定し、パッチ番号は固定していない。

```dockerfile
ARG NVENCC_CUDA_VERSION=13.3.0
...
...
cuda-nvrtc-12-9
cuda-nvrtc-dev-12-9
```

12.9のライブラリをCUDA 13.3側より先にロードするため、次のパスも設定した。

```dockerfile
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.9/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
```

これにより、次の役割を分離している。

```text
コンパイル: CUDA 13.3（sm_120を含むネイティブコード生成）
NVRTC実行: CUDA 12.9系
```

### `NVEncCore/rgy_version.h`

バージョン定義を9.30.2へ更新した。

```text
VER_FILEVERSION:          0,9,30,2
VER_STR_FILEVERSION:      9.30.2
VER_STR_FILEVERSION_TCHAR: 9.30.2
```

リポジトリ全体を検索し、9.30.1のバージョン定義が残っていないことを確認した。

## 検証状況

完了した検証:

- Docker BuildKitのDockerfile構文チェック成功。
- CUDA 13.3ベースの一時コンテナへNVRTC 12.9系（検証時12.9.86）をインストール。
- `LD_LIBRARY_PATH`を変更したPython `ctypes`ロード試験で `NVRTC 12.9` を確認。
- NVRTC 12.9/13.3のcolorspace対照エンコード。
- libplaceboトーンマッピング単体およびリサイズ併用試験。
- `git diff --check`相当の差分検査。
- 9.30.1の残存バージョン参照がないことを確認。

未実施事項:

- 変更後のDockerfileによる公式ベースイメージ全体の再ビルド。
- バージョンを9.30.2へ更新したソースのフルCUDAビルド。
- GitHub Actionsでの公式パッケージ生成。

今回の実機エンコード試験で使用したNVEncCは、バージョン番号を9.30.2へ更新する前の9.30.1ソースでビルドしたもの。9.30.2への変更はバージョン定義のみで、エンコード実装の変更はない。

## 配布時の注意

Linuxの `nvencc` パッケージはNVRTCライブラリ自体を同梱しておらず、通常は `libc6`のみを依存関係として持つ。今回のDockerfile変更は公式ビルド環境とビルド時のNVRTCロード順を修正するもので、既存環境へ配布したバイナリが自動的にNVRTC 12.9を同梱する変更ではない。

実運用環境でもcolorspaceを使用する場合は、NVRTC 12.9系をインストールし、NVEncCから見えるライブラリ検索パスに配置する必要がある。

推奨構成は次のとおり。

```text
NVEncC: 9.30.2以降
ビルドツールチェーン: CUDA 13.3
実行時NVRTC: 12.9系
GPUドライバ: 595.84（今回の試験環境）
```

## 前回の記録以降に判明したこと・追加変更

### Vulkan stub版libplaceboの原因確定

GitHub Actionsのbase imageログと生成された`.deb`を追加確認した結果、stub版libplaceboはNVEncのリンク時に偶然混入したものではなく、base imageのlibplaceboビルド時点で生成されていたことが判明した。

`libplacebo 7.360.1`に対して、Vulkan Loader/Headerが`1.3.295`、shadercが`2023.8.1`のままだったため、libplaceboの次の検査が失敗していた。

```text
shaderc_env_version_vulkan_1_4: NO
VK_VERSION_1_4: NO
vk-proc-addr: NO
vulkan: NO
Compiling .../vulkan_stubs.c.o
```

このため、NVEnc側のMesonが`Vulkan: true`、`libplacebo: true`と表示されても、実際に静的リンクされたlibplaceboはVulkan stub版となった。生成された9.30.2パッケージでも、`pl_vulkan_create`のサイズが`0x22`で、`libplacebo compiled without Vulkan support!`が含まれることを確認した。

### build_scriptsの依存ライブラリ更新

原因に対応するため、`Oomugi413/build_scripts`の`ffmpeg_dll/build_ffmpeg_dll.sh`を更新した。

- shaderc: `v2024.1` → `v2026.2`
- Vulkan Loader/Header: `v1.3.295` → `v1.4.356`
- libplacebo: `v7.360.1`を継続使用

libplacebo 7.360.1が要求するVulkan 1.4およびshaderc対応環境を揃え、Vulkan stub版ではなく実装版を生成できる構成にした。変更は`Oomugi413/build_scripts`のcommit `faa2498`としてpush済み。

詳細な変更記録は次を参照。

- [Oomugi413/build_scripts/.codex/libplacebo_update.md](https://github.com/Oomugi413/build_scripts/blob/master/.codex/libplacebo_update.md)

### base imageタグの再現性に関する追加確認

base image作成時にpushされたdigestは`sha256:fbbd5c56...`だったが、パッケージ作成時の`docker pull`では`sha256:169ec670...`が取得されていた。同じmutableなタグを使用しているため、base image作成ジョブとパッケージ作成ジョブで異なるimageを使用する可能性がある。今後はbase image digestを固定してパッケージを作成する必要がある。

## セッション情報

```text
Codex session ID: 019fceed-24b2-7c83-ac28-4068c31a6bb0
```
