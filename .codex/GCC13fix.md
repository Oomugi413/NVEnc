# GCC 13 / Linux NVDEC CPU使用率修正まとめ

## セッション

- Codex session ID: `019f8d9c-bbb9-7fe1-8fa4-fcd28130a9db`
- 作業日: `2026-07-23`
- 対象リポジトリ: `/home/oomugi413/git/NVEnc`
- GitHubリポジトリ: `Oomugi413/NVEnc`
- 前段のCUDA 13.3配布対応: `.codex/RTX50fix.md`
- CUDA 12時点のCPU調査: `.codex/CUDA12-RTX50.md`
- 対象GPU: NVIDIA GeForce RTX 5070 Ti
- OS: Ubuntu 24.04
- Driver: 610.43.02
- CUDA: 13.3
- Compiler: GCC 13.3

## 発端

CUDA 11.2 / GCC 9.4でビルドされた公式NVEncC 9.25は、KonomiTV相当のリアルタイム入力試験でCPU使用率が約9%だった。

同一NVEncソースをCUDA 12.4、CUDA 12.8、CUDA 13.3とGCC 13.3でビルドすると、同じ試験でNVEncCが約96～99%、ほぼCPU 1コアを使用した。

CUDAイベント、CUDAコンテキストのスケジュール、出力スレッド、性能監視を変更しても改善しなかった。

## 原因調査

### 高負荷スレッド

CUDA 13.3版をスレッド単位で計測すると、補助スレッドではなくNVEncCのメインスレッドが継続して約99%を使用していた。

GDBで停止したCUDA 13.3 / GCC 13.3版のスタック:

```text
pthread_mutex_lock()
CUVIDFrameQueue::dequeue()
PipelineTaskNVDecode::getOutputFrame()
NVEncCore::Encode()
main()
```

メインスレッドは、NVDECの出力フレームキューが空の間、`CUVIDFrameQueue::dequeue()`を高速に再試行していた。

### Windows版とLinux版の差

`FrameQueue::waitForQueueUpdate()`はWindowsでは次のように最大10ms待機する。

```cpp
WaitForSingleObject(hEvent_, 10);
```

`CUVIDFrameQueue::enqueue()`と`FrameQueue::endDecode()`は`SetEvent()`で待機スレッドへ通知する。

一方、従来のLinux版では`waitForQueueUpdate()`、`set_event()`、`reset_event()`が空実装だった。キューを保護する`pthread_mutex_t`は存在するが、キューが空のときに待機する処理がなかった。

### CUDA 11版で低負荷だった理由

Linux版`PipelineTaskNVDecode`はデコードスレッド管理に`std::future`を使用し、空キュー時に次の終了確認を行う。

```cpp
m_thDecoder.wait_for(std::chrono::milliseconds(0))
```

CUDA 11.2版に静的リンクされたGCC 9系libstdc++では、この0ms確認がfutexシステムコールを発行していた。これが偶然CPUをOSへ返し、Linux版に欠けていたキュー待機の代わりになっていた。

GCC 13系libstdc++では0ms確認がユーザー空間の状態確認で完結する。その結果、空実装の`waitForQueueUpdate()`と組み合わさり、キューを連続ポーリングする潜在不具合が表面化した。

12秒間のstrace比較:

| ビルド | futex呼び出し |
|---|---:|
| CUDA 11.2 / GCC 9.4 | 166,772 |
| CUDA 13.3 / GCC 13.3 | 1,183 |

CUDA 11版のfutexのうち166,478回は即時タイムアウトだった。

### CUDA API待機の除外

CUDA 13.3版をNsight Systemsで18秒追跡した。

```text
cudaEventSynchronize: 約2.7ms
cuEventSynchronize: 約1.4ms
追跡対象CUDA API全体: 約0.46秒
```

約1コアを消費していた時間の大部分はCUDA API内ではない。このため、次の既存切り分けで改善しなかった結果とも整合する。

- `--cuda-schedule sync`
- `cudaEventBlockingSync`
- `--output-thread 1`
- performance monitor無効化

CPU使用率の増加はCUDA 13固有ではなく、Ubuntu 20.04 / GCC 9からUbuntu 24.04 / GCC 13へ移行したことで、Linux版NVDECキューの待機欠落が表面化したものと判断した。

## 修正内容

### `NVEncCore/FrameQueue.h`

Linux用に`pthread_cond_t oQueueUpdateCondition_`を追加した。

### `NVEncCore/FrameQueue.cpp`

- コンストラクタでcondition variableを初期化。
- デストラクタでcondition variableを破棄。
- `waitForQueueUpdate()`で、Windows版と同じ最大10msの待機を実装。
- 待機前に同じmutex下で`nFramesInQueue_`と`bEndOfDecode_`を再確認し、通知取りこぼしを防止。
- `set_event()`で`pthread_cond_signal()`を実行。
- Linux版`endDecode()`で終了フラグをmutex下で更新してから通知。

処理の流れ:

```text
dequeueで空
  ↓
mutex下でキューと終了状態を再確認
  ↓
最大10msのcondition variable待機
  ↓
enqueueまたはendDecodeから通知されれば即起床
  ↓
dequeueを再試行
```

Windows版のソースには動作変更を加えていない。

### `.gitignore`

ビルド試験に使用するプロジェクト内一時フォルダを追加した。

```text
/.build-cuda13-framequeue-test/
```

Dockerイメージのビルドコンテキストは使用していないため、`.dockerignore`の追加は不要だった。

## ビルド試験

一時フォルダ:

```text
/home/oomugi413/git/NVEnc/.build-cuda13-framequeue-test/
```

GitHub Actionsと同じ`ghcr.io/oomugi413/nvenc-build:ubuntu2404_cuda13`を使用し、次の構成でフルビルドした。

- Ubuntu 24.04
- GCC 13.3
- CUDA 13.3
- Meson release
- LTO有効
- libplacebo静的リンク
- libvmaf静的リンク
- AviSynthPlus / VapourSynth / Vshipヘッダー有効

結果:

```text
218/218 targets: success
9.25.1更新後のincremental build 179/179 targets: success
NVEnc 9.25.1 r3931
NVENC API 13.0
CUDA 13.3
sm_120 cubins: 84
check_options.py: exit 0
```

生成バイナリ:

```text
SHA-256: 6c02284675e8f14cad0af098c46b3c100e89b327ede3b27276563a78da3888a4
```

`libav*`、libplacebo、libvmaf、Vulkan、NPP、CUDA Runtimeの予期しない動的依存はなかった。

## RTX 5070 Tiエンコード試験

`RTX50fix.md`で96%を記録したものと同じKonomiTV相当条件を使用した。

- 実録画MPEG-TSをFFmpeg `-re`で30秒供給
- MPEG-TS標準入力
- NVDEC `--avhw`
- HEVC Main 10
- 1440x1080
- `--vpp-deband`
- `--vpp-afs preset=default`
- interlace TFF
- lookahead 16
- full-resolution multipass
- AAC 192K / 48kHz / stereo
- low latency
- output thread無効

結果:

```text
FFmpeg exit code: 0
NVEncC exit code: 0
encoded 884 frames, 30.57 fps, 2931.55 kbps, 10.31 MB
RunEncode2: finished.
real=30.00
user=0.81
sys=0.92
cpu=5%
maxrss_kb=467416
voluntary_cs=11269
involuntary_cs=163
```

修正前後:

| 項目 | 修正前 | 修正後 |
|---|---:|---:|
| CPU使用率 | 96% | 5% |
| user CPU時間 | 28.00秒 | 0.81秒 |
| sys CPU時間 | 1.09秒 | 0.92秒 |
| 出力フレーム | 884 | 884 |
| 終了コード | 0 | 0 |

出力確認:

```text
container: MPEG-TS
duration: 29.5295 seconds
video: HEVC Main 10, yuv420p10le, 1440x1080, 30000/1001
audio: AAC LC, 48000 Hz, stereo
```

NVDEC終了、NVENC flush、出力writer終了、CUDA context破棄まで正常に完了した。

## 9.25.1タグ

CPU使用率修正版の配布用バージョンを`9.25.1`とする。

`NVEncCore/rgy_version.h`:

```cpp
#define VER_FILEVERSION             0,9,25,1
#define VER_STR_FILEVERSION          "9.25.1"
#define VER_STR_FILEVERSION_TCHAR _T("9.25.1")
```

`.github/workflows/build_releases.yml`と`.github/workflows/build_packages.yml`は任意タグpushで起動し、タグ名と`VER_STR_FILEVERSION`が一致しなければ失敗する。`9.25.1`タグと上記ヘッダーは一致する。

タグ実行時には次が行われる。

- Windows release build
- Ubuntu 24.04 / CUDA 13.3 Linux package build
- GitHub Releaseへの成果物追加

Linuxパッケージ名:

```text
nvencc_9.25.1_amd64.deb
```

プロジェクト内一時ビルド成果物は`.gitignore`対象のためGitへ保存しない。
