# GLib静的リンク混在の修正確認

## Codex session id

`01a00a5c-88c5-7553-992f-a15f1313a34c`

## 修正内容

対象リポジトリは `/home/oomugi413/gitbuild/NVEnc`。

`meson.options` の `libass_static` の既定値を `true` から `false` に変更した。Linuxでlibassを静的リンクしたい場合は、従来どおり `-Dlibass_static=true` を明示して選択できる。

`meson.build` には、この既定値を共有リンクにする理由をコメントとして追加した。

従来の既定設定では、静的libassの依存関係から実行ファイルへ静的GLibが入り、共有FFmpegライブラリが使用する共有GLib/GObjectと同一プロセス内で混在していた。このシンボルの混在によりGLibのグローバル状態が一致せず、起動時に次のようなアサーションで異常終了していた。

```text
g_hash_table_lookup: assertion 'hash_table != NULL'
g_hash_table_insert_internal: assertion 'hash_table != NULL'
g_type_register_static: assertion failed: (static_quark_type_flags)
```

libassを共有リンクにすることで、FFmpeg/GObjectを含む共有GLib経路に統一した。GLib本体やFFmpeg本体の変更は行っていない。

## ビルド確認

- Mesonビルドディレクトリ: `/tmp/nvencc-oomugi-glibfix.EiEpQv`
- `libass static: false` を確認
- `meson compile -C /tmp/nvencc-oomugi-glibfix.EiEpQv nvencc`: `222/222` 完了
- 生成版: `NVEnc (x64) 9.32.1 (r4085)` / CUDA 13.2
- `nvencc --version`: 終了コード0、stderr空
- `ldd`で `libgobject-2.0.so.0`、`libglib-2.0.so.0`、`libass.so.9` を確認
- 実行ファイル内に静的GLibの主要シンボルはなし

## エンコード確認

入力は `/mnt/recording/20260816_test4K.ts`（3840x2160、HEVC Main 10、59.94fps、AAC 48kHz stereo）。FFmpegから `-re` で10秒相当をMPEG-TS標準入力へ送り、生成したNVEncCで実行した。

出力ログは `/tmp/nvencc-oomugi-enc.2dRLQS` に保存した。3経路ともFFmpeg/NVEncCの終了コードは0で、`RunEncode2: finished.` と `encoded 551 frames` を確認した。

| 経路 | 結果 |
|---|---|
| 通常エンコード | 成功。3840x2160、551 frames、60.74 fps |
| `--vpp-colorspace matrix=bt2020nc:bt709,colorprim=bt2020:bt709,transfer=bt2020-10:bt709,range=limited:limited` | 成功。3840x2160、551 frames、62.97 fps |
| `--vpp-libplacebo-tonemapping src_csp=hlg,dst_csp=sdr` + `--vpp-resize libplacebo-ewa-lanczos --output-res 1920x1080` | 成功。1920x1080、551 frames、57.06 fps |

3経路のstderr/stdoutを `g_hash_table`、`g_type_register`、`static_quark_type_flags`、GLib assertionで検索したが、一致はなかった。今回の修正後ビルドではGLibエラーは再現していない。
