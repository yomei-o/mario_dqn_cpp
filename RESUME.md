# RESUME — 明日以降ここから再開

このファイルは、作業を中断→再開するときの「現在地と次の一手」。Claude（や自分）が
読めば続きを進められるようにしてある。シリーズ全体は各リポジトリのREADME参照。

## このプロジェクトのゴール

**依存最小のC++で DQN を自作autogradから組み、本物のNESエミュ(LaiNES)でスーパーマリオを
プレイ。最終的に NES+DQN を WASM(Emscripten) 化してブラウザ(HTML+JS canvas)で表示する。**
CPUのみ（この環境ではGPU不可：M2 Mac→UTM→Windows、CUDA/Metal使えず）。

## フェーズ状況

- **Phase 1 ✅ CartPoleでDQNコア実証** — greedy 500/500 で SOLVED。
  （リプレイ・ターゲットネット・Double DQN・Huber・ε-greedy・勾配クリップ・Adam）
- **Phase 2 ✅ ヘッドレスNES統合** — LaiNES(BSD-2)コアをvendorしSDL/音/GUI除去。
  `nes.h/.cpp` の frame駆動API。**MSVCでビルド可**（GNU case範囲をif/elseに書換え済み）。
  実機SMB起動→右入力でマリオ前進を確認。
- **Phase 3 🚧 DQN×マリオ（RAM特徴量）** — `mario.h/.cpp`（環境）+ `mario_dqn.cpp`（学習）。
  環境動作確認済み。チェックポイントは **greedy評価で最良保存**。
  **2026-07 更新: 観測を強化した（STATE_DIM 19→37）。**
    1. **敵相対x座標のバグ修正** — 従来は低バイト単純減算でページ境界(256px)を跨ぐと符号反転し、
       前方のクリボーが「後方」と誤認されていた。ページ考慮のレベル絶対座標に修正
       （敵x = `0x006E+i`*256 + `0x0087+i`）。envtestで接近が `dx=1.0→0.08` と正しく見える。
    2. **前方タイル窓を観測に追加** — SMB背景タイルRAM `$0500` からマリオ前方 6列×3行を読み、
       平地=`.##`／穴=`...`／土管=`###` として地形を数値化。土管・穴が「見える」ようになった。
    3. **停滞シェーピング** — 60ステップ前進なしで軽ペナルティ＋エピソード終了（学習効率化）。
    4. **train_freq=4** — Atari-DQN流に4環境ステップごとに1勾配更新。NESエミュ(~57 step/s)が
       ボトルネックなので学習は本質的に長時間（20万stepで約1時間）。→ **バックグラウンドで長回し**。
  診断ツール `tilescan` でタイル配置とマリオ位置をダンプ可能。
  **結果(2026-07): 約24万step学習で greedy が x=709 に到達＝旧baseline(681)を更新**
  （`mario_best.bin`=この37次元ネット, `web/run.bin`=greedy再生を録画済み）。eps=0.05まで減衰後
  greedy評価はノイズが大きい(reset時の「操作可能まで進める」ループがフレーム単位でずれるため
  296〜709とばらつく)＝真のスキルは~600前後で、更なる収束は下記の残タスク次第。
- **Phase 4 ⬜ WASM化してブラウザ表示** — 未着手（設計は下記）。

## ビルド & 実行（MSVC / Visual Studio）

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release

build/Release/dqn.exe                                   # Phase1: CartPole
build/Release/nes_test.exe "Super Mario Bros (JU) (PRG 0).nes"   # Phase2: 起動テスト
build/Release/mario_dqn.exe envtest "Super Mario Bros (JU) (PRG 0).nes"  # 環境チェック(観測を表示)
build/Release/mario_dqn.exe tilescan "Super Mario Bros (JU) (PRG 0).nes" # タイルRAM/位置ダンプ(診断)
build/Release/mario_dqn.exe "Super Mario Bros (JU) (PRG 0).nes"          # Phase3: 学習(長時間・要バックグラウンド)
```

- clang / g++(w64devkit) / emcc でも同一コードが通る。**MinGWのバイナリはアンチウイルス
  誤検知するので、配布用は MSVC ビルドを使う**（ユーザー方針）。
- 学習済み重み: `mario_best.bin`（best_x を更新するたび保存）。

## ファイル地図

```
autograd.h/.cpp   自作autograd（mini-yolov5流用）+ Huber
qnet.h            QネットMLP + Adam + copy_from + save/load
replay.h          経験リプレイ
cartpole.h        Phase1環境 / main.cpp = CartPole学習
nes.h/.cpp        ヘッドレスNES API（load_file/load_bytes/set_buttons/step_frame/pixels/ram）
third_party/laines/  vendorしたLaiNESコア（de-GNU化済み、SDL/音源除去、mapper24削除）
mario.h/.cpp      SMB 1-1 環境（RAM特徴量・行動・報酬・reset）/ mario_dqn.cpp = 学習/録画
web/index.html    録画(run.bin)をcanvasで再生するビューア
web/serve.py      Python簡易HTTPサーバ / server.cpp = cpp-httplib版(third_party/httplib.h)
```

## 結果をブラウザで見る（実装済み）

`mario_dqn record ROM mario_best.bin web/run.bin` で greedy プレイを録画 → `python web/serve.py`
または `server.exe` で配信 → ブラウザで `web/index.html`。**Phase4(WASMライブ)の前段**（今は
「ネイティブ録画→ブラウザ再生」）。現状greedyはクリボー(x312)で力尽きる＝学習収束が先決。

## 次の一手（Phase 3 収束）

観測強化(地形タイル窓＋敵座標修正)と停滞シェーピングは実装済み(上記)。残るは**収束**：
1. **学習を長く回す** — `eps_end=0.05`, `eps_decay_steps=200000`, `train_freq=4` 設定済み。
   NESエミュ律速で ~57 step/s なので 20万step≈1時間。バックグラウンド実行し、ログの
   `GREEDY_x`/`best_greedy` で真の到達距離を追う（25エピソードごとにgreedy評価）。
   `mario_best.bin` は best_greedy 更新のたび自動保存されるので途中停止しても最良ネットは残る。
2. **まだ弱ければ**: 地形窓の縦方向をジャンプ中でも接地面が見えるよう固定行アンカーにする、
   タイル窓を広げる、報酬に穴落下の早期ペナルティ、lr/γ/target_sync 調整。
3. framebuffer ダウンサンプルを観測に足す（CNNなしでも粗い占有格子は効く）。

SMB RAM（実装済み/参考）: x=0x6D*256+0x86, y=0xCE, x速度=0x57(signed), floatState=0x1D,
状態=0x0E(0x06/0x0B=死), lives=0x75A, 敵active=0x0F..0x13, **敵x=0x6E+i(page)*256+0x87+i(low)**,
敵y=0xCF.., 縦page=0xB5, **背景タイルバッファ=0x0500..0x069F(2画面, 13行×16列, page*0xD0+row*16+col)**。

## 次の一手（Phase 4 WASM）

emsdk は `C:\prog\emsdk`。方針:
- **推論のみ**をブラウザで（学習はネイティブで済ませ `mario_best.bin` を読む）。
- emcc で `nes.cpp + third_party/laines/*.cpp + qnet(推論) + mario観測` をWASM化。
  `EMSCRIPTEN_KEEPALIVE` で `reset/step/agent_act/framebuffer_ptr/ram/load_rom(bytes)` を公開。
- **ROMはJSからバイト列で** `nes::load_bytes` に渡す（`Cartridge::load_data` 実装済み）。
- HTML+JS: `<canvas>`、`requestAnimationFrame` で毎フレーム `agent_act()`→`step()`→
  `pixels()` を ImageData で canvas に描画。重みは fetch でロード。
- 注意: スレッド不要な経路（マリオはMLP、convスレッド未使用）なので pthread/COOP-COEP は不要。

## メモ / 落とし穴

- LaiNES原文の GNU拡張 `case a ... b:` は全てif/elseに書換え済み（MSVC対応）。iso646
  (and/or/not) は `/permissive-`。ppu.cppに`<cstring>`追加済み。mapper24(VRC6)は外部音源
  依存で削除（SMBはmapper0なので不要）。
- ROM `Super Mario Bros (JU) (PRG 0).nes` はユーザー提供。著作権物ゆえコード側では扱わず、
  ルート配置のものを各実行ファイルが読む。
