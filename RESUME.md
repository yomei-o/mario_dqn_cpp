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
  環境動作確認済み。学習中に **ゴール(x=3161)到達を確認**（探索中の1回）。
  → **次にやること: 学習を収束させる**（下記）。
- **Phase 4 ⬜ WASM化してブラウザ表示** — 未着手（設計は下記）。

## ビルド & 実行（MSVC / Visual Studio）

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release

build/Release/dqn.exe                                   # Phase1: CartPole
build/Release/nes_test.exe "Super Mario Bros (JU) (PRG 0).nes"   # Phase2: 起動テスト
build/Release/mario_dqn.exe envtest "Super Mario Bros (JU) (PRG 0).nes"  # 環境チェック
build/Release/mario_dqn.exe "Super Mario Bros (JU) (PRG 0).nes"          # Phase3: 学習
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
mario.h/.cpp      SMB 1-1 環境（RAM特徴量・行動・報酬・reset）/ mario_dqn.cpp = 学習
```

## 次の一手（Phase 3 収束）

いまの学習は探索中にゴール到達するが、方策として安定していない（avg50_x が ~900 前後で
頭打ち気味）。改善候補（効きそうな順）:
1. **学習を長く回す** — `episodes` を増やし eps を 0.05 まで、`eps_decay_steps` を伸ばす。
   最良ネットのgreedy評価を定期実行して真の到達距離を測る（CartPoleでやったのと同じ）。
2. **地形が観測に無いのが弱点** — マリオは土管/穴を「見て」いない（x速度の停滞で間接的に
   しか分からない）。SMBのタイルRAM(0x0500-0x069F)から前方の小さなタイル窓を観測に足すと
   障害物回避が学べる。または framebuffer を粗くダウンサンプルして足す。
3. **報酬シェーピング** — 前進報酬に加え、穴落下の早期ペナルティ、停滞ペナルティ等。
4. ハイパラ: lr、target_sync、frame_skip(いまは4)、γ の調整。

SMB RAM（実装済み/参考）: x=0x6D*256+0x86, y=0xCE, x速度=0x57(signed), floatState=0x1D,
状態=0x0E(0x06/0x0B=死), lives=0x75A, 敵active=0x0F..0x13, 敵x=0x87.., 敵y=0xCF.., 縦page=0xB5。

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
