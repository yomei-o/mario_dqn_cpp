# RESUME — 明日以降ここから再開

このファイルは、作業を中断→再開するときの「現在地と次の一手」。Claude（や自分）が
読めば続きを進められるようにしてある。シリーズ全体は各リポジトリのREADME参照。

---
## 🔴 引き継ぎメモ（2026-07-16, 別マシン/別Claudeへ）

**目標: 1-1を先頭から通しでgreedyクリア（フラッグポール x≈3160）＋道中で高得点・キノコ取得。**

### 2026-07-16 追加変更（CPU制御 + 報酬を「得点優先」に転換）
- **CPU使用率スロットル**: 学習が1コアを100%占有して他作業を邪魔しないよう、プロセスに
  デューティ比制御を追加。環境変数 **`MARIO_CPU`**（0〜1, 既定 **0.8**）で「稼働=起きている
  wall時間の割合」を指定。`1.0`で無効。累積busy/slept会計で自己補正、Windowsは`timeBeginPeriod(1)`
  でsleep精度確保（`winmm`リンク）。実測: 既定0.8で1コアあたり約73–78%（安全側に少し低め）。
  例: `MARIO_CPU=0.5 build/Release/mario_dqn.exe "$ROM"`。`train_parallel.sh`の各ワーカーにも効く。
- **報酬を得点優先に再設計**（`mario.cpp` `Env::step`）: 従来は前進距離(~2000/レベル)が支配的で
  得点(~70)を無視し敵に突っ込んで詰まっていた。今回:
  - 前進 dense を **重み1.0→0.3**（クリップ±15）に縮小＝右進は「軽い後押し」に格下げ。
  - **得点増を主役**に: `min(80, 4.0*Δscore)`（踏み+10→+40 / コイン+20→+80cap。デコード値は小さい整数）。
  - パワーアップ取得 **+40**（旧+30）、被弾（パワー喪失）**−15**（旧−12, 死亡−50より軽い）。
  - **近接ジャンプ整形**: 空中(floatState=1)で前方の敵が踏み込み窓(rel_x∈[-8,56], rel_y∈[-8,40])に
    いると **+2/step**（ユーザの狙い「近づいたらジャンプして踏む」を誘導。farmingは時間ペナルティ
    -0.1/step＋STALL_LIMITで抑制）。終端(クリア+100/死亡−50)はクリップ外で従来通り。
  - ⚠️ 報酬スケールが変わったので **warm-startネットは学習初期に一時劣化**し得る（既知の性質,
    下記の壁を参照）。新目標で再収束させる想定。`gendemo`のデモも新方策が出たら作り直す。

### 2026-07-16 ★突破: greedy 2017 → 2228（避け報酬＋全速で「穴＋クリボー」を越えた）
アイテムは諦め「1-1のgreedy到達距離を伸ばす」に舵を切り、**初めてwarm-startの2017をgreedyで超えた（x=2228, metric2248）**。効いた要因（ユーザーの洞察がそのまま効いた）:
- **敵の"避け(飛び越え)"を報酬化し、踏みより高く**（避け+25 / 踏み+12）。小マリオは踏むより避ける方が安全なので、そちらへ誘導。判定はEnv内（敵が前方→背後に回った＋生存＋空中）＝結果ベースで farm 不可。踏みは`Env::stomped()`（潰れ状態の遅延を考慮した緩め検出）。
- **前進報酬は全速のまま**（速度上限=アンチラッシュを一度入れたら到達が~1441に頭打ち→撤去）。**穴を越えて着地点のクリボーを飛び越すにはダッシュジャンプの距離が要る**ため。
- **~1524の壁の正体**（ユーザーが動画で特定）: 穴を大ジャンプで越えた**着地点にクリボー**。早めに踏み切って飛び越すのが正解（＝避け報酬が誘導）。
- 高速リセット（`reset_from(0)`は起動後スナップショットを`load_state`＝タイトル画面ブート省略で試行ほぼ倍）。純progress、warm-start 2017（`load_expand` 90×256→160×256）、隠れ層256、全域カリキュラム、アイテム報酬なし。
- **保存**: 突破ネットは `mario_best_run2.bin` に自動保存（metric>2087で保存）。手動バックアップ `mario_break2228.bin`。※2228は最初一度出したが再起動時に`rm mario_best_run*`で消した失敗あり→**run/latestは消さない**。
- 現状: 2228で新たなプラトー（その先にまた壁）。`web/run.bin`に2228走行を録画済み（ブラウザで可視化）。次は2228超え or マルチレベル(1-2)へ。

### 2026-07-16 結論: アイテム統合は未解決（この後2228突破。上記参照）
**ゴール**: 1つのgreedy方策で「通し走行＋キノコ/コイン/踏み」を両立。**結果: 到達不能と判断、2017に戻した。**
- 試した手（すべてgreedyの通し走行を数分で~434へ崩壊させた）: ①得点優先報酬 ②全域/早期カリキュラム
  ③観測に？ブロック格子追加(STATE_DIM 90→160) ④キノコ/踏みお手本の**模倣シード**(demo_item/demo_stomp.bin, `finditem`で総当たり取得) ⑤状態依存報酬(小マリオはキノコ+50) ⑥**net2widerで512に拡張**(2017を関数保存, `load_widen`) ⑦**土台凍結アダプター**(旧重み勾配0で新ユニットのみ学習)。
- ⑦でも崩壊: 重みは凍結でも**新ユニットの出力"残差"がQ値のargmaxを乗っ取る**（重み凍結≠出力凍結）。
- **診断で分かった核心**: 2017方策はfine-tuningに極めて脆く、アイテム方向へ僅かに動かすと通し走行が壊れる。
  さらに早期限定カリキュラムは後半経験をリプレイから枯渇させ忘却を加速。CPU・自作DQN・小MLPの容量/学習
  ダイナミクス上の**本質的限界**。「アイテムは探索/デモでは出せるが、1つのgreedyに束ねられない」。
- **最終状態**: `mario_best.bin`(=90次元256隠れの2017通し走行ネット, greedy x=2017/score70)をチャンピオンとして確定。
  ブラウザ/README GIFは「スタート→旗 x=3161」の通しクリア（探索キャプチャの決定的再生）。
- **残った資産（コードは保持）**: 模倣ライブラリ(`demos/`+`load_actions`), スキルお手本キャプチャ(ITEM/STOMP/FLAG),
  `finditem`総当たり, `recorddemo`, `load_widen`(net2wider), 土台凍結(`freeze_base_grads`), CPUスロットル(`MARIO_CPU`),
  クロスプラットフォーム(winmm=WIN32)。将来アイテム統合を再挑戦するなら「残差の大きさ制限/正則化」「優先度リプレイで
  前半経験を保証」「もっと大きなネットをゼロから長時間」あたりが次の候補。

### 2026-07-16 追加: 勝利キャプチャ + フロンティア自動前進 + 通しクリア録画
- **勝利キャプチャ**: 学習中にエピソードが**フラッグポール到達(is_win)**したら、その**起動(x=0)からの完全な行動列**を
  `demo_win_<seed>.bin` に保存。内訳 = 開始チェックポイントまでのデモ接頭辞（`Env::checkpoint_demo_len(k)`）＋
  そのエピソードの自力行動。**より自力な勝ち(接頭辞が小さい方)だけ更新**するので、学習が進むほど自力区間が伸び、
  最終的に接頭辞0＝**スタートから通しクリア**に近づく。`mario.h`に`won()`/`checkpoint_demo_len()`/`demo_actions()`を追加。
- **`recorddemo` モード**: 保存した行動列を`env.reset()`から再生し`web/run.bin`へ録画（決定的なので勝ちデモは
  通しクリア動画になる）。`build/Release/mario_dqn.exe recorddemo "$ROM" demo_win_0.bin web/run.bin`。
- **フロンティア前進(=カリキュラム拡張)**: 勝ちデモ(`demo_win_*.bin`)は旗まで届く完全デモなので、これを`demo.bin`に
  昇格すると`build_curriculum`のチェックポイントが**最終盤(例 x=2178..3087)まで拡張**され、未攻略の終盤を集中練習できる。
  手順: ワーカー停止→`cp demo.bin demo_2017.bin; cp demo_win_0.bin demo.bin`→`snaptest`で確認→再起動。
- 運用: 3ワーカー全速(`MARIO_CPU=1.0`)で並列学習しつつ、①`demo_win_*.bin`変化を監視して最も自力な勝ちを
  `recorddemo`で自動録画、②`server.exe`(:8080)＋`web/index.html`でブラウザ再生。より自力な勝ちが出るたび動画が更新される。


**現在地**: greedy x=2017（1-1の約2/3）が確定済みスキル。`mario_best.bin` は **89次元** の
このネット（コードは今 STATE_DIM=90 だが、`load`系は全て `load_expand` で89→90を吸収するので
そのまま読める。学習すると90次元で保存される）。

**いま分かっている壁（重要）**:
- カリキュラム（セーブステートで難所手前から開始）＋得点/パワー報酬＋warm-startを入れたが、
  **warm-start方策が学習開始直後に劣化**（先頭からのgreedyが 2017→~400〜1400 に落ちる）。
  原因は報酬スケール激変＋探索でQがズレてMLPが早期区間を忘却（catastrophic forgetting）。
- 穏やかfine-tuning（eps0.1/lr1e-4/カリキュラム漸減）で初回eval~1439まで改善したが単一設定では頭打ち。
- **チェックポイントからはゴール到達実績あり**（train_max_x=3161を1回記録）＝セグメントは解ける。
  課題は「全区間を1つのgreedy方策に束ねる」こと。

**いま走らせていたもの**: `./train_parallel.sh` = 8ワーカーの多様sweep（eps0.05→0.4, lr5e-5→2.5e-4,
カリキュラム0.3→0.7）。各`mario_best_<seed>.bin`に保存、最後に`eval`でbest選抜→`mario_best.bin`。
※このプロセス群と`mario_best_<i>.bin`/`demo.bin`はローカル限定（gitignore）。**別マシンでは再生成が必要**。

**別マシンでの再開手順**:
1. ROMをルートに置く（著作権物・非コミット）。`cmake -S . -B build -G "Visual Studio 17 2022" -A x64 && cmake --build build --config Release`
2. `build/Release/mario_dqn.exe gendemo "$ROM" mario_best.bin demo.bin`（カリキュラム用デモ再生成・必須）
3. `build/Release/mario_dqn.exe snaptest "$ROM"` でセーブステート正当性を確認（OKが出ること）
4. `./train_parallel.sh` で並列sweep開始 → 進捗は `train_*.log` の `best_greedy`
5. 良いものが出たら `eval` で選抜、`record` で `web/run.bin` 録画、`mario_best.bin` を更新してコミット

**次の一手（2017超え/クリアのための案, 効きそうな順）**:
1. **チャンピオン共有の集団学習**: sweepの全体ベストを定期的に各ワーカーへ移住（elitism）。
2. **優先度/バランス・リプレイ**: 先頭区間の遷移を必ず一定割合サンプル→忘却を防いで通し方策を維持。
3. **アイテム位置の観測追加**（キノコ/フラワーを能動的に取りに行く）: `load_expand`で方策を保ったまま特徴追加可。
4. 報酬の再調整（得点/パワーは効くが、報酬スケール変更が忘却を招くので控えめに）、lr/γ調整。
5. デモを更新して前進: 2017超えのネットが出たら`gendemo`でデモを作り直し、フロンティアを前へ。
---

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
  **2026-07 更新: 観測を大幅強化（STATE_DIM 19→37→89）。**
    1. **敵相対x座標のバグ修正** — 従来は低バイト単純減算でページ境界(256px)を跨ぐと符号反転し、
       前方のクリボーが「後方」と誤認されていた。ページ考慮のレベル絶対座標に修正
       （敵x = `0x006E+i`*256 + `0x0087+i`）。envtestで接近が `dx=1.0→0.08` と正しく見える。
    2. **前方地形の占有格子を観測に追加** — SMB背景タイルRAM `$0500` からマリオ前方
       **10列×7行(画面固定行6..12)** を読取。列を上→下で読むと縦プロファイルになり、
       平地=`.....##`／穴=全0／土管=下から積み上がる`#`。**行をマリオ相対でなく画面固定にした**のが要点で、
       ジャンプ中も前方の地面/着地点/土管高さが見え続ける（＝踏み切り・着地を学べる）。
       視野も6列→10列(~160px)に拡大して高い土管/穴の早期認識に対応。
    3. **停滞シェーピング** — 90ステップ前進なしで軽ペナルティ＋エピソード終了（土管ジャンプ試行の余地を確保）。
    4. **train_freq=4** — Atari-DQN流に4環境ステップごとに1勾配更新。NESエミュ(~57 step/s)が
       ボトルネックなので学習は本質的に長時間（20万stepで約1時間）。→ **バックグラウンドで長回し**。
    5. **探索確保** — eps下限 0.1、`eps_decay_steps`=25万（土管越えの連続A保持timingを見つけるため）。
  診断ツール `tilescan`/`dettest`/`snaptest`/`scoretest` あり。envtestは観測の地形格子も表示。
  **経過(2026-07)**:
    - 相対3行窓(37次元)版 → greedy x=709（旧baseline 681更新）。ただし土管クラスター(x≈650〜900)で頭打ち。
    - **固定行占有格子(89次元)版 → greedy x=2017 に到達（1-1の約2/3, ゴール~3160）**。土管群・穴地帯を突破。
    - **得点＋パワーアップ＋カリキュラム(90次元)版で完全クリアを狙って学習中**（下記）。
  ✓**評価ノイズの正体（判明）**: 学習中の `GREEDY_x` のばらつきは reset非決定ではなく
    **ネットが学習中で毎回別物**だから。`dettest`で **エミュは同一入力列に対し完全決定的**と確認済み
    （3試行とも同一x）。best_greedy(=保存済みネット)を `record`/`eval` で回すと毎回完全再現する。

  **⑥ 完全クリア狙い（2026-07, 実装済み・学習中）**:
    - **報酬シェーピング**: 得点増を加点（コイン+200,踏み+100 等 → 道中で高得点）。
      **パワーアップ取得=+30**（キノコ/フラワーは被弾で死なず生存価値大）、パワー喪失=−12（死亡−50より軽い）。
      終端はクリップ除外（クリア+100/死亡−50）。SMB RAM: 得点`$07DD-$07E2`(6桁), コイン`$075E`, パワー`$0756`。
    - **観測にパワー状態を追加**（小/スーパー/ファイア, index 89）→ STATE_DIM 89→90。
      `QNet::load_expand` で旧89次元の重みを引き継ぎwarm-start（新特徴の重みは0初期化＝初期方策は不変）。
    - **セーブステート型カリキュラム**: `nes::save_state/load_state`（CPU+PPU全状態をメモリに退避／復元。
      mapper0不変・APUは音のみで除外）を新規実装。`snaptest`で再生と完全一致を検証済み。
      デモ(=best netのgreedy行動列, `gendemo`で生成→`demo.bin`)の後半に16チェックポイントを起動時1パスで作成し、
      学習の70%をそこから開始＝**未攻略の最終区間を集中反復**（瞬時復元で高速）。残り30%は先頭から。
    - **チェックポイント選択=距離+得点の複合指標**（同距離なら高得点/パワー保持の方策を保存, 距離優先は維持）。
    - **並列学習(best-of-N)**: エミュがグローバル・シングルトンなので1プロセス=1コア。`mario_dqn [ROM] [seed] [out.bin]`
      で別seed/別出力のワーカーを複数プロセス起動→`train_parallel.sh`が全コア活用しbest-of-N選抜（`eval`で採点）。
      （さらに効率化する場合は「チャンピオン共有の集団学習」を追加可能=未実装）
- **Phase 4 ⬜ WASM化してブラウザ表示** — 未着手（設計は下記）。

## ビルド & 実行（MSVC / Visual Studio）

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release

build/Release/dqn.exe                                   # Phase1: CartPole
build/Release/nes_test.exe "$ROM"                       # Phase2: 起動テスト
build/Release/mario_dqn.exe envtest  "$ROM"             # 環境チェック(観測を表示)
build/Release/mario_dqn.exe tilescan "$ROM"             # タイルRAM/位置ダンプ(診断)
build/Release/mario_dqn.exe dettest  "$ROM"             # エミュ決定性チェック
build/Release/mario_dqn.exe scoretest "$ROM" mario_best.bin   # 得点/コイン/パワーRAM確認
build/Release/mario_dqn.exe gendemo  "$ROM" mario_best.bin demo.bin  # カリキュラム用デモ生成(先に必須)
build/Release/mario_dqn.exe snaptest "$ROM"             # セーブステート正当性検証(demo.bin必須)
build/Release/mario_dqn.exe "$ROM"                      # Phase3: 学習(長時間・要バックグラウンド)
build/Release/mario_dqn.exe eval "$ROM" mario_best_0.bin      # チェックポイント採点(best-of-N選抜用)
./train_parallel.sh 6                                   # 並列学習(6ワーカー=6コア, demo.bin必須)
build/Release/mario_dqn.exe record  "$ROM" mario_best.bin web/run.bin  # ブラウザ再生用に録画
```
（`$ROM` = `"Super Mario Bros (JU) (PRG 0).nes"`。カリキュラム/並列学習の前に `gendemo` で `demo.bin` を作る）

- clang / g++(w64devkit) / emcc でも同一コードが通る。**MinGWのバイナリはアンチウイルス
  誤検知するので、配布用は MSVC ビルドを使う**（ユーザー方針）。
- 学習済み重み: `mario_best.bin`（複合指標=距離+得点 を更新するたび保存, 90次元）。
  `demo.bin` はカリキュラム用デモ（`gendemo`で生成, .gitignore対象＝再生成可）。

## ファイル地図

```
autograd.h/.cpp   自作autograd（mini-yolov5流用）+ Huber
qnet.h            QネットMLP + Adam + copy_from + save/load + load_expand(次元拡張warm-start)
replay.h          経験リプレイ
cartpole.h        Phase1環境 / main.cpp = CartPole学習
nes.h/.cpp        ヘッドレスNES API（+ save_state/load_state = CPU+PPU全状態の退避/復元）
third_party/laines/  vendorしたLaiNESコア（de-GNU化済, SDL/音源除去, cpu/ppuにsave_state追加）
mario.h/.cpp      SMB 1-1 環境（観測/報酬/reset + カリキュラム: demo再生・チェックポイント瞬時復元）
mario_dqn.cpp     学習(warm-start/カリキュラム/並列)・診断(envtest/tilescan/dettest/snaptest/
                  scoretest)・gendemo/record/eval
train_parallel.sh best-of-N並列学習ランチャ（別プロセス=別コア）
web/index.html    録画(run.bin)をcanvasで再生するビューア
web/serve.py      Python簡易HTTPサーバ / server.cpp = cpp-httplib版(third_party/httplib.h)
```

## 結果をブラウザで見る（実装済み）

`mario_dqn record ROM mario_best.bin web/run.bin` で greedy プレイを録画 → `python web/serve.py`
または `server.exe` で配信 → ブラウザで `web/index.html`。**Phase4(WASMライブ)の前段**（今は
「ネイティブ録画→ブラウザ再生」）。相対窓版のgreedy(x=709)を録画済み。占有格子版は再学習後に録り直す。

## 次の一手（Phase 3 収束）

観測強化(固定行占有格子＋視野拡大＋敵座標修正)・停滞シェーピング・探索確保は実装済み(上記①〜⑤)。
残るは**収束**と、なお土管/穴を越えられない場合の追い込み：
1. **学習を長く回す** — `eps_end=0.1`, `eps_decay_steps=250000`, `train_freq=4` 設定済み。
   NESエミュ律速で ~57 step/s なので 25万step≈1.2時間。バックグラウンド実行し、ログの
   `GREEDY_x`/`best_greedy` で真の到達距離を追う（25エピソードごとにgreedy評価）。
   `mario_best.bin` は best_greedy 更新のたび自動保存されるので途中停止しても最良ネットは残る。
2. **まだ土管/穴で止まるなら（追い込み）**:
   - 報酬シェーピング — 穴落下の早期ペナルティ、土管/穴を越えた瞬間のボーナス。
   - **カリキュラム(最有効)** — 難所手前のNES状態スナップショットからエピソードを開始し集中反復。
     LaiNESにセーブ/ロード状態(CPU/PPU/RAM一括)を足す必要あり＝実装量中。
   - framebufferダウンサンプルを観測に追加、lr/γ/target_sync/frame_skip 調整。

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
