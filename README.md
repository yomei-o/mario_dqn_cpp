# mario_dqn_cpp — 依存ゼロC++で作るDQN（→ 本物のNESでスーパーマリオ）

*日本語 | [English](README.en.md)*

**外部ライブラリを使わず（C++標準ライブラリ・CPUのみ）自作の自動微分エンジンから DQN
（Deep Q-Network）** を組み上げ、最終的に **本物のNESエミュレータでスーパーマリオをプレイさせる**
ことを目指すプロジェクト。DQNは [othello_alphazero_cpp](https://github.com/yomei-o/othello_alphazero_cpp)
の AlphaZero（方策＋探索）と対になる **価値ベース・オフポリシー** の強化学習。

autograd は [mini-yolov5-cpp](https://github.com/yomei-o/mini-yolov5-cpp) の自作エンジンを流用。

## ロードマップ

| Phase | 内容 | 状態 |
|------|------|------|
| **1** | **CartPole で DQN コアを実証**（リプレイ・ターゲットネット・Double DQN・ε-greedy・勾配クリップ・Adam） | ✅ 完了 |
| 2 | **LaiNES**（C++ NESエミュ）を組み込み、`step/観測/報酬/done` のRL APIを実装 | 予定 |
| 3 | DQN × スーパーマリオ（**RAM特徴量**：マリオのx/y・敵・タイル）。到達距離が伸びるのを学習 | 予定 |
| 4 | （余力）生ピクセル版 | 任意 |

> **設計判断**：CPU＋自作autogradで「生ピクセルから本物マリオを1面クリア」は非現実的（Atari DQNは
> 当時GPUで数日）。そこで環境は**本物のNESエミュ**、状態は**RAM特徴量**（本物のRAMを読む）にして
> “本物マリオが現実的な時間で上達する”を両立する。

## Phase 1（このリポジトリの現状）

CartPole-v1（棒立て）を DQN で解く最小実装。**同じエージェントをそのままマリオに接続**するための土台。

```
autograd.h/.cpp   自作autograd（mini-yolov5から流用）＋ Huber loss を追加
cartpole.h        CartPole-v1 環境（Gym準拠の物理）
qnet.h            QネットMLP ＋ Adam ＋ ターゲットネット複製
replay.h          経験リプレイバッファ（リング）
main.cpp          学習ループ（Double DQN・ε減衰・勾配クリップ・最良ネット保存・評価）
```

### ビルド & 実行

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
build/Release/dqn.exe        # CartPole を学習し、貪欲方策の評価を表示
```

Linux/macなら `cmake -S . -B build && cmake --build build`。依存はコンパイラのみ。

### DQNの要点（学習ポイント）

- **経験リプレイ**：遷移 `(s, a, r, s', done)` を貯めてランダムにミニバッチ学習（相関を断つ）
- **ターゲットネット**：TD目標の計算に固定コピーを使い発散を防ぐ（AlphaZeroで作った `copy_from` を流用）
- **Double DQN**：次手の選択はオンライン網、評価はターゲット網（過大評価バイアス低減）
- **TD損失**：取った行動のQだけを `r + γ·maxQ'` に近づける（Huber損失）
- **ε-greedy**：探索率を 1.0→0.05 に減衰
- **安定化**：勾配クリップ（global norm）＋ Adam ＋ 最良ネットのチェックポイント

## スーパーマリオについて（Phase 2以降）

- 環境は **本物のNESエミュレータ（LaiNES）** を組み込む（`gym-super-mario-bros` の中核と同じ系統）
- **ROMは著作権物**。各自が正規に用意すること（このリポジトリのコードにROMは含めない）
- 報酬は `Δ(マリオのx位置) − 時間ペナルティ − 死亡ペナルティ + ゴールボーナス`

## シリーズ

C++/CPUだけでAIを一から作るシリーズ。関連：
[mini-yolov5-cpp](https://github.com/yomei-o/mini-yolov5-cpp) /
[othello_alphazero_cpp](https://github.com/yomei-o/othello_alphazero_cpp) /
[nanoGPT-cpp](https://github.com/yomei-o/nanoGPT-cpp) /
[nanochat-cpp](https://github.com/yomei-o/nanochat-cpp) /
[lecun1989-cpp](https://github.com/yomei-o/lecun1989-cpp)
