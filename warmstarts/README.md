# Warm-start checkpoints & demos (1-1 & 1-2)

Reusable nets/demos captured across experiments. Observation is `STATE_DIM=160`
(mario.h). Load a net into a `QNet(S, A, HID)` — **HID must match the file** (see
suffix). Tools accept these as `[weights]`:

```
build/Release/mario_dqn.exe eval   "ROM.nes" warmstarts/<file>     # greedy x + score
build/Release/mario_dqn.exe record "ROM.nes" warmstarts/<file> web/run.bin
# warm-start training: set HID in mario_dqn.cpp to match, put the file at mario_best.bin
```

## Nets

| file | HID | greedy x | notes |
|------|-----|----------|-------|
| `dqn_progress_2228_hid256.bin`     | 256 | **2228** | Best DQN (progress reward). Small Mario, rushes, ignores items. Farthest value-based greedy. |
| `bc_clear_x2370_hid512.bin`        | **512** | **2370** | Behavior-cloning net (imitates clear demos). **Only net to pass the x≈2226 step.** Needs HID=512. |
| `score_mushroom_x434_hid256.bin`   | 256 | 434 | Score-reward net trained from scratch. **Grabs the mushroom (power=1, score 120)** — big Mario. Short reach. |
| `warmprog_small_x1525_hid256.bin`  | 256 | 1525 | Warm(mushroom)→progress run. Dropped the mushroom, rushed (small Mario). Stops ~x1513 nav spot. |
| `dqn12_1-2_x978_hid512.bin`        | **512** | **978** (in 1-2) | **World 1-2** net (`mario12`, 6 actions incl. NOOP). Warm-started from `bc_clear_x2370` then DQN in 1-2 with ratchet reward + wait-friendly stall. Reaches x≈978 (past the "pit + Buzzy Beetle" timing spot, using NOOP to wait). Load with the **6-action** `mario12` env, not the 5-action 1-1 env. |
| `dqn13_1-3_x627_hid512.bin`        | **512** | **627** (in 1-3) | **World 1-3** net (`mario13`, 6 actions). Warp-started (RAM level-warp), warm-started from the 1-2 net, DQN ~30 min. Reaches x≈627 into the athletic platform section. 6-action `mario13` env. |

## Demos (action sequences; for curriculum & behavior cloning)

| file | notes |
|------|-------|
| `demo_clear_1-1.bin` | Full 1-1 clear (~902 actions, reaches the flag x≈3160). Used as `demo.bin` for curriculum checkpoints and as BC training data. |

## Key findings (why these differ)
- Reward shapes playstyle: **progress → rush/skip items (2228)**; **score → grab mushroom (434)**.
- Value-based DQN greedy plateaus at **2228** (the x≈2226 step); every reward/DQfD/capacity/EMA variant hit the same wall.
- **Behavior cloning is the only method that passed the step (2370)** — imitating clear demos directly optimizes the greedy policy, sidestepping DQN's instability.
- x≈1513 is a navigation spot (no jump / precise) — big Mario does NOT help there.
