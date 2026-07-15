# mario_dqn_cpp — DQN from scratch in C++ (→ Super Mario on a real NES)

*[日本語](README.md) | English*

Build **DQN (Deep Q-Network) from scratch on a self-made autograd engine (C++ standard
library, CPU only)**, and ultimately make it **play Super Mario Bros on a real NES
emulator**. DQN is the **value-based, off-policy** counterpart to the AlphaZero
(policy + search) of [othello_alphazero_cpp](https://github.com/yomei-o/othello_alphazero_cpp).

The autograd is reused from [mini-yolov5-cpp](https://github.com/yomei-o/mini-yolov5-cpp).

## Roadmap

| Phase | What | Status |
|------|------|--------|
| **1** | **Prove the DQN core on CartPole** (replay, target net, Double DQN, ε-greedy, grad clipping, Adam) | ✅ done |
| **2** | Embed **LaiNES** (C++ NES emulator); headless `step/obs/reward/done` API | ✅ done |
| 3 | DQN × Super Mario (**RAM features**: Mario x/y, enemies, tiles); watch progress distance grow | planned |
| 4 | Compile NES + DQN to **WASM (Emscripten)**; play in the browser (HTML + JS canvas) | planned |

## Phase 2 (headless NES) — done

`third_party/laines/` vendors the **LaiNES** (BSD-2) core, **stripped of SDL/audio/GUI**
(`GUI::new_frame` = grab framebuffer, `GUI::get_joypad_state` = inject input, APU stubbed).
`nes.h/.cpp` exposes a WASM-friendly frame-driven API (`load_file`/`load_bytes`, `set_buttons`,
`step_frame`, `pixels`, `ram`). It **builds with plain MSVC/Visual Studio** — LaiNES's GNU
case-ranges were rewritten to if/else, so no MinGW is needed (avoids antivirus false positives);
clang/g++/emcc compile the same code. Verified: real SMB boots and Mario walks right (RAM
x-position increases) under `nes_test`.

> **Design call**: "raw pixels → clear a real Mario level on CPU + a from-scratch autograd"
> is not realistic (Atari DQN took days on GPUs back then). So the environment is a **real
> NES emulator** while the state is **RAM features** (read from the real RAM) — that makes
> "real Mario improving in a reasonable time" achievable.

## Phase 1 (current state)

A minimal DQN that solves CartPole-v1 — the scaffold for plugging the *same agent* into Mario.

```
autograd.h/.cpp   self-made autograd (reused from mini-yolov5) + Huber loss
cartpole.h        CartPole-v1 environment (Gym-accurate physics)
qnet.h            Q-net MLP + Adam + target-network copy
replay.h          experience replay buffer (ring)
main.cpp          training loop (Double DQN, ε decay, grad clip, best-net checkpoint, eval)
```

### Build & run

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
build/Release/dqn.exe        # trains CartPole, prints greedy-policy evaluation
```

Linux/mac: `cmake -S . -B build && cmake --build build`. Compiler only.

Result: solves CartPole (greedy eval reaches the 500-step cap).

### DQN essentials

- **Experience replay**: store `(s, a, r, s', done)`, train on random minibatches (breaks correlation)
- **Target network**: a frozen copy for the TD target to prevent divergence (reuses AlphaZero's `copy_from`)
- **Double DQN**: online net picks the next action, target net scores it (less overestimation)
- **TD loss**: move only the taken action's Q toward `r + γ·maxQ'` (Huber loss)
- **ε-greedy**: exploration decays 1.0 → 0.05
- **Stability**: global-norm gradient clipping + Adam + best-net checkpointing

## About Super Mario (Phase 2+)

- Embeds a **real NES emulator (LaiNES)** (same lineage as the core of `gym-super-mario-bros`)
- **ROMs are copyrighted** — supply your own; no ROM is included in the code
- Reward: `Δ(Mario x-position) − time penalty − death penalty + goal bonus`

## Series

Building AI from scratch in C++/CPU. Related:
[mini-yolov5-cpp](https://github.com/yomei-o/mini-yolov5-cpp) /
[othello_alphazero_cpp](https://github.com/yomei-o/othello_alphazero_cpp) /
[nanoGPT-cpp](https://github.com/yomei-o/nanoGPT-cpp) /
[nanochat-cpp](https://github.com/yomei-o/nanochat-cpp) /
[lecun1989-cpp](https://github.com/yomei-o/lecun1989-cpp)
