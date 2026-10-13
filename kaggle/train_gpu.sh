#!/bin/bash
# Train the 1-1 agent on a Kaggle GPU session (T4) -- or any Linux box with nvcc.
# Drive it from your machine through kbridge (https://github.com/yomei-o/kaggle_server_cpp):
#
#   curl -s -X POST localhost:8787/job -H 'Content-Type: application/json' \
#        -d '{"name":"mario","cmd":"bash mario_dqn_cpp/kaggle/train_gpu.sh"}'
#
# Kaggle has 4 vCPUs: 3 actor processes (the NES emulator is the bottleneck) + the
# GPU learner. Results land in mario_dqn_cpp/: mario_gpu_clear.bin (the net whose
# greedy run clears 1-1), mario_gpu_best.bin, gpu_run/greedy_clear.bin, run.bin (video).
#   ACTORS=3 MINUTES=120 ARCH=75 bash kaggle/train_gpu.sh
set -e
cd "$(dirname "$0")/.."
ROM="Super Mario Bros (JU) (PRG 0).nes"
ACTORS="${ACTORS:-3}"
MINUTES="${MINUTES:-120}"
ARCH="${ARCH:-75}"              # T4 = sm_75 (P100 = 60)
DEMO="${DEMO:-warmstarts/demo_robust_1-1_3act.bin}"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH" > /dev/null
cmake --build build --target mario_gpu -j"$(nproc)" 2>&1 | grep -iE "error|warning: unused" || true
./build/mario_gpu parity | tail -2
./build/mario_gpu bench
./build/mario_gpu train "$ROM" "$DEMO" "$ACTORS" "$MINUTES" ${WARM:-} || true
if [ -f mario_gpu_clear.bin ]; then
    ./build/mario_gpu eval "$ROM" mario_gpu_clear.bin
    ./build/mario_gpu record "$ROM" mario_gpu_clear.bin run.bin
fi
