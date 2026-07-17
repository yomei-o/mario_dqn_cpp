#!/bin/bash
# Parallel best-of-N training for World 1-2. Each worker = its own process/core
# (the NES core is a global singleton). Workers span preserve<->explore: gentle
# low-eps keeps the transferred 1-1 run skill; aggressive high-eps discovers 1-2's
# pits/pipes. All warm-start from a 1-1 net (identical obs). 6 workers -> leaves
# cores free so the machine stays usable over RDP.
#
#   ./train12_parallel.sh
#
# Worker i -> mario12_best_i.bin, train12_i.log, and demo_clear_1-2_i.bin on a flag win.
set -u
ROM="Super Mario Bros (JU) (PRG 0).nes"
EXE="build/Release/mario12_dqn.exe"
WARM="warmstarts/bc_clear_x2370_hid512.bin"

# "seed eps lr"  (preserve --> explore)
CONFIGS=(
  "0 0.20 8e-5"
  "1 0.30 1e-4"
  "2 0.35 1.2e-4"
  "3 0.45 1.5e-4"
  "4 0.55 2e-4"
  "5 0.70 2.5e-4"
)

echo "launching ${#CONFIGS[@]} 1-2 workers ..."
for cfg in "${CONFIGS[@]}"; do
    read -r seed eps lr <<< "$cfg"
    nohup "$EXE" "$ROM" "$seed" "mario12_best_$seed.bin" "$WARM" "$eps" "$lr" > "train12_$seed.log" 2>&1 &
    echo "  worker $seed: eps=$eps lr=$lr -> mario12_best_$seed.bin (train12_$seed.log) pid $!"
    disown
done
echo "running in background. tail train12_*.log to watch; kill %jobs to stop."
