#!/bin/bash
# Generic parallel best-of-N launcher for a warp-started level trainer.
#   ./train_lv_parallel.sh <exe> <tag> [nworkers]
# e.g. ./train_lv_parallel.sh build/Release/mario13_dqn.exe 13 3
# Worker i -> mario<tag>_best_i.bin, train<tag>_i.log. Warm-starts from the 1-2 net.
set -u
ROM="Super Mario Bros (JU) (PRG 0).nes"
EXE="$1"; TAG="$2"; N="${3:-3}"
WARM="warmstarts/dqn12_1-2_x978_hid512.bin"
EPS=(0.30 0.45 0.60 0.25 0.55 0.70)
LR=(1e-4 1.5e-4 2e-4 8e-5 2e-4 2.5e-4)

echo "launching $N workers for level $TAG ($EXE) ..."
for ((i=0;i<N;i++)); do
    nohup "$EXE" "$ROM" "$i" "mario${TAG}_best_$i.bin" "$WARM" "${EPS[$i]}" "${LR[$i]}" \
        > "train${TAG}_$i.log" 2>&1 &
    echo "  w$i: eps=${EPS[$i]} lr=${LR[$i]} -> mario${TAG}_best_$i.bin (train${TAG}_$i.log) pid $!"
    disown
done
echo "running in background."
