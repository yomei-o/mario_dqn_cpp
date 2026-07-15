#!/bin/bash
# Parallel training: the NES core is a global singleton, so each training worker
# needs its own process (= its own CPU core). We launch N workers with distinct
# seeds and checkpoint paths (best-of-N), all warm-starting from the current
# mario_best.bin, then pick the best resulting checkpoint.
#
#   ./train_parallel.sh [N]      # N workers (default: CPU cores - 2)
#
# Requires: build/Release/mario_dqn.exe and demo.bin present.
set -u
ROM="Super Mario Bros (JU) (PRG 0).nes"
EXE="build/Release/mario_dqn.exe"
CORES=$(nproc 2>/dev/null || echo 4)
N=${1:-$(( CORES > 3 ? CORES - 2 : 2 ))}

echo "launching $N parallel workers (cores=$CORES) ..."
pids=()
for i in $(seq 0 $((N-1))); do
    "$EXE" "$ROM" "$i" "mario_best_$i.bin" > "train_$i.log" 2>&1 &
    pids+=($!)
    echo "  worker $i -> mario_best_$i.bin (train_$i.log) pid $!"
done

echo "waiting for workers to finish (Ctrl+C to stop; checkpoints are saved as they improve) ..."
for p in "${pids[@]}"; do wait "$p"; done

echo "=== evaluating checkpoints ==="
best_metric=-1; best_file=""
for i in $(seq 0 $((N-1))); do
    f="mario_best_$i.bin"
    [ -f "$f" ] || continue
    line=$("$EXE" eval "$ROM" "$f")
    echo "$line"
    m=$(echo "$line" | grep -oE 'metric=[0-9]+' | grep -oE '[0-9]+')
    if [ -n "$m" ] && [ "$m" -gt "$best_metric" ]; then best_metric=$m; best_file=$f; fi
done

if [ -n "$best_file" ]; then
    cp "$best_file" mario_best.bin
    echo "=== best: $best_file (metric=$best_metric) -> copied to mario_best.bin ==="
else
    echo "no checkpoints produced"
fi
