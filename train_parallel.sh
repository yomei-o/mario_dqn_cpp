#!/bin/bash
# Parallel training: the NES core is a global singleton, so each worker needs its
# own process (= its own CPU core). We run a DIVERSE best-of-N sweep -- workers
# span the preserve<->explore spectrum (gentle low-eps/low-lr keeps the warm-
# started policy; aggressive high-eps explores past the frontier) -- all warm-
# starting from mario_best.bin, then pick the winning checkpoint.
#
#   ./train_parallel.sh          # 8 preset workers (needs demo.bin + mario_best.bin)
#
# Each worker i logs to train_i.log and saves mario_best_i.bin.
set -u
ROM="Super Mario Bros (JU) (PRG 0).nes"
EXE="build/Release/mario_dqn.exe"

# worker configs: "seed eps lr curriculum_prob"  (preserve --> explore)
CONFIGS=(
  "0 0.05 5e-5 0.30"
  "1 0.10 1e-4 0.40"
  "2 0.10 5e-5 0.50"
  "3 0.20 1e-4 0.50"
  "4 0.20 1.5e-4 0.60"
  "5 0.30 2e-4 0.60"
  "6 0.30 2.5e-4 0.70"
  "7 0.40 2e-4 0.70"
)

echo "launching ${#CONFIGS[@]} diverse workers ..."
pids=()
for cfg in "${CONFIGS[@]}"; do
    read -r seed eps lr cp <<< "$cfg"
    "$EXE" "$ROM" "$seed" "mario_best_$seed.bin" "$eps" "$lr" "$cp" > "train_$seed.log" 2>&1 &
    pids+=($!)
    echo "  worker $seed: eps=$eps lr=$lr cp=$cp -> mario_best_$seed.bin (train_$seed.log) pid $!"
done

echo "waiting (Ctrl+C to stop; checkpoints save as they improve) ..."
for p in "${pids[@]}"; do wait "$p"; done

echo "=== evaluating checkpoints ==="
best_metric=-1; best_file=""
for cfg in "${CONFIGS[@]}"; do
    read -r seed _ <<< "$cfg"
    f="mario_best_$seed.bin"
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
