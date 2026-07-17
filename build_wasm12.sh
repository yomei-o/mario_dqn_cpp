#!/bin/bash
# Build the World 1-2 WebAssembly demo (headless NES + 1-2 QNet live inference).
# Separate from build_wasm.sh (1-1). Embeds the 1-2 net + the 1-1 clear (used to
# reach the 1-2 start). ROM is NOT embedded (copyright) -- JS hands it in.
#
#   source /c/prog/emsdk/emsdk/emsdk_env.sh   # (once per shell)
#   ./build_wasm12.sh
set -e
cd "$(dirname "$0")"

NET=warmstarts/dqn12_1-2_x978_hid512.bin      # embedded as net12.bin (6 actions, HID=512)
DEMO=warmstarts/demo_clear_1-1.bin            # embedded as demo11.bin (clears 1-1 -> reach 1-2)
OUT=wasmdist
mkdir -p "$OUT"

LAINES=$(ls third_party/laines/*.cpp third_party/laines/mappers/*.cpp)

emcc -O2 -std=c++17 \
  -I. -Ithird_party/laines/include \
  wasm12.cpp mario12.cpp nes.cpp autograd.cpp $LAINES \
  --embed-file "$NET@net12.bin" \
  --embed-file "$DEMO@demo11.bin" \
  -sMODULARIZE=1 -sEXPORT_NAME=createMario12 -sENVIRONMENT=web \
  -sALLOW_MEMORY_GROWTH=1 \
  -sEXPORTED_FUNCTIONS=_wasm_load_rom,_wasm_reset,_wasm_step_agent,_wasm_width,_wasm_height,_wasm_mario_x,_wasm_area,_wasm_framebuffer,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=cwrap,ccall,HEAPU8 \
  -o "$OUT/mario12.js"

echo "built: $OUT/mario12.js + $OUT/mario12.wasm"
ls -la "$OUT"/mario12.*
