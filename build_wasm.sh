#!/bin/bash
# Build the WebAssembly demo (headless NES + QNet inference / clear-run replay).
# Requires the emsdk environment (emcc). Outputs wasmdist/mario.js + mario.wasm
# with the net weights and clear demo embedded in the WASM FS. The ROM is NOT
# embedded (copyright) -- JS hands it in at runtime.
#
#   source /c/prog/emsdk/emsdk/emsdk_env.sh   # (once per shell)
#   ./build_wasm.sh
set -e
cd "$(dirname "$0")"

NET=warmstarts/bc_clear_x2370_hid512.bin      # embedded as net.bin (HID=512 in wasm.cpp)
DEMO=warmstarts/demo_clear_1-1.bin            # embedded as demo.bin (reaches the flag)
OUT=wasmdist
mkdir -p "$OUT"

LAINES=$(ls third_party/laines/*.cpp third_party/laines/mappers/*.cpp)

emcc -O2 -std=c++17 \
  -I. -Ithird_party/laines/include \
  wasm.cpp mario.cpp nes.cpp autograd.cpp $LAINES \
  --embed-file "$NET@net.bin" \
  --embed-file "$DEMO@demo.bin" \
  -sMODULARIZE=1 -sEXPORT_NAME=createMario -sENVIRONMENT=web \
  -sALLOW_MEMORY_GROWTH=1 \
  -sEXPORTED_FUNCTIONS=_wasm_load_rom,_wasm_reset,_wasm_step_agent,_wasm_step_demo,_wasm_width,_wasm_height,_wasm_mario_x,_wasm_demo_len,_wasm_framebuffer,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=cwrap,ccall,HEAPU8 \
  -o "$OUT/mario.js"

echo "built: $OUT/mario.js + $OUT/mario.wasm"
ls -la "$OUT"
