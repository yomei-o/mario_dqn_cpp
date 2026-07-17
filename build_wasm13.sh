#!/bin/bash
# Build the World 1-3 WASM demo. NET is overridable so the agent can be SWAPPED
# later without touching the page:  NET=warmstarts/<trained>.bin ./build_wasm13.sh
# (placeholder default = the 1-2 warm net until a trained 1-3 net exists).
set -e; cd "$(dirname "$0")"
NET="${NET:-warmstarts/dqn12_1-2_x978_hid512.bin}"     # embedded as net13.bin
OUT=wasmdist; mkdir -p "$OUT"
LAINES=$(ls third_party/laines/*.cpp third_party/laines/mappers/*.cpp)
emcc -O2 -std=c++17 -I. -Ithird_party/laines/include \
  wasm13.cpp mario13.cpp nes.cpp autograd.cpp $LAINES \
  --embed-file "$NET@net13.bin" \
  -sMODULARIZE=1 -sEXPORT_NAME=createMario13 -sENVIRONMENT=web -sALLOW_MEMORY_GROWTH=1 \
  -sEXPORTED_FUNCTIONS=_wasm_load_rom,_wasm_reset,_wasm_step_agent,_wasm_width,_wasm_height,_wasm_mario_x,_wasm_area,_wasm_framebuffer,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=cwrap,ccall,HEAPU8 \
  -o "$OUT/mario13.js"
echo "built $OUT/mario13.js (+.wasm) with NET=$NET"
