// WebAssembly bindings for World 1-2: run the headless NES + the trained 1-2 QNet
// (live greedy inference) in the browser. SEPARATE from the 1-1 bindings (wasm.cpp)
// but shares the level-agnostic core. The 1-2 start is reached by replaying an
// embedded 1-1 clear and riding the transition (done inside Env::reset, cached so
// only the first reset is slow). ROM bytes are handed in from JS (not shipped).
//
// Build: see build_wasm12.sh. NOTE: the 1-2 net has 6 actions (incl. NOOP).
#include "nes.h"
#include "mario12.h"
#include "qnet.h"
#include "autograd.h"
#include <emscripten.h>
#include <vector>
#include <cstdio>
#include <cstdint>

using namespace ag;
using namespace rl;

static mario12::Env g_env;
static QNet*        g_net = nullptr;
static std::vector<uint8_t> g_rgba;      // WIDTH*HEIGHT*4 (RGBA for canvas)
static bool         g_ready = false;

static const int S = mario12::Env::STATE_DIM;
static const int A = mario12::N_ACTIONS;   // 6 (incl. NOOP)
static const int NET_HID = 512;

static int argmax_row(const std::vector<float>& q, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (q[i] > q[best]) best = i;
    return best;
}

extern "C" {

// Load the ROM (from JS), lazily load the embedded 1-2 net, load the embedded 1-1
// clear (needed to reach the 1-2 start), then reset (first reset builds+caches the
// 1-2 start; it replays the 1-1 clear so it takes a moment).
EMSCRIPTEN_KEEPALIVE
int wasm_load_rom(const uint8_t* rom, int n) {
    if (!g_env.init_bytes(rom, n)) return 0;
    if (!g_net) {
        g_net = new QNet(S, A, NET_HID);
        g_net->load("net12.bin");
    }
    g_env.load_entry_demo("demo11.bin");   // 1-1 clear -> reach 1-2 (also invalidates the cached start)
    g_env.reset();
    g_rgba.assign((size_t)nes::WIDTH * nes::HEIGHT * 4, 255);
    g_ready = true;
    return 1;
}

EMSCRIPTEN_KEEPALIVE
void wasm_reset() { if (g_ready) g_env.reset(); }

// One step of the LIVE agent (greedy net). Returns 0=running, 1=won(flag), 2=done(other).
EMSCRIPTEN_KEEPALIVE
int wasm_step_agent() {
    if (!g_ready) return 2;
    auto x = Tensor::from(g_env.observation(), {1, S}, false);
    int a = argmax_row(g_net->forward(x).data(), A);
    bool done; g_env.step(a, done);
    return done ? (g_env.won() ? 1 : 2) : 0;
}

EMSCRIPTEN_KEEPALIVE int wasm_width()   { return nes::WIDTH; }
EMSCRIPTEN_KEEPALIVE int wasm_height()  { return nes::HEIGHT; }
EMSCRIPTEN_KEEPALIVE int wasm_mario_x() { return g_env.mario_x(); }
EMSCRIPTEN_KEEPALIVE int wasm_area()    { return g_env.area(); }

EMSCRIPTEN_KEEPALIVE
uint8_t* wasm_framebuffer() {
    const uint32_t* px = nes::pixels();
    int n = nes::WIDTH * nes::HEIGHT;
    for (int i = 0; i < n; ++i) {
        uint32_t p = px[i];
        g_rgba[i * 4 + 0] = (p >> 16) & 0xFF;
        g_rgba[i * 4 + 1] = (p >> 8)  & 0xFF;
        g_rgba[i * 4 + 2] =  p        & 0xFF;
        g_rgba[i * 4 + 3] = 255;
    }
    return g_rgba.data();
}

}  // extern "C"
