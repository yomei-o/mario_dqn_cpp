// WebAssembly bindings for World 1-3 (athletic). Runs the headless NES + the 1-3
// QNet (live greedy) in the browser. The 1-3 start is reached by an in-core RAM
// level-warp (Env::reset -> death-reload), so NO demo is embedded -- only the net
// (net13.bin). Rebuild with a better net to swap the agent in place. ROM from JS.
#include "nes.h"
#include "mario13.h"
#include "qnet.h"
#include "autograd.h"
#include <emscripten.h>
#include <vector>
#include <cstdio>
#include <cstdint>

using namespace ag;
using namespace rl;

static mario13::Env g_env;
static QNet*        g_net = nullptr;
static std::vector<uint8_t> g_rgba;
static bool         g_ready = false;
static const int S = mario13::Env::STATE_DIM;
static const int A = mario13::Env::ACTIONS;
static const int NET_HID = 512;

static int argmax_row(const std::vector<float>& q, int n) {
    int best = 0; for (int i = 1; i < n; ++i) if (q[i] > q[best]) best = i; return best;
}

extern "C" {

EMSCRIPTEN_KEEPALIVE
int wasm_load_rom(const uint8_t* rom, int n) {
    if (!g_env.init_bytes(rom, n)) return 0;
    if (!g_net) { g_net = new QNet(S, A, NET_HID); g_net->load("net13.bin"); }
    g_env.reset();                 // warps to 1-3 (cached after first call)
    g_rgba.assign((size_t)nes::WIDTH * nes::HEIGHT * 4, 255);
    g_ready = true;
    return 1;
}

EMSCRIPTEN_KEEPALIVE
void wasm_reset() { if (g_ready) g_env.reset(); }

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
        g_rgba[i*4+0]=(p>>16)&0xFF; g_rgba[i*4+1]=(p>>8)&0xFF; g_rgba[i*4+2]=p&0xFF; g_rgba[i*4+3]=255;
    }
    return g_rgba.data();
}

}  // extern "C"
