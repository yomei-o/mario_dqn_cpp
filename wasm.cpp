// WebAssembly bindings: run the headless NES + a trained QNet (live greedy
// inference) OR replay a captured clear-run action sequence, in the browser.
// The clear-run replay reliably reaches the flagpole; live mode shows the agent
// playing. ROM bytes are handed in from JS (the copyrighted ROM is NOT shipped).
//
// Build: see build_wasm.sh. Exposes C functions consumed from JS via cwrap.
#include "nes.h"
#include "mario.h"
#include "qnet.h"
#include "autograd.h"
#include <emscripten.h>
#include <vector>
#include <cstdio>
#include <cstdint>

using namespace ag;
using namespace rl;

static mario::Env g_env;
static QNet*      g_net = nullptr;
static std::vector<uint8_t> g_demo;      // clear-run action bytes
static size_t     g_demo_i = 0;
static std::vector<uint8_t> g_rgba;      // WIDTH*HEIGHT*4 (RGBA for canvas)
static bool       g_ready = false;

static const int S = mario::Env::STATE_DIM;
static const int A = mario::N_ACTIONS;
static const int NET_HID = 512;          // matches the embedded net (bc net, hidden=512)

static int argmax_row(const std::vector<float>& q, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (q[i] > q[best]) best = i;
    return best;
}

// Read an action-sequence file embedded in the WASM FS (u32 count + count bytes).
static void load_actions(const char* path, std::vector<uint8_t>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return;
    uint32_t n = 0;
    if (std::fread(&n, 4, 1, f) == 1) { out.resize(n); if (n) std::fread(out.data(), 1, n, f); }
    std::fclose(f);
}

extern "C" {

// Load the ROM (from JS), lazily load the embedded net + clear demo, reset.
EMSCRIPTEN_KEEPALIVE
int wasm_load_rom(const uint8_t* rom, int n) {
    if (!g_env.init_bytes(rom, n)) return 0;
    if (!g_net) {
        g_net = new QNet(S, A, NET_HID);
        if (!g_net->load("net.bin")) g_net->load_expand("net.bin");
        load_actions("demo.bin", g_demo);
    }
    g_env.reset();
    g_demo_i = 0;
    g_rgba.assign((size_t)nes::WIDTH * nes::HEIGHT * 4, 255);
    g_ready = true;
    return 1;
}

EMSCRIPTEN_KEEPALIVE
void wasm_reset() { if (g_ready) { g_env.reset(); g_demo_i = 0; } }

// One step of the LIVE agent (greedy net). Returns 0=running, 1=won(flag), 2=done(other).
EMSCRIPTEN_KEEPALIVE
int wasm_step_agent() {
    if (!g_ready) return 2;
    auto x = Tensor::from(g_env.observation(), {1, S}, false);
    int a = argmax_row(g_net->forward(x).data(), A);
    bool done; g_env.step(a, done);
    return done ? (g_env.won() ? 1 : 2) : 0;
}

// One step of the CLEAR-RUN replay. Returns 0=running, 1=won(flag), 2=done(other).
EMSCRIPTEN_KEEPALIVE
int wasm_step_demo() {
    if (!g_ready) return 2;
    int a = (g_demo_i < g_demo.size()) ? (int)g_demo[g_demo_i++] : 0;
    bool done; g_env.step(a, done);
    return done ? (g_env.won() ? 1 : 2) : 0;
}

EMSCRIPTEN_KEEPALIVE int wasm_width()   { return nes::WIDTH; }
EMSCRIPTEN_KEEPALIVE int wasm_height()  { return nes::HEIGHT; }
EMSCRIPTEN_KEEPALIVE int wasm_mario_x() { return g_env.mario_x(); }
EMSCRIPTEN_KEEPALIVE int wasm_demo_len(){ return (int)g_demo.size(); }

// Convert the NES framebuffer (0xXXRRGGBB) to RGBA and return a pointer to it.
EMSCRIPTEN_KEEPALIVE
uint8_t* wasm_framebuffer() {
    const uint32_t* px = nes::pixels();
    int n = nes::WIDTH * nes::HEIGHT;
    for (int i = 0; i < n; ++i) {
        uint32_t p = px[i];
        g_rgba[i * 4 + 0] = (p >> 16) & 0xFF;   // R
        g_rgba[i * 4 + 1] = (p >> 8)  & 0xFF;   // G
        g_rgba[i * 4 + 2] =  p        & 0xFF;   // B
        g_rgba[i * 4 + 3] = 255;                // A
    }
    return g_rgba.data();
}

}  // extern "C"
