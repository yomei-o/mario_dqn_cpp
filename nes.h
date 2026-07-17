#pragma once
#include <vector>
// Headless NES front-end around the vendored LaiNES core (BSD). Exposes a clean,
// WASM-friendly, frame-driven API: load a ROM, inject controller buttons, step
// one frame, then read the 256x240 framebuffer and the CPU RAM. No SDL, no audio,
// no threads -- exactly what we need to (a) train a DQN natively and (b) later
// compile the same core to WebAssembly and drive it from JavaScript.
#include <cstdint>

namespace nes {

// NES controller bit layout (matches LaiNES joypad shift order: A first).
enum Button {
    A = 0x01, B = 0x02, SELECT = 0x04, START = 0x08,
    UP = 0x10, DOWN = 0x20, LEFT = 0x40, RIGHT = 0x80
};

constexpr int WIDTH = 256, HEIGHT = 240;

bool load_file(const char* path);            // load iNES ROM from disk (native)
bool load_bytes(const uint8_t* data, int n); // load ROM from memory (WASM/JS)
bool loaded();
void set_buttons(int player, uint8_t bits);  // player 0/1, OR of Button flags
void step_frame();                            // advance exactly one video frame
const uint32_t* pixels();                     // WIDTH*HEIGHT, LaiNES 0xXXRRGGBB
uint8_t ram(uint16_t addr);                   // read CPU RAM $0000-$07FF
void ram_write(uint16_t addr, uint8_t v);     // write CPU RAM (e.g. SMB level-warp poke)

// Fast in-memory save/restore of the full emulator state (CPU + PPU; mapper0 is
// immutable, APU is sound-only). Deterministic at frame boundaries: restoring a
// snapshot reproduces the exact trajectory. Used for curriculum checkpoints.
void save_state(std::vector<uint8_t>& buf);
void load_state(const std::vector<uint8_t>& buf);

}  // namespace nes
