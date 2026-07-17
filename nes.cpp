// Headless backend for the LaiNES core. Implements the two hooks the core calls
// (GUI::new_frame to receive the finished framebuffer, GUI::get_joypad_state to
// read controller input) and wraps load/step/read into the nes:: API.
#include "nes.h"
#include "cpu.hpp"
#include "ppu.hpp"
#include "cartridge.hpp"
#include "apu.hpp"
#include "gui.hpp"
#include <cstring>

static uint32_t g_framebuffer[nes::WIDTH * nes::HEIGHT];
static uint8_t g_buttons[2] = {0, 0};

// --- hooks the LaiNES core calls into ---
namespace GUI {
void new_frame(u32* px) { std::memcpy(g_framebuffer, px, sizeof(g_framebuffer)); }
u8 get_joypad_state(int n) { return g_buttons[n & 1]; }
}

namespace nes {

bool load_file(const char* path) {
    APU::init();
    Cartridge::load(path);
    return Cartridge::loaded();
}

bool load_bytes(const uint8_t* data, int n) {
    APU::init();
    Cartridge::load_data(data, n);
    return Cartridge::loaded();
}

bool loaded() { return Cartridge::loaded(); }
void set_buttons(int player, uint8_t bits) { g_buttons[player & 1] = bits; }
void step_frame() { CPU::run_frame(); }
const uint32_t* pixels() { return g_framebuffer; }
uint8_t ram(uint16_t addr) { return CPU::ram[addr & 0x7FF]; }
void ram_write(uint16_t addr, uint8_t v) { CPU::ram[addr & 0x7FF] = v; }

void save_state(std::vector<uint8_t>& buf) {
    buf.clear();
    CPU::save_state(buf);
    PPU::save_state(buf);
}
void load_state(const std::vector<uint8_t>& buf) {
    const uint8_t* p = buf.data();
    p = CPU::load_state(p);
    p = PPU::load_state(p);
}

}  // namespace nes
