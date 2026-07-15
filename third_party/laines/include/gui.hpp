#pragma once
// Headless replacement for LaiNES's SDL GUI. The emulator core calls
// GUI::new_frame() to emit a finished 256x240 framebuffer and
// GUI::get_joypad_state() to read controller input; our backend (nes.cpp)
// captures the frame and injects the RL agent's / player's buttons.
#include "common.hpp"
namespace GUI {
void new_frame(u32* pixels);      // called by PPU at end of frame
u8 get_joypad_state(int n);       // controller n button bitmask
}
