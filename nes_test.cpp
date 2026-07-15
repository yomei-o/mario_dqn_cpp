// Phase 2 integration check: boot Super Mario Bros headless and confirm the
// emulator responds to input -- hold RIGHT and watch Mario's level x-position
// (SMB RAM: 0x006D = page, 0x0086 = x within page) increase.
//
//   nes_test "Super Mario Bros (JU) (PRG 0).nes"
#include "nes.h"
#include <cstdio>

// SMB level x-position from RAM.
static int mario_x() { return nes::ram(0x006D) * 256 + nes::ram(0x0086); }

// Simple 32-bit framebuffer checksum, to confirm the PPU is actually rendering.
static uint32_t fb_sum() {
    const uint32_t* p = nes::pixels();
    uint32_t s = 0;
    for (int i = 0; i < nes::WIDTH * nes::HEIGHT; ++i) s = s * 31 + p[i];
    return s;
}

static void run(int frames, uint8_t buttons) {
    nes::set_buttons(0, buttons);
    for (int i = 0; i < frames; ++i) nes::step_frame();
}

int main(int argc, char** argv) {
    const char* rom = argc > 1 ? argv[1] : "Super Mario Bros (JU) (PRG 0).nes";
    if (!nes::load_file(rom)) { std::printf("failed to load ROM: %s\n", rom); return 1; }
    std::printf("ROM loaded. booting...\n");

    run(60, 0);                       // let it boot to the title screen
    std::printf("after boot: fb_sum=%08x  x=%d\n", fb_sum(), mario_x());

    run(10, nes::START);              // press START
    run(10, 0);                       // release
    run(40, 0);                       // let 1-1 load
    std::printf("after start: fb_sum=%08x  x=%d\n", fb_sum(), mario_x());

    std::printf("holding RIGHT for 200 frames:\n");
    int x0 = mario_x();
    for (int t = 0; t < 200; t += 20) {
        run(20, nes::RIGHT);
        std::printf("  frame +%3d: x=%d  fb_sum=%08x\n", t + 20, mario_x(), fb_sum());
    }
    int x1 = mario_x();
    std::printf("\nx moved %d -> %d  (%s)\n", x0, x1,
                x1 > x0 ? "OK: Mario ran right!" : "*** no progress ***");
    return x1 > x0 ? 0 : 2;
}
