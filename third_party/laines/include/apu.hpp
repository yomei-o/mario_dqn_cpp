#pragma once
// Headless APU stub: no audio (not needed for RL). Satisfies the CPU/cartridge
// call sites; APU IRQ is disabled (SMB gameplay does not depend on it).
#include "common.hpp"
namespace APU {
template <bool write> u8 access(int elapsed, u16 addr, u8 v = 0, bool is_put_cycle = false);
void run_frame(int elapsed);
void end_buffer_frame(int elapsed);
void reset();
void init();
bool check_irq(int elapsed);
}
