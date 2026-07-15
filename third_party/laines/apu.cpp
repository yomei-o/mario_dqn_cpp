// Headless APU stub (no sound). See include/apu.hpp.
#include "apu.hpp"
namespace APU {
template <bool write> u8 access(int, u16 addr, u8 v, bool) {
    // $4015 status reads return 0 (no active channels / no frame IRQ); writes ignored.
    return write ? v : 0;
}
template u8 access<0>(int, u16, u8, bool);
template u8 access<1>(int, u16, u8, bool);
void run_frame(int) {}
void end_buffer_frame(int) {}
void reset() {}
void init() {}
bool check_irq(int) { return false; }   // APU never raises IRQ in headless mode
}
