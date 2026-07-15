#pragma once
// Minimal shim for the blargg Nes_Snd_Emu types that LaiNES headers reference.
// We stub the APU, so only these typedefs are needed to compile the core.
#include <cstdint>
typedef unsigned cpu_addr_t;
typedef int      cpu_time_t;
typedef short    blip_sample_t;
