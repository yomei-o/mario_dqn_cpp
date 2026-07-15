#include <cstdio>
#include <cstring>
#include <string>
#include "apu.hpp"
#include "cpu.hpp"
#include "mappers/mapper0.hpp"
#include "mappers/mapper1.hpp"
#include "mappers/mapper2.hpp"
#include "mappers/mapper3.hpp"
#include "mappers/mapper4.hpp"
#include "mappers/mapper5.hpp"
#include "mappers/mapper7.hpp"
#include "mappers/mapper9.hpp"
#include "mappers/mapper10.hpp"
#include "mappers/mapper11.hpp"
#include "mappers/mapper34.hpp"
#include "mappers/mapper66.hpp"
#include "ppu.hpp"
#include "cartridge.hpp"

namespace Cartridge {


Mapper* mapper = nullptr;  // Mapper chip.
std::string currentRomPath;  // Path to currently loaded ROM
u8 currentMapperId = 0;  // Current mapper ID

/* PRG-ROM access */
template <bool wr> u8 access(u16 addr, u8 v)
{
    if (!wr) return mapper->read(addr);
    else     return mapper->write(addr, v);
}
template u8 access<0>(u16, u8); template u8 access<1>(u16, u8);

/* CHR-ROM/RAM access */
template <bool wr> u8 chr_access(u16 addr, u8 v)
{
    if (!wr) {
        mapper->ppu_read_hook(addr);
        return mapper->chr_read(addr);
    }
    else return mapper->chr_write(addr, v);
}
template u8 chr_access<0>(u16, u8); template u8 chr_access<1>(u16, u8);

void signal_scanline(int scanline)
{
    mapper->signal_scanline(scanline);
}

void ppu_write_hook(u16 index, u8 v)
{
    mapper->ppu_write_hook(index, v);
}

/* Build the mapper from an iNES image (takes ownership of `rom`) and power on. */
static void init_from_rom(u8* rom)
{
    int mapperNum = (rom[7] & 0xF0) | (rom[6] >> 4);
    currentMapperId = mapperNum;  // Store mapper ID
    if (loaded())
    {
        delete mapper;
        mapper = nullptr;
    }
    switch (mapperNum)
    {
        case 0:  mapper = new Mapper0(rom); break;
        case 1:  mapper = new Mapper1(rom); break;
        case 2:  mapper = new Mapper2(rom); break;
        case 3:  mapper = new Mapper3(rom); break;
        case 4:  mapper = new Mapper4(rom); break;
        case 5:  mapper = new Mapper5(rom); break;
        case 7:  mapper = new Mapper7(rom); break;
        case 9:  mapper = new Mapper9(rom); break;
        case 10: mapper = new Mapper10(rom); break;
        case 11: mapper = new Mapper11(rom); break;
        case 34: mapper = new Mapper34(rom); break;
        case 66: mapper = new Mapper66(rom); break;
        default:
            fprintf(stderr, "mapper %d not supported\n", mapperNum);
            delete[] rom;
            return;
    }

    PPU::reset();  // Reset PPU first so it's ready when CPU::power() ticks
    APU::reset();
    CPU::power();  // power() calls INT<RESET>() which ticks the PPU
}

/* Load the ROM from a file (native). */
void load(const char* fileName)
{
    FILE* f = fopen(fileName, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open ROM file: %s\n", fileName);
        return;
    }
    fseek(f, 0, SEEK_END);
    int size = ftell(f);
    fseek(f, 0, SEEK_SET);
    u8* rom = new u8[size];
    if (fread(rom, size, 1, f) != 1) { fclose(f); delete[] rom; return; }
    fclose(f);
    init_from_rom(rom);
    currentRomPath = fileName;
}

/* Load the ROM from a memory buffer (WASM/JS hand off the bytes). */
void load_data(const u8* data, int size)
{
    u8* rom = new u8[size];
    memcpy(rom, data, size);
    init_from_rom(rom);
    currentRomPath = "";
}

void reset()
{
    if (!currentRomPath.empty()) {
        load(currentRomPath.c_str());
    }
}

bool loaded()
{
    return mapper != nullptr;
}

// Run mapper expansion audio up to given cycle count
void run_mapper_audio(int elapsed)
{
    if (mapper && mapper->has_audio())
        mapper->run_audio(elapsed);
}

// End audio frame for mapper expansion audio
void end_mapper_audio_frame(int elapsed)
{
    if (mapper && mapper->has_audio())
        mapper->end_audio_frame(elapsed);
}

// Check if mapper IRQ should be active at given time
bool check_mapper_irq(int elapsed)
{
    if (mapper)
        return mapper->check_irq(elapsed);
    return false;
}

std::string get_rom_path()
{
    return currentRomPath;
}

u8 get_mapper_id()
{
    return currentMapperId;
}

Mapper* get_mapper()
{
    return mapper;
}

bool handles_expansion_addr(u16 addr)
{
    if (!mapper) return false;
    return mapper->handles_expansion_addr(addr);
}


}
