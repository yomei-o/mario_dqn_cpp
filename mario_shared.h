#pragma once
// Shared, LEVEL-AGNOSTIC core for the Super Mario Bros RL envs. Both the 1-1 env
// (mario::Env) and the 1-2 env (mario12::Env) build on this so the observation
// layout, action set, RAM decoding, boot sequence and terrain grid stay IDENTICAL
// across levels (a net trained on one level's features is meaningful on another).
// Level-SPECIFIC policy -- where you start, what counts as a win, and the reward
// shaping (e.g. 1-2's "enter the exit pipe" bonus) -- lives in each level's env,
// NOT here. Header-only (inline) so it links into every build with no extra .cpp.
#include "nes.h"
#include <vector>
#include <cstdint>
#include <algorithm>

namespace smb {

// Discrete action set (right-biased, like gym's SIMPLE_MOVEMENT subset).
// B = run/fire, A = jump. No DOWN: the right-biased set can only enter a
// HORIZONTAL pipe by walking right into it -- which for 1-2 is exactly the level
// exit, so "entered a new area" is an unambiguous good signal (no vertical
// bonus/warp pipes are reachable without DOWN).
enum { A_RIGHT, A_RIGHT_A, A_RIGHT_B, A_RIGHT_AB, A_A, N_ACTIONS };

// Terrain occupancy grid sampled ahead of Mario (fixed screen rows, see build_obs).
constexpr int TILE_COLS = 10;
constexpr int TILE_ROW0 = 6;
constexpr int TILE_ROWS = 7;
constexpr int TILE_N    = TILE_COLS * TILE_ROWS;
// obs: [0..3] mario, [4..18] 5 enemies, [19..18+TILE_N] terrain, [19+TILE_N] power,
// [20+TILE_N..] ?-block grid.
constexpr int STATE_DIM = 19 + TILE_N + 1 + TILE_N;
constexpr int FRAME_SKIP = 4;

// --- Super Mario Bros RAM map (well-documented addresses) -------------------
inline int    RAM(uint16_t a)     { return nes::ram(a); }
inline int    level_x()           { return RAM(0x006D) * 256 + RAM(0x0086); }  // x in level
inline int    player_y()          { return RAM(0x00CE); }                       // y on screen
inline int8_t x_speed()           { return (int8_t)RAM(0x0057); }               // signed
inline int    player_state()      { return RAM(0x000E); }                       // 0x08 normal, 0x06/0x0B dead
inline int    float_state()       { return RAM(0x001D); }                       // 0 ground,1 air,3 climbing(flag)
inline int    lives()             { return RAM(0x075A); }
inline int    vert_page()         { return RAM(0x00B5); }                        // >1 => fell into a pit
inline int    power_state()       { return RAM(0x0756); }                        // 0 small,1 super,2 fire
inline int    coin_count()        { return RAM(0x075E); }                        // coin counter
inline int    area_type()         { return RAM(0x074E); }                        // 0 water,1 ground,2 underground,3 castle
inline int    area_index()        { return RAM(0x0760); }                        // internal area/sublevel counter
inline int    world_num()         { return RAM(0x075F); }                        // 0-indexed world
inline int    score_val()         { int s=0; for(int i=0;i<6;++i) s=s*10+RAM(0x07DD+i); return s; }
inline int    enemy_level_x(int i) { return RAM(0x006E + i) * 256 + RAM(0x0087 + i); }

// One background metatile from SMB's 2-screen tile buffer ($0500-$069F).
// 0 = sky/empty (passable), nonzero = solid ground/blocks/pipes.
inline int tile_at(int x_px, int y_px) {
    if (y_px < 32) return 0;
    int row = (y_px - 32) / 16;
    if (row < 0 || row > 12) return 0;
    int col  = x_px / 16;
    int page = (col / 16) % 2;
    int colp = col % 16;
    return RAM(0x0500 + page * 0xD0 + row * 16 + colp);
}

inline uint8_t action_buttons(int a) {
    using namespace nes;
    switch (a) {
        case A_RIGHT:    return RIGHT;
        case A_RIGHT_A:  return RIGHT | A;
        case A_RIGHT_B:  return RIGHT | B;
        case A_RIGHT_AB: return RIGHT | A | B;
        case A_A:        return A;
    }
    return 0;
}

// Fill obs (resized to STATE_DIM) from the CURRENT emulator RAM. Level-agnostic:
// it reads whatever level is loaded, so it works unchanged for 1-1 and 1-2.
inline void build_obs(std::vector<float>& obs) {
    obs.assign(STATE_DIM, 0.f);
    int px = RAM(0x0086), py = player_y();
    int lx = level_x();
    obs[0] = px / 255.f;
    obs[1] = py / 255.f;
    obs[2] = std::max(-1.f, std::min(1.f, x_speed() / 48.f));
    obs[3] = float_state() / 3.f;
    for (int i = 0; i < 5; ++i) {
        int active = RAM(0x000F + i) != 0;
        int rel_x  = enemy_level_x(i) - lx;
        int ey     = RAM(0x00CF + i);
        obs[4 + i * 3 + 0] = (float)active;
        obs[4 + i * 3 + 1] = active ? std::max(-1.f, std::min(1.f, rel_x / 128.f)) : 0.f;
        obs[4 + i * 3 + 2] = active ? std::max(-1.f, std::min(1.f, (ey - py) / 128.f)) : 0.f;
    }
    int base  = 19;
    int qbase = 19 + TILE_N + 1;
    for (int c = 0; c < TILE_COLS; ++c) {
        int tx = lx + c * 16 + 8;
        for (int r = 0; r < TILE_ROWS; ++r) {
            int ty = 32 + (TILE_ROW0 + r) * 16 + 8;
            int v  = tile_at(tx, ty);
            obs[base  + c * TILE_ROWS + r] = v != 0 ? 1.f : 0.f;
            obs[qbase + c * TILE_ROWS + r] = (v >= 0xC0 && v <= 0xC3) ? 1.f : 0.f;
        }
    }
    obs[19 + TILE_N] = std::min(1.f, power_state() / 2.f);
}

// Re-power the console and drive the title screen until Mario is in normal play.
// Lands at the World 1-1 start (fresh power-on). Level-specific envs continue
// from here (1-2 replays a 1-1 clear to transition).
inline void boot(const std::vector<uint8_t>& rom) {
    nes::load_bytes(rom.data(), (int)rom.size());
    auto run = [&](int frames, uint8_t btn) {
        nes::set_buttons(0, btn);
        for (int i = 0; i < frames; ++i) nes::step_frame();
    };
    run(40, 0);
    run(8, nes::START);
    run(4, 0);
    nes::set_buttons(0, 0);
    for (int i = 0; i < 300; ++i) {
        nes::step_frame();
        if (player_state() == 0x08 && level_x() > 0) break;
    }
}

// Hold action a's buttons for FRAME_SKIP frames (fast-forward, no bookkeeping).
inline void apply_action_frames(int a) {
    nes::set_buttons(0, action_buttons(a));
    for (int i = 0; i < FRAME_SKIP; ++i) nes::step_frame();
}

}  // namespace smb
