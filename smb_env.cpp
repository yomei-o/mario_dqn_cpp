#include "smb_env.h"
#include "mario_shared.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace smbx {

using smb::RAM;

uint8_t action_buttons(int a) {
    using namespace nes;
    switch (a) {
        case A_RUN:      return RIGHT | B;
        case A_RUNJUMP:  return RIGHT | A | B;
        case A_HOP:      return A;
        case A_WAIT:     return 0;
        case A_BACK:     return LEFT;
        case A_BACKJUMP: return LEFT | A;
    }
    return 0;
}

const char* action_name(int a) {
    static const char* n[] = {"run", "runjump", "hop", "wait", "back", "backjump"};
    return a >= 0 && a < N_ACTIONS ? n[a] : "?";
}

bool parse_level(const std::string& s, Level& lv) {
    int w = 0, st = 0;
    if (std::sscanf(s.c_str(), "%d-%d", &w, &st) != 2 || w < 1 || w > 8 || st < 1 || st > 4) return false;
    lv.world = w; lv.stage = st;
    return true;
}

bool Env::init(const char* rom_path) {
    FILE* f = std::fopen(rom_path, "rb");
    if (!f) { std::printf("cannot open ROM: %s\n", rom_path); return false; }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    rom_.resize(n);
    bool ok = std::fread(rom_.data(), 1, n, f) == (size_t)n;
    std::fclose(f);
    obs_.assign(STATE_DIM, 0.f);
    return ok;
}

bool Env::init_bytes(const uint8_t* data, int n) {
    if (!data || n <= 0) return false;
    rom_.assign(data, data + n);
    obs_.assign(STATE_DIM, 0.f);
    return true;
}

void Env::set_level(const Level& lv) {
    if (lv.world != level_.world || lv.stage != level_.stage) { have_start_ = false; ckpt_.clear(); }
    level_ = lv;
}

// Boot lands on 1-1. Any other stage: write world/stage/area and force a death;
// the reload reads the poked stage (re-poked every frame so the load sees it).
// The area index follows the game's level table: worlds 1, 2, 4 and 7 have an
// intro area before stage 2, which shifts later stages by one (1-2 = area 2,
// 1-3 = area 3, 1-4 = area 4).
void Env::goto_start() {
    smb::boot(rom_);
    if (level_.world == 1 && level_.stage == 1) return;
    int w = level_.world - 1, s = level_.stage - 1;
    int area = s + ((level_.world == 1 || level_.world == 2 || level_.world == 4 || level_.world == 7) && s >= 1 ? 1 : 0);
    auto poke = [&]() {
        nes::ram_write(0x075F, (uint8_t)w);
        nes::ram_write(0x075C, (uint8_t)s);
        nes::ram_write(0x0760, (uint8_t)area);
    };
    nes::ram_write(0x075A, 9);                 // lives: the warp death must not game-over
    poke();
    nes::ram_write(0x000E, 0x06);              // dying -> area reload
    for (int k = 0; k < 600; ++k) {
        poke();
        nes::set_buttons(0, 0);
        nes::step_frame();
        if (k > 90 && smb::player_state() == 0x08 && smb::level_x() > 0) break;
    }
}

void Env::begin_episode() {
    start_lives_ = smb::lives();
    prev_x_ = max_x_ = smb::level_x();
    steps_ = stall_ = 0;
    prev_power_ = smb::power_state();
    cur_area_ = smb::area_index();
    area_count_ = 0;
    start_world_ = smb::world_num();
    start_stage_ = RAM(0x075C);
    won_ = stomped_ = false;
    build_obs();
}

const std::vector<float>& Env::reset() {
    if (have_start_) nes::load_state(start_state_);
    else { goto_start(); nes::save_state(start_state_); have_start_ = true; }
    begin_episode();
    return obs_;
}

void Env::snapshot(Snapshot& s) const {
    nes::save_state(s.nes);
    s.prev_x = prev_x_; s.steps = steps_; s.start_lives = start_lives_; s.max_x = max_x_;
    s.stall = stall_; s.prev_power = prev_power_; s.cur_area = cur_area_; s.area_count = area_count_; s.won = won_;
}

void Env::restore(const Snapshot& s) {
    nes::load_state(s.nes);
    prev_x_ = s.prev_x; steps_ = s.steps; start_lives_ = s.start_lives; max_x_ = s.max_x;
    stall_ = s.stall; prev_power_ = s.prev_power; cur_area_ = s.cur_area; area_count_ = s.area_count; won_ = s.won;
    stomped_ = false;
    build_obs();
}

void Env::build_checkpoints(const std::vector<uint8_t>& demo, int k_points) {
    ckpt_.clear();
    int L = (int)demo.size();
    if (L < 40 || k_points < 1) return;
    reset();
    std::vector<int> targets;
    int lo = std::max(1, L / 12), hi = std::max(1, L - 8);
    for (int j = 0; j < k_points; ++j) targets.push_back(k_points == 1 ? hi : lo + (hi - lo) * j / (k_points - 1));
    bool done = false;
    size_t t = 0;
    for (int i = 0; i < hi && t < targets.size() && !done; ++i) {
        step(demo[i], done);
        while (t < targets.size() && i + 1 >= targets[t]) {
            if (!done) { ckpt_.emplace_back(); snapshot(ckpt_.back()); ckpt_.back().steps = 0; }
            ++t;
        }
    }
}

const std::vector<float>& Env::reset_to_checkpoint(int k) {
    restore(ckpt_[k]);
    steps_ = 0; stall_ = 0; won_ = false;
    return obs_;
}

int Env::mario_x() const { return smb::level_x(); }
int Env::progress() const { return area_count_ * 4096 + max_x_; }
int Env::power() const { return smb::power_state(); }

bool Env::is_dead() const {
    int st = smb::player_state();
    return st == 0x06 || st == 0x0B || smb::vert_page() > 1 || smb::lives() < start_lives_;
}

bool Env::is_win() const {
    return smb::float_state() == 0x03                      // flagpole
        || RAM(0x0770) == 2                                // victory mode (castle axe)
        || smb::world_num() != start_world_ || RAM(0x075C) != start_stage_;
}

// Any active enemy just ahead (or right on Mario): waiting here is not stalling.
static bool hazard_ahead() {
    int lx = smb::level_x();
    for (int i = 0; i < 5; ++i) {
        if (RAM(0x000F + i) == 0) continue;
        int rel = smb::enemy_level_x(i) - lx;
        if (rel >= -16 && rel <= 96) return true;
    }
    return false;
}

float Env::step(int action, bool& done) {
    int pa[5], prx[5], pey[5], ppy = smb::player_y(), psc = smb::score_val();
    for (int i = 0; i < 5; ++i) {
        pa[i] = RAM(0x000F + i) != 0;
        prx[i] = smb::enemy_level_x(i) - smb::level_x();
        pey[i] = RAM(0x00CF + i);
    }
    nes::set_buttons(0, action_buttons(action));
    for (int i = 0; i < FRAME_SKIP; ++i) nes::step_frame();
    ++steps_;

    stomped_ = false;
    if (smb::score_val() > psc)
        for (int i = 0; i < 5; ++i)
            if (pa[i] && prx[i] >= -24 && prx[i] <= 56 && ppy <= pey[i] + 4) { stomped_ = true; break; }

    float r = 0.f;
    int x = smb::level_x();
    int area = smb::area_index();
    if (area != cur_area_) {                     // through a pipe into another sub-area
        cur_area_ = area; ++area_count_;
        r += 50.f;
        prev_x_ = max_x_ = x; stall_ = 0;
    } else {
        if (x > max_x_) { r += std::min(25.f, (float)(x - max_x_)); max_x_ = x; stall_ = 0; }
        else if (!hazard_ahead()) { ++stall_; r -= 0.3f; }
        prev_x_ = x;
    }
    if (stomped_) r += 12.f;
    else if (smb::float_state() == 0x01)
        for (int i = 0; i < 5; ++i)
            if (pa[i] && RAM(0x000F + i) != 0 && prx[i] >= 0 && prx[i] <= 48 && smb::enemy_level_x(i) - x < 0) {
                r += 25.f; break;
            }
    int pw = smb::power_state();
    if (pw > prev_power_) r += 40.f;            // mushroom / flower
    else if (pw < prev_power_) r -= 20.f;       // got hit (still alive)
    prev_power_ = pw;

    done = false;
    if (is_win())                    { r += 100.f; done = true; won_ = true; }
    else if (is_dead())              { r -= 50.f; done = true; }
    else if (stall_ >= STALL_LIMIT)  { r -= 10.f; done = true; }
    else if (steps_ >= MAX_STEPS)    { done = true; }
    build_obs();
    return r;
}

// Egocentric board. Columns are 16px tiles from 3 behind Mario to 12 ahead,
// rows are the 13 playfield rows of the screen (so ceilings and the ground are
// both in view). SMB keeps an object's y 16px above its sprite for small sizes;
// enemies and Mario are marked on the rows their body covers.
void Env::build_obs() {
    obs_.assign(STATE_DIM, 0.f);
    auto cell = [&](int c, int row, int col) -> float& {
        return obs_[(size_t)(c * GRID_R + row) * GRID_W + col];
    };
    int lx = smb::level_x();
    int col0 = lx / 16 - GRID_BEHIND;
    for (int col = 0; col < GRID_W; ++col) {
        int tx = (col0 + col) * 16 + 8;
        if (tx < 0) continue;
        for (int row = 0; row < GRID_R; ++row) {
            int v = smb::tile_at(tx, 32 + row * 16 + 8);
            if (!v) continue;
            bool hittable = (v >= 0xC0 && v <= 0xC3) || v == 0x51 || v == 0x52;   // ?-blocks, bricks
            cell(hittable ? 1 : 0, row, col) = 1.f;
        }
    }
    auto mark = [&](int c, int obj_x, int obj_y, int body_lo, int body_hi) {
        int col = (obj_x + 8) / 16 - col0;
        if (col < 0 || col >= GRID_W) return;
        for (int yy = obj_y + body_lo; yy <= obj_y + body_hi; yy += 8) {
            int row = (yy - 32) / 16;
            if (row >= 0 && row < GRID_R) cell(c, row, col) = 1.f;
        }
    };
    for (int i = 0; i < 5; ++i)
        if (RAM(0x000F + i)) mark(2, smb::enemy_level_x(i), RAM(0x00CF + i), 8, 24);
    if (RAM(0x0014)) mark(3, smb::enemy_level_x(5), RAM(0x00D4), 8, 24);   // power-up object slot
    bool big = smb::power_state() > 0;
    int py = smb::player_y();
    mark(4, lx, py, big ? 0 : 16, 31);

    float* sc = &obs_[GRID_N];
    sc[0] = py / 240.f;
    sc[1] = std::max(-1.f, std::min(1.f, smb::x_speed() / 48.f));
    sc[2] = std::max(-1.f, std::min(1.f, (int8_t)RAM(0x009F) / 8.f));   // vertical speed
    sc[3] = smb::float_state() == 0x01 ? 1.f : 0.f;                      // airborne
    sc[4] = std::min(1.f, smb::power_state() / 2.f);
    sc[5] = (lx % 16) / 16.f;                                            // sub-tile offset
    sc[6] = smb::area_type() / 3.f;                                      // water/ground/underground/castle
    sc[7] = RAM(0x0033) == 1 ? 1.f : 0.f;                                // facing right
}

}  // namespace smbx
