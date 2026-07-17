#include "mario14.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace mario14 {

using namespace smb;

uint8_t action_buttons(int a) {
    using namespace nes;
    switch (a) {
        case A_RIGHT:    return RIGHT;
        case A_RIGHT_A:  return RIGHT | A;
        case A_RIGHT_B:  return RIGHT | B;
        case A_RIGHT_AB: return RIGHT | A | B;
        case A_A:        return A;
        case A_NOOP:     return 0;
    }
    return 0;
}

static bool hazard_ahead() {
    int lx = level_x();
    for (int i = 0; i < 5; ++i) {
        if (RAM(0x000F + i) == 0) continue;
        int rel = enemy_level_x(i) - lx;
        if (rel >= -8 && rel <= 72) return true;
    }
    return false;
}

bool Env::init(const char* rom_path) {
    FILE* f = std::fopen(rom_path, "rb");
    if (!f) { std::printf("cannot open ROM: %s\n", rom_path); return false; }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    rom_.resize(n);
    if (std::fread(rom_.data(), 1, n, f) != (size_t)n) { std::fclose(f); return false; }
    std::fclose(f);
    obs_.assign(STATE_DIM, 0.f);
    return true;
}

bool Env::init_bytes(const uint8_t* data, int n) {
    if (!data || n <= 0) return false;
    rom_.assign(data, data + n);
    obs_.assign(STATE_DIM, 0.f);
    return true;
}

void Env::goto_1_4_start() {
    boot(rom_);
    auto poke = [](){
        nes::ram_write(0x075F, (uint8_t)WARP_WORLD);
        nes::ram_write(0x075C, (uint8_t)WARP_STAGE);
        nes::ram_write(0x0760, (uint8_t)WARP_AREA);
    };
    nes::ram_write(0x075A, 9);
    poke();
    nes::ram_write(0x000E, 0x06);     // die -> area reload reads 1-4
    for (int k = 0; k < 600; ++k) {
        poke();
        nes::set_buttons(0, 0);
        nes::step_frame();
        if (k > 90 && player_state() == 0x08 && level_x() > 0) break;
    }
}

bool Env::is_dead() const {
    int st = player_state();
    return st == 0x06 || st == 0x0B || vert_page() > 1 || lives() < start_lives_;
}

// 1-4 has no flagpole: cleared when the castle area is left (area_index changes)
// or the world advances past the warp world (-> reached the axe).
bool Env::is_win() const {
    return area_index() != start_area_ || world_num() != WARP_WORLD;
}

const std::vector<float>& Env::reset() {
    if (have_start_state_) {
        nes::load_state(start_state_);
    } else {
        goto_1_4_start();
        nes::save_state(start_state_);
        have_start_state_ = true;
    }
    start_lives_ = lives();
    prev_x_ = level_x();
    max_x_  = prev_x_;
    steps_  = 0;
    stall_  = 0;
    start_area_ = area_index();
    cur_area_   = start_area_;
    won_ = false; stomped_ = false;
    build_obs(obs_);
    return obs_;
}

float Env::step(int action, bool& done) {
    int pa[5], prx[5], pey[5], ppy = player_y(), psc = score_val();
    for (int i = 0; i < 5; ++i) {
        pa[i]  = RAM(0x000F + i) != 0;
        prx[i] = enemy_level_x(i) - level_x();
        pey[i] = RAM(0x00CF + i);
    }

    nes::set_buttons(0, mario14::action_buttons(action));
    for (int i = 0; i < FRAME_SKIP; ++i) nes::step_frame();
    ++steps_;

    stomped_ = false;
    if (score_val() > psc)
        for (int i = 0; i < 5; ++i)
            if (pa[i] && prx[i] >= -24 && prx[i] <= 56 && ppy <= pey[i] + 4) { stomped_ = true; break; }

    float r = 0.f;
    int x = level_x();
    prev_x_ = x;
    if (x > max_x_) { r += std::min(25.f, (float)(x - max_x_)); max_x_ = x; stall_ = 0; }
    else if (!hazard_ahead()) { ++stall_; }
    if (stomped_) r += 12.f;
    else if (float_state() == 0x01) {
        for (int i = 0; i < 5; ++i)
            if (pa[i] && RAM(0x000F + i) != 0 && prx[i] >= 0 && prx[i] <= 48 &&
                (enemy_level_x(i) - x) < 0) { r += 25.f; break; }
    }

    done = false;
    // Check win (axe/level-clear) BEFORE death: leaving the area advances world/area.
    if (is_win())                   { r += 100.f; done = true; won_ = true; }
    else if (is_dead())             { r -= 50.f; done = true; }
    else if (stall_ >= STALL_LIMIT) { r -= 10.f; done = true; }
    else if (steps_ >= MAX_STEPS)   { done = true; }

    build_obs(obs_);
    return r;
}

}  // namespace mario14
