#include "mario.h"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace mario {

// --- Super Mario Bros RAM map (well-documented addresses) -------------------
static inline int RAM(uint16_t a) { return nes::ram(a); }
static int level_x() { return RAM(0x006D) * 256 + RAM(0x0086); }   // x in level
static int player_y() { return RAM(0x00CE); }                       // y on screen
static int8_t x_speed() { return (int8_t)RAM(0x0057); }             // signed
static int player_state() { return RAM(0x000E); }                   // 0x08 normal, 0x06/0x0B dead/dying
static int float_state() { return RAM(0x001D); }                    // 0 ground, 1 air, 3 climbing
static int lives() { return RAM(0x075A); }
static int vert_page() { return RAM(0x00B5); }                      // >1 => fell into a pit

// Button mask for each discrete action.
static uint8_t action_buttons(int a) {
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

bool Env::is_dead() const {
    int st = player_state();
    return st == 0x06 || st == 0x0B || vert_page() > 1 || lives() < start_lives_;
}

bool Env::is_win() const {
    // 1-1 flagpole/level-end: player enters "end of level" states, or x passes ~3160.
    return float_state() == 0x03 || level_x() > 3160;
}

const std::vector<float>& Env::reset() {
    nes::load_bytes(rom_.data(), (int)rom_.size());   // re-power the console
    auto run = [&](int frames, uint8_t btn) {
        nes::set_buttons(0, btn);
        for (int i = 0; i < frames; ++i) nes::step_frame();
    };
    run(40, 0);                    // boot to title
    run(8, nes::START);            // press START
    run(4, 0);
    // advance until Mario is actually in normal play (level loaded)
    nes::set_buttons(0, 0);
    for (int i = 0; i < 300; ++i) {
        nes::step_frame();
        if (player_state() == 0x08 && level_x() > 0) break;
    }
    start_lives_ = lives();
    prev_x_ = level_x();
    steps_ = 0;
    build_obs();
    return obs_;
}

float Env::step(int action, bool& done) {
    uint8_t btn = action_buttons(action);
    nes::set_buttons(0, btn);
    for (int i = 0; i < FRAME_SKIP; ++i) nes::step_frame();
    ++steps_;

    int x = level_x();
    float dx = (float)(x - prev_x_);
    prev_x_ = x;

    // reward: progress right, small time penalty, big penalty on death, bonus on win
    float r = dx - 0.1f;
    done = false;
    if (is_dead()) { r -= 25.f; done = true; }
    else if (is_win()) { r += 50.f; done = true; }
    else if (steps_ >= MAX_STEPS) { done = true; }
    r = std::max(-25.f, std::min(25.f, r));   // clip

    build_obs();
    return r;
}

int Env::mario_x() const { return level_x(); }

void Env::build_obs() {
    obs_.assign(STATE_DIM, 0.f);
    int px = RAM(0x0086), py = player_y();
    obs_[0] = px / 255.f;
    obs_[1] = py / 255.f;
    obs_[2] = std::max(-1.f, std::min(1.f, x_speed() / 48.f));
    obs_[3] = float_state() / 3.f;
    // 5 enemy slots: active flag + relative position to Mario
    for (int i = 0; i < 5; ++i) {
        int active = RAM(0x000F + i) != 0;
        int ex = RAM(0x0087 + i), ey = RAM(0x00CF + i);
        obs_[4 + i * 3 + 0] = (float)active;
        obs_[4 + i * 3 + 1] = active ? std::max(-1.f, std::min(1.f, (ex - px) / 128.f)) : 0.f;
        obs_[4 + i * 3 + 2] = active ? std::max(-1.f, std::min(1.f, (ey - py) / 128.f)) : 0.f;
    }
}

}  // namespace mario
