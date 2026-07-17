#include "mario12.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace mario12 {

using namespace smb;   // RAM accessors, build_obs, boot, etc.

// 1-2 action mapping: the 5 shared right-biased actions + NOOP (stand still, no
// buttons) so the agent can WAIT for a hazard (walking Buzzy Beetle) to clear.
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

// Is there an active enemy in the danger window just ahead of Mario? Used to
// FREEZE the stall timer / not punish standing still -- i.e. waiting for a
// walking enemy to clear is legitimate, not dithering.
static bool hazard_ahead() {
    int lx = level_x();
    for (int i = 0; i < 5; ++i) {
        if (RAM(0x000F + i) == 0) continue;
        int rel = enemy_level_x(i) - lx;
        if (rel >= -8 && rel <= 72) return true;   // ~within a jump ahead
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
    // Default entry demo: the committed 1-1 clear. Reaching 1-2 requires clearing
    // 1-1 first (deterministic), so this is how we "start at 1-2".
    if (entry_demo_.empty()) load_entry_demo("warmstarts/demo_clear_1-1.bin");
    return true;
}

bool Env::init_bytes(const uint8_t* data, int n) {
    if (!data || n <= 0) return false;
    rom_.assign(data, data + n);
    obs_.assign(STATE_DIM, 0.f);
    return true;
}

bool Env::load_entry_demo(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::printf("cannot open entry demo: %s\n", path); return false; }
    uint32_t n = 0;
    if (std::fread(&n, 4, 1, f) != 1) { std::fclose(f); return false; }
    entry_demo_.resize(n);
    size_t got = n ? std::fread(entry_demo_.data(), 1, n, f) : 0;
    std::fclose(f);
    have_start_state_ = false;   // invalidate cache: new entry -> new 1-2 start
    return got == n;
}

// boot -> clear 1-1 (replay entry demo) -> ride the flag/castle transition until
// the 1-2 underground is loaded and Mario is controllable.
void Env::goto_1_2_start() {
    boot(rom_);
    for (uint8_t a : entry_demo_) apply_action_frames(a);   // clear 1-1 to the flag
    // No input through the flag-slide / walk-to-castle / level load, until the
    // underground (area_type 2) sublevel (area_index 2) is up with x reset small.
    for (int k = 0; k < 1500; ++k) {
        nes::set_buttons(0, 0); nes::step_frame();
        if (area_index() == 2 && area_type() == 2 && level_x() < 200) break;
    }
    // let the entry animation finish (state 8 = normal control)
    for (int k = 0; k < 120 && player_state() != 8; ++k) {
        nes::set_buttons(0, nes::RIGHT); nes::step_frame();
    }
}

bool Env::is_dead() const {
    int st = player_state();
    return st == 0x06 || st == 0x0B || vert_page() > 1 || lives() < start_lives_;
}

bool Env::is_win() const {
    // 1-2 exit sub-area ends in a flagpole (like 1-1). float_state 3 = flag-climb.
    return float_state() == 0x03;
}

const std::vector<float>& Env::reset() {
    if (have_start_state_) {
        nes::load_state(start_state_);
    } else {
        goto_1_2_start();
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
    won_ = false; stomped_ = false; entered_pipe_ = false;
    build_obs(obs_);
    return obs_;
}

float Env::step(int action, bool& done) {
    // enemy snapshot BEFORE advancing (for stomp/evade detection, as in 1-1)
    int pa[5], prx[5], pey[5], ppy = player_y(), psc = score_val();
    for (int i = 0; i < 5; ++i) {
        pa[i]  = RAM(0x000F + i) != 0;
        prx[i] = enemy_level_x(i) - level_x();
        pey[i] = RAM(0x00CF + i);
    }

    nes::set_buttons(0, mario12::action_buttons(action));
    for (int i = 0; i < FRAME_SKIP; ++i) nes::step_frame();
    ++steps_;

    stomped_ = false;
    if (score_val() > psc)
        for (int i = 0; i < 5; ++i)
            if (pa[i] && prx[i] >= -24 && prx[i] <= 56 && ppy <= pey[i] + 4) { stomped_ = true; break; }

    float r = 0.f;
    entered_pipe_ = false;
    int a = area_index();
    if (a != cur_area_) {
        // Went through a pipe into a new sub-area. With the no-DOWN action set the
        // only such transition reachable is the HORIZONTAL exit pipe, so this is
        // the "entered the exit pipe" milestone -- a reward SEPARATE from progress.
        entered_pipe_ = true;
        cur_area_ = a;
        r += PIPE_BONUS;
        prev_x_ = level_x();          // x restarts in the new area -- rebaseline
        max_x_  = prev_x_;
        stall_  = 0;
    } else {
        int x = level_x();
        prev_x_ = x;
        // RATCHET progress: reward ONLY genuine new ground (max-x advance), and give
        // NO per-step time penalty. Standing still therefore costs nothing -- so
        // WAITING for a walking Buzzy Beetle to clear before jumping the pit is not
        // punished (the old dx-0.1 made rushing right always beat waiting, which is
        // exactly why 1-2's timing spots were unlearnable). Death still dominates.
        if (x > max_x_) {
            r += std::min(25.f, (float)(x - max_x_));
            max_x_ = x; stall_ = 0;
        } else if (!hazard_ahead()) {
            ++stall_;                 // only count stall when NOT legitimately waiting
        }
        if (stomped_) r += 12.f;      // defeated an enemy on the forward line
        else if (float_state() == 0x01) {
            for (int i = 0; i < 5; ++i)
                if (pa[i] && RAM(0x000F + i) != 0 && prx[i] >= 0 && prx[i] <= 48 &&
                    (enemy_level_x(i) - x) < 0) { r += 25.f; break; }   // evaded/jumped over
        }
    }

    done = false;
    if (is_dead())               { r -= 50.f; done = true; }
    else if (is_win())           { r += 100.f; done = true; won_ = true; }
    else if (stall_ >= STALL_LIMIT) { r -= 10.f; done = true; }
    else if (steps_ >= MAX_STEPS)   { done = true; }

    build_obs(obs_);
    return r;
}

}  // namespace mario12
