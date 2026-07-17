#pragma once
// RL env for Super Mario Bros World 1-4 (castle: fire bars, Bowser, the axe).
// SEPARATE from the other level envs, sharing the level-agnostic core. 1-4 has NO
// flagpole -- it ends when Mario reaches the AXE (bridge collapses, advance to 2-1).
// Specifics here:
//   * start:  WARP -- poke the SMB stage RAM (world0/stage3/area4) + death-reload.
//   * win:    the 1-4 area is left (area_index changes) or the world advances past 1
//             -- i.e. the axe was reached / the castle was cleared.
//   * reward: RATCHET progress + stomp/evade (fire bars are unstompable hazards;
//             death penalty teaches avoidance) + the A_NOOP wait action for timing
//             through the fire bars.
#include "mario_shared.h"
#include <vector>
#include <string>
#include <cstdint>

namespace mario14 {

enum { A_RIGHT, A_RIGHT_A, A_RIGHT_B, A_RIGHT_AB, A_A, A_NOOP, N_ACTIONS };
uint8_t action_buttons(int a);

class Env {
public:
    static constexpr int STATE_DIM  = smb::STATE_DIM;
    static constexpr int ACTIONS    = N_ACTIONS;
    static constexpr int FRAME_SKIP = smb::FRAME_SKIP;
    static constexpr int TILE_COLS  = smb::TILE_COLS;
    static constexpr int MAX_STEPS  = 1200;
    static constexpr int STALL_LIMIT = 120;
    static constexpr int WARP_WORLD = 0, WARP_STAGE = 3, WARP_AREA = 4;

    bool init(const char* rom_path);
    bool init_bytes(const uint8_t* data, int n);

    const std::vector<float>& reset();
    float step(int action, bool& done);
    const std::vector<float>& observation() const { return obs_; }

    int  mario_x() const { return smb::level_x(); }
    int  score()   const { return smb::score_val(); }
    int  power()   const { return smb::power_state(); }
    int  area()    const { return smb::area_index(); }
    bool won()     const { return won_; }
    bool stomped() const { return stomped_; }

private:
    std::vector<uint8_t> rom_;
    std::vector<uint8_t> start_state_;
    bool have_start_state_ = false;
    std::vector<float> obs_;
    int prev_x_ = 0, steps_ = 0, start_lives_ = 0, max_x_ = 0, stall_ = 0;
    int start_area_ = 0, cur_area_ = 0;
    bool won_ = false, stomped_ = false;

    void goto_1_4_start();
    bool is_dead() const;
    bool is_win() const;
};

}  // namespace mario14
