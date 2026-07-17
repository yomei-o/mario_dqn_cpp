#pragma once
// RL env for Super Mario Bros World 1-3 (athletic / tree-top platforms over pits).
// SEPARATE from the 1-1/1-2 envs, sharing the level-agnostic core (mario_shared.h).
// 1-3 specifics that live HERE:
//   * start:  can't be reached by clearing 1-1/1-2 (no 1-2 clear asset), so we WARP
//             -- poke the SMB stage RAM (world0/stage2/area3) and force a death-reload
//             so the area-load routine reads 1-3. Cached snapshot for fast reset.
//   * win:    flagpole (float_state==3). 1-3 ends in a flagpole (no pipe).
//   * reward: RATCHET progress (new-max-x only, waiting unpunished) + stomp/evade,
//             plus the A_NOOP wait action -- 1-3 has moving-lift timing over
//             bottomless pits, so being able to wait/time a jump matters.
#include "mario_shared.h"
#include <vector>
#include <string>
#include <cstdint>

namespace mario13 {

// Same 6-action set as the 1-2 env (5 shared right-biased + A_NOOP wait). Indices
// match smb's first 5 so a 1-1/1-2 net's output columns map directly.
enum { A_RIGHT, A_RIGHT_A, A_RIGHT_B, A_RIGHT_AB, A_A, A_NOOP, N_ACTIONS };
uint8_t action_buttons(int a);

class Env {
public:
    static constexpr int STATE_DIM  = smb::STATE_DIM;
    static constexpr int ACTIONS    = N_ACTIONS;   // for the generic trainer
    static constexpr int FRAME_SKIP = smb::FRAME_SKIP;
    static constexpr int TILE_COLS  = smb::TILE_COLS;
    static constexpr int MAX_STEPS  = 1200;
    static constexpr int STALL_LIMIT = 120;
    // SMB stage-warp RAM values for 1-3 (world 0, stage 2, area 3).
    static constexpr int WARP_WORLD = 0, WARP_STAGE = 2, WARP_AREA = 3;

    bool init(const char* rom_path);              // read ROM (native)
    bool init_bytes(const uint8_t* data, int n);  // ROM from memory (WASM/JS)

    const std::vector<float>& reset();            // restore/produce the cached 1-3 start; returns obs
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
    int prev_x_ = 0, steps_ = 0, start_lives_ = 0, max_x_ = 0, stall_ = 0, cur_area_ = 0;
    bool won_ = false, stomped_ = false;

    void goto_1_3_start();     // boot + warp to 1-3 via death-reload
    bool is_dead() const;
    bool is_win() const;
};

}  // namespace mario13
