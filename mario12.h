#pragma once
// RL env for Super Mario Bros World 1-2 (underground). SEPARATE from the 1-1 env
// (mario::Env) but shares the level-agnostic core in mario_shared.h (obs layout,
// action set, RAM decoding, boot). What's 1-2-specific and lives HERE:
//   * start:  boot -> replay a 1-1 clear -> ride the transition into 1-2.
//   * win:    reach the flagpole (float_state==3) in 1-2's exit sub-area.
//   * reward: rightward progress + an ENTER-THE-EXIT-PIPE bonus (a separate
//             signal from progress -- 1-2 requires going through a pipe, and with
//             the right-biased/no-DOWN action set the only reachable area change
//             IS the horizontal exit pipe, so it is an unambiguous milestone).
#include "mario_shared.h"
#include <vector>
#include <string>
#include <cstdint>

namespace mario12 {

// 1-2-SPECIFIC action set: the 5 shared right-biased actions (SAME order/indices
// as smb, so a 1-1 net's output columns map 1:1) PLUS A_NOOP (stand still). 1-2
// has TIMING puzzles -- e.g. a pit whose far side has a walking Buzzy Beetle -- so
// the agent must be able to WAIT for the enemy to clear before jumping. The pure
// right-biased set (no wait) cannot express that; NOOP is what makes it learnable.
enum { A_RIGHT, A_RIGHT_A, A_RIGHT_B, A_RIGHT_AB, A_A, A_NOOP, N_ACTIONS };
uint8_t action_buttons(int a);   // 1-2 mapping (A_NOOP -> no buttons)

class Env {
public:
    static constexpr int STATE_DIM  = smb::STATE_DIM;
    static constexpr int FRAME_SKIP = smb::FRAME_SKIP;
    static constexpr int TILE_COLS  = smb::TILE_COLS;
    static constexpr int MAX_STEPS  = 1500;   // 1-2 is long; roomier than 1-1
    static constexpr int STALL_LIMIT = 120;   // roomy for 1-2's harder jumps
    static constexpr float PIPE_BONUS = 100.f; // reward for entering the exit pipe (new area)

    bool init(const char* rom_path);              // read ROM + the 1-1 entry demo (native)
    bool init_bytes(const uint8_t* data, int n);  // ROM from memory (WASM/JS)
    bool load_entry_demo(const char* path);       // 1-1 clear action seq used to reach 1-2
    void set_entry_demo(const std::vector<uint8_t>& d) { entry_demo_ = d; have_start_state_ = false; }

    const std::vector<float>& reset();            // restore/produce the cached 1-2 start; returns obs
    float step(int action, bool& done);           // apply action FRAME_SKIP frames
    const std::vector<float>& observation() const { return obs_; }

    int  mario_x() const { return smb::level_x(); }
    int  score()   const { return smb::score_val(); }
    int  coins()   const { return smb::coin_count(); }
    int  power()   const { return smb::power_state(); }
    int  area()    const { return smb::area_index(); }
    bool won()     const { return won_; }
    bool stomped() const { return stomped_; }
    bool entered_pipe() const { return entered_pipe_; }  // did this step go through the exit pipe?
    int  entry_demo_len() const { return (int)entry_demo_.size(); }

private:
    std::vector<uint8_t> rom_;
    std::vector<uint8_t> entry_demo_;      // 1-1 clear actions to reach 1-2
    std::vector<uint8_t> start_state_;     // cached 1-2 start snapshot (fast reset)
    bool have_start_state_ = false;
    std::vector<float> obs_;
    int prev_x_ = 0, steps_ = 0, start_lives_ = 0, max_x_ = 0, stall_ = 0;
    int start_area_ = 0, cur_area_ = 0;
    bool won_ = false, stomped_ = false, entered_pipe_ = false;

    void goto_1_2_start();                 // boot + replay entry demo + wait for underground
    bool is_dead() const;
    bool is_win() const;
};

}  // namespace mario12
