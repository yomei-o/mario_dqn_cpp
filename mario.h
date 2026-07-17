#pragma once
// A gym-style RL wrapper around the headless NES, specialized for Super Mario
// Bros 1-1. Observation = a compact RAM-feature vector (Mario pos/vel/state +
// nearby enemies + terrain grid + power). Reward = PROGRESS-FIRST + ENEMY-HANDLING:
// rightward progress - time dominates, plus an immediate bonus for getting past an
// enemy -- jump-OVER/evade (+25, safer for small Mario) or STOMP (+12) -- with
// death/stall/win terminals. No item (mushroom/coin) reward (delayed/multi-step,
// hard to credit). See Env::step. This is what the DQN agent optimizes against.
#include "nes.h"
#include <vector>
#include <string>
#include <cstdint>

namespace mario {

// Discrete action set (right-biased, like gym's SIMPLE_MOVEMENT subset).
// B = run/fire, A = jump.
enum { A_RIGHT, A_RIGHT_A, A_RIGHT_B, A_RIGHT_AB, A_A, N_ACTIONS };

class Env {
public:
    // Terrain occupancy grid sampled ahead of Mario from SMB's background tile
    // RAM. Rows are FIXED screen rows (NOT relative to Mario), so the grid stays
    // meaningful mid-jump: the agent keeps seeing the pit/pipe/ground profile
    // ahead even while airborne (its own height is obs[1], its speed obs[2]).
    static constexpr int TILE_COLS = 10;     // columns from Mario's, going right (~160px lookahead)
    static constexpr int TILE_ROW0 = 6;      // topmost screen row sampled (row 0..12 playfield)
    static constexpr int TILE_ROWS = 7;      // rows 6..12: covers the tallest pipe down to ground
    static constexpr int TILE_N = TILE_COLS * TILE_ROWS;
    // obs layout: [0..3] mario pos/vel/state, [4..18] 5 enemy slots,
    // [19..18+TILE_N] terrain grid (solid?), [19+TILE_N] power (0 small/0.5 super/1 fire),
    // [20+TILE_N..19+2*TILE_N] ? -block grid (item/coin blocks to hit from below).
    static constexpr int STATE_DIM = 19 + TILE_N + 1 + TILE_N;
    static constexpr int FRAME_SKIP = 4;
    static constexpr int MAX_STEPS = 1200;   // ~ episode time limit (env steps)
    static constexpr int STALL_LIMIT = 90;   // end episode after this many stalled steps
                                             // (roomy enough to attempt a pipe jump)

    bool init(const char* rom_path);         // read ROM bytes once (native)
    bool init_bytes(const uint8_t* data, int n);  // ROM from memory (WASM/JS)
    const std::vector<float>& reset();        // reboot + start 1-1; returns obs
    // Curriculum: reboot, replay the loaded demonstration's first `start_idx`
    // actions to fast-forward to a checkpoint, then begin the episode there.
    // The emulator is deterministic in-process, so this lands on the same state
    // every time. start_idx=0 is identical to reset().
    const std::vector<float>& reset_from(int start_idx);
    bool load_demo(const char* path);         // load a demo action sequence
    int demo_len() const { return (int)demo_.size(); }
    // Precompute k_points emulator snapshots spread across the back half of the
    // demo (one forward pass). Afterwards reset_to_checkpoint(k) restores state
    // instantly instead of replaying -- this is what makes the curriculum fast.
    void build_curriculum(int k_points);
    int num_checkpoints() const { return (int)ckpt_states_.size(); }
    int checkpoint_x(int k) const { return ckpt_x_[k]; }
    // How many demo actions were replayed to REACH checkpoint k. Concatenating
    // demo_actions()[0..this] with the actions taken FROM checkpoint k yields a
    // valid from-boot sequence reproducing that trajectory (the emulator is
    // deterministic and the snapshot == replaying that demo prefix; see snaptest).
    int checkpoint_demo_len(int k) const { return ckpt_target_[k]; }
    const std::vector<uint8_t>& demo_actions() const { return demo_; }
    const std::vector<float>& reset_to_checkpoint(int k);
    float step(int action, bool& done);       // apply action FRAME_SKIP frames
    const std::vector<float>& observation() const { return obs_; }
    int mario_x() const;                       // level x-position (for logging)
    int score() const;                         // SMB score (decoded, x10 implied)
    int coins() const;                         // coin counter
    int power() const;                         // 0=small 1=super 2=fire
    bool won() const { return won_; }           // did the episode just end at the flagpole?
    bool stomped() const { return stomped_; }    // did this step just defeat an enemy (stomp)?

private:
    std::vector<uint8_t> rom_;
    std::vector<float> obs_;
    std::vector<uint8_t> demo_;               // demonstration actions for curriculum
    std::vector<std::vector<uint8_t>> ckpt_states_;  // precomputed emulator snapshots
    std::vector<int> ckpt_x_;                 // level-x at each checkpoint (for logging)
    std::vector<int> ckpt_target_;            // demo-action count replayed to reach each checkpoint
    std::vector<uint8_t> start_state_;        // cached post-boot level-start snapshot (fast reset)
    bool have_start_state_ = false;
    int prev_x_ = 0, steps_ = 0, start_lives_ = 0;
    int max_x_ = 0, stall_ = 0;                // for stagnation shaping
    int prev_score_ = 0, prev_power_ = 0;      // for score / power-up reward shaping
    bool won_ = false;                         // set when an episode ends at the flagpole
    bool stomped_ = false;                     // set on a step that defeats an enemy (stomp)
    void boot();                               // title screen -> in-level play
    void apply_action_frames(int a);           // raw FRAME_SKIP advance (no reward/done)
    void build_obs();
    bool is_dead() const;
    bool is_win() const;
};

}  // namespace mario
