#pragma once
// One environment for EVERY Super Mario Bros stage (smbx::Env). The per-level
// envs (mario / mario12 / mario13 / mario14) each hand-tuned their start, win
// test, reward and observation; this one uses a single rule set so that one
// net can be trained on several stages at once and tested on stages it never
// saw -- the way to make it learn the game's rules instead of one level's route.
//
//   start      warp to any World-Stage through the death-reload trick (1-1 is a
//              plain boot), cached as a snapshot after the first time
//   win        flagpole grab, axe / castle victory, or the stage counter moving on
//   reward     new ground (ratchet), +power-up, -power loss, stomp / jump-over,
//              new sub-area (pipe), win, death. Standing still costs a little,
//              EXCEPT while an enemy is close ahead (waiting it out is allowed)
//   actions    run, running jump, hop, wait, walk back, back-jump
//   obs        an egocentric board around Mario: 5 channels x 13 screen rows x
//              16 columns (3 behind .. 12 ahead) -- solid, hittable block, enemy,
//              power-up item, Mario himself -- plus a few scalars. Ceilings,
//              walls, pits, enemies and items are all visible in the same form on
//              every stage.
#include "nes.h"
#include <vector>
#include <string>
#include <cstdint>

namespace smbx {

enum { A_RUN, A_RUNJUMP, A_HOP, A_WAIT, A_BACK, A_BACKJUMP, N_ACTIONS };
uint8_t action_buttons(int a);
const char* action_name(int a);

struct Level {
    int world = 1, stage = 1;                 // 1-based, e.g. {1,2} = 1-2
    std::string name() const { return std::to_string(world) + "-" + std::to_string(stage); }
};
bool parse_level(const std::string& s, Level& lv);   // "1-2" -> {1,2}

class Env {
public:
    static constexpr int GRID_C = 5, GRID_R = 13, GRID_W = 16, GRID_BEHIND = 3;
    static constexpr int GRID_N = GRID_C * GRID_R * GRID_W;
    static constexpr int N_SCALAR = 8;
    static constexpr int STATE_DIM = GRID_N + N_SCALAR;
    static constexpr int FRAME_SKIP = 4;
    static constexpr int MAX_STEPS = 2000;
    static constexpr int STALL_LIMIT = 120;

    struct Snapshot {
        std::vector<uint8_t> nes;
        int prev_x, steps, start_lives, max_x, stall, prev_power, cur_area, area_count;
        bool won;
    };

    bool init(const char* rom_path);
    bool init_bytes(const uint8_t* data, int n);
    void set_level(const Level& lv);          // takes effect at the next reset()
    const Level& level() const { return level_; }

    const std::vector<float>& reset();
    float step(int action, bool& done);
    const std::vector<float>& observation() const { return obs_; }
    void snapshot(Snapshot& s) const;
    void restore(const Snapshot& s);

    // Curriculum: snapshots along a demo of the current level.
    void build_checkpoints(const std::vector<uint8_t>& demo, int k_points);
    int num_checkpoints() const { return (int)ckpt_.size(); }
    const std::vector<float>& reset_to_checkpoint(int k);

    int mario_x() const;                       // x within the current sub-area
    int progress() const;                      // monotone-ish: 4096 * sub-areas entered + max x
    int power() const;
    bool won() const { return won_; }
    bool stomped() const { return stomped_; }

private:
    std::vector<uint8_t> rom_;
    std::vector<float> obs_;
    Level level_;
    std::vector<uint8_t> start_state_;
    bool have_start_ = false;
    std::vector<Snapshot> ckpt_;
    int prev_x_ = 0, steps_ = 0, start_lives_ = 0, max_x_ = 0, stall_ = 0, prev_power_ = 0;
    int cur_area_ = 0, area_count_ = 0, start_world_ = 0, start_stage_ = 0;
    bool won_ = false, stomped_ = false;
    void goto_start();
    void begin_episode();
    void build_obs();
    bool is_dead() const;
    bool is_win() const;
};

// Progress measure used by the shared trainer helpers (found by ADL).
inline int prog(const Env& e) { return e.progress(); }

}  // namespace smbx
