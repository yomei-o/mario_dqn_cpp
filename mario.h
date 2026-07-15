#pragma once
// A gym-style RL wrapper around the headless NES, specialized for Super Mario
// Bros 1-1. Observation = a compact RAM-feature vector (Mario pos/vel/state +
// nearby enemies); reward = rightward progress - time - death + goal. This is
// what the DQN agent sees and is optimized against.
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
    static constexpr int STATE_DIM = 19;
    static constexpr int FRAME_SKIP = 4;
    static constexpr int MAX_STEPS = 1200;   // ~ episode time limit (env steps)

    bool init(const char* rom_path);         // read ROM bytes once
    const std::vector<float>& reset();        // reboot + start 1-1; returns obs
    float step(int action, bool& done);       // apply action FRAME_SKIP frames
    const std::vector<float>& observation() const { return obs_; }
    int mario_x() const;                       // level x-position (for logging)

private:
    std::vector<uint8_t> rom_;
    std::vector<float> obs_;
    int prev_x_ = 0, steps_ = 0, start_lives_ = 0;
    void build_obs();
    bool is_dead() const;
    bool is_win() const;
};

}  // namespace mario
