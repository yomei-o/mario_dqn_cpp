#pragma once
// CartPole-v1, matching the classic OpenAI Gym dynamics. A tiny control task
// used to prove the DQN core works before we wire it to the NES emulator.
// State: [x, x_dot, theta, theta_dot]. Actions: 0 = push left, 1 = push right.
// Reward +1 per step; episode ends on fall / out-of-bounds / 500 steps.
#include "autograd.h"
#include <array>
#include <cmath>

namespace rl {

class CartPole {
public:
    static constexpr int STATE_DIM = 4;
    static constexpr int N_ACTIONS = 2;

    std::array<float, 4> s{};
    int steps = 0;

    const std::array<float, 4>& reset() {
        for (auto& v : s) v = (ag::randf() * 2.f - 1.f) * 0.05f;   // U(-0.05, 0.05)
        steps = 0;
        return s;
    }

    // Returns reward; sets `done`. Advances the state by one 20ms tick.
    float step(int action, bool& done) {
        const float g = 9.8f, mc = 1.0f, mp = 0.1f, mt = mc + mp;
        const float len = 0.5f, pml = mp * len, force_mag = 10.f, tau = 0.02f;
        float x = s[0], xd = s[1], th = s[2], thd = s[3];
        float force = action == 1 ? force_mag : -force_mag;
        float ct = std::cos(th), st = std::sin(th);
        float temp = (force + pml * thd * thd * st) / mt;
        float thacc = (g * st - ct * temp) / (len * (4.f / 3.f - mp * ct * ct / mt));
        float xacc = temp - pml * thacc * ct / mt;
        s[0] = x + tau * xd;
        s[1] = xd + tau * xacc;
        s[2] = th + tau * thd;
        s[3] = thd + tau * thacc;
        ++steps;
        done = s[0] < -2.4f || s[0] > 2.4f ||
               s[2] < -0.2095f || s[2] > 0.2095f || steps >= 500;
        return 1.f;   // survived one more step
    }
};

}  // namespace rl
