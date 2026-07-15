// DQN on CartPole -- Phase 1 of the "Mario DQN in C++" project. This proves the
// DQN core (replay + target network + Double-DQN TD loss + epsilon-greedy +
// gradient clipping + best-net checkpointing) on a tiny control task before we
// wire the same agent to the NES emulator.
//
// Everything runs on the self-made autograd; only dependency is the C++ stdlib.
#include "autograd.h"
#include "cartpole.h"
#include "qnet.h"
#include "replay.h"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <deque>
#include <cmath>

using namespace ag;
using namespace rl;

static int argmax_row(const std::vector<float>& q, int off, int A) {
    int best = 0;
    for (int a = 1; a < A; ++a) if (q[off + a] > q[off + best]) best = a;
    return best;
}

static int greedy_action(QNet& net, const std::array<float, 4>& s) {
    auto x = Tensor::from({s[0], s[1], s[2], s[3]}, {1, CartPole::STATE_DIM}, false);
    return argmax_row(net.forward(x).data(), 0, net.adim);
}

// Average greedy return over `n` episodes (no exploration) -- the honest metric.
static float greedy_eval(QNet& net, CartPole& env, int n) {
    float total = 0;
    for (int e = 0; e < n; ++e) {
        const auto& s = env.reset(); bool done = false;
        while (!done) total += env.step(greedy_action(net, s), done);
    }
    return total / n;
}

// Clip the global L2 norm of all gradients to `max_norm` (stabilizes DQN).
static void clip_grads(std::vector<Tensor>& ps, float max_norm) {
    double sq = 0;
    for (auto& p : ps) for (float g : p.grad()) sq += (double)g * g;
    float norm = (float)std::sqrt(sq);
    if (norm > max_norm) {
        float s = max_norm / (norm + 1e-6f);
        for (auto& p : ps) for (float& g : p.grad()) g *= s;
    }
}

int main() {
    seed(0);
    const int S = CartPole::STATE_DIM, A = CartPole::N_ACTIONS;
    const float gamma = 0.99f;
    const int batch = 64, warmup = 1000, target_sync = 500, episodes = 800;
    const float eps_start = 1.0f, eps_end = 0.05f, eps_decay_steps = 12000.f;

    QNet online(S, A, 128), target(S, A, 128), best(S, A, 128);
    target.copy_from(online);
    best.copy_from(online);
    Adam opt(online.params(), 5e-4f);
    auto params = online.params();
    Replay replay(50000);
    CartPole env;

    long total_steps = 0;
    float best_eval = 0;

    for (int ep = 1; ep <= episodes; ++ep) {
        const auto& s = env.reset();
        bool done = false;
        while (!done) {
            float eps = std::max(eps_end, eps_start - (eps_start - eps_end) * total_steps / eps_decay_steps);
            int a = (ag::randf() < eps) ? (int)(ag::randf() * A) % A : greedy_action(online, s);
            std::array<float, 4> s_prev = s;
            float r = env.step(a, done);
            replay.push({{s_prev.begin(), s_prev.end()}, {s.begin(), s.end()}, a, r, done});
            ++total_steps;

            if ((int)replay.size() >= warmup) {
                auto idx = replay.sample(batch);
                auto xs = Tensor::zeros({batch, S}, false), xns = Tensor::zeros({batch, S}, false);
                for (int b = 0; b < batch; ++b) {
                    const auto& tr = replay.buf[idx[b]];
                    for (int k = 0; k < S; ++k) { xs.data()[b * S + k] = tr.s[k]; xns.data()[b * S + k] = tr.ns[k]; }
                }
                // Double DQN: online chooses next action, target evaluates it
                auto q_ns_online = online.forward(xns);
                auto q_ns_target = target.forward(xns);
                auto q = online.forward(xs);                 // graph we backprop
                auto tgt = Tensor::from(q.data(), {batch, A}, false);
                for (int b = 0; b < batch; ++b) {
                    const auto& tr = replay.buf[idx[b]];
                    int a_star = argmax_row(q_ns_online.data(), b * A, A);
                    float boot = tr.done ? 0.f : gamma * q_ns_target.data()[b * A + a_star];
                    tgt.data()[b * A + tr.a] = tr.r + boot;   // TD target on taken action only
                }
                auto loss = mul_scalar(sum(huber(sub(q, tgt))), 1.0f / batch);
                opt.zero_grad();
                loss.backward();
                clip_grads(params, 10.f);
                opt.step();
            }
            if (total_steps % target_sync == 0) target.copy_from(online);
        }

        // periodic honest greedy eval + keep the best network (like AlphaZero)
        if (ep % 20 == 0) {
            float ev = greedy_eval(online, env, 10);
            if (ev > best_eval) { best_eval = ev; best.copy_from(online); }
            float eps = std::max(eps_end, eps_start - (eps_start - eps_end) * total_steps / eps_decay_steps);
            std::printf("ep %3d  greedy-eval %5.1f  (best %5.1f)  eps %.2f  steps %ld\n",
                        ep, ev, best_eval, eps, total_steps);
            std::fflush(stdout);
            if (best_eval >= 475.f) { std::printf("\nSOLVED: greedy avg >= 475\n"); break; }
        }
    }

    std::printf("\nbest greedy-eval = %.1f\n== final greedy eval of BEST net (5 eps) ==\n", best_eval);
    for (int e = 0; e < 5; ++e) {
        const auto& s = env.reset(); bool done = false; float ret = 0;
        while (!done) ret += env.step(greedy_action(best, s), done);
        std::printf("  eval ep %d: return %.0f\n", e, ret);
    }
    return 0;
}
