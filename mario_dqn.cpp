// Phase 3: DQN plays Super Mario Bros 1-1 (RAM features, real NES via LaiNES).
// Reuses the exact DQN core proven on CartPole (QNet + Adam + replay + target
// net + Double DQN + grad clip). Same agent, new environment.
//
//   mario_dqn envtest "ROM.nes"     sanity: reset + hold RIGHT, print x
//   mario_dqn "ROM.nes"             train; logs episode reward and max distance
#include "autograd.h"
#include "qnet.h"
#include "replay.h"
#include "mario.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>

using namespace ag;
using namespace rl;

static int argmax_row(const std::vector<float>& q, int off, int A) {
    int best = 0;
    for (int a = 1; a < A; ++a) if (q[off + a] > q[off + best]) best = a;
    return best;
}

static int greedy_action(QNet& net, const std::vector<float>& s) {
    auto x = Tensor::from(s, {1, (int)s.size()}, false);
    return argmax_row(net.forward(x).data(), 0, net.adim);
}

static void clip_grads(std::vector<Tensor>& ps, float max_norm) {
    double sq = 0;
    for (auto& p : ps) for (float g : p.grad()) sq += (double)g * g;
    float norm = (float)std::sqrt(sq);
    if (norm > max_norm) { float s = max_norm / (norm + 1e-6f); for (auto& p : ps) for (float& g : p.grad()) g *= s; }
}

static int envtest(const char* rom) {
    mario::Env env;
    if (!env.init(rom)) return 1;
    env.reset();
    std::printf("after reset: x=%d\n", env.mario_x());
    for (int t = 0; t < 40; ++t) {
        bool done; float r = env.step(mario::A_RIGHT_B, done);   // run right
        if (t % 5 == 0) std::printf("  step %2d: x=%d r=%.1f done=%d\n", t, env.mario_x(), r, done);
        if (done) { std::printf("  episode ended at step %d (x=%d)\n", t, env.mario_x()); break; }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "envtest") == 0)
        return envtest(argc > 2 ? argv[2] : "Super Mario Bros (JU) (PRG 0).nes");
    const char* rom = argc > 1 ? argv[1] : "Super Mario Bros (JU) (PRG 0).nes";

    seed(0);
    const int S = mario::Env::STATE_DIM, A = mario::N_ACTIONS;
    const float gamma = 0.99f;
    const int batch = 32, warmup = 3000, target_sync = 2000, episodes = 4000;
    const float eps_start = 1.0f, eps_end = 0.1f, eps_decay_steps = 60000.f;

    QNet online(S, A, 256), target(S, A, 256), best(S, A, 256);
    target.copy_from(online); best.copy_from(online);
    Adam opt(online.params(), 2.5e-4f);
    auto params = online.params();
    Replay replay(100000);
    mario::Env env;
    if (!env.init(rom)) return 1;

    std::printf("== DQN x Super Mario Bros 1-1 (RAM features, real NES) ==\n");
    std::printf("   state_dim=%d actions=%d | batch=%d target_sync=%d\n", S, A, batch, target_sync);

    long total_steps = 0;
    std::deque<int> recent_x;   // last-50 episode max distances
    int best_x = 0;

    for (int ep = 1; ep <= episodes; ++ep) {
        std::vector<float> s = env.reset();
        bool done = false;
        float ep_ret = 0; int ep_max_x = 0;
        while (!done) {
            float eps = std::max(eps_end, eps_start - (eps_start - eps_end) * total_steps / eps_decay_steps);
            int a = (ag::randf() < eps) ? (int)(ag::randf() * A) % A : greedy_action(online, s);
            std::vector<float> s_next_holder;
            bool d;
            float r = env.step(a, d);
            const std::vector<float>& ns = env.observation();
            ep_ret += r; ep_max_x = std::max(ep_max_x, env.mario_x());
            replay.push({s, ns, a, r, d});
            s = ns; done = d;
            ++total_steps;

            if ((int)replay.size() >= warmup) {
                auto idx = replay.sample(batch);
                auto xs = Tensor::zeros({batch, S}, false), xns = Tensor::zeros({batch, S}, false);
                for (int b = 0; b < batch; ++b) {
                    const auto& tr = replay.buf[idx[b]];
                    for (int k = 0; k < S; ++k) { xs.data()[b * S + k] = tr.s[k]; xns.data()[b * S + k] = tr.ns[k]; }
                }
                auto q_ns_online = online.forward(xns);
                auto q_ns_target = target.forward(xns);
                auto q = online.forward(xs);
                auto tgt = Tensor::from(q.data(), {batch, A}, false);
                for (int b = 0; b < batch; ++b) {
                    const auto& tr = replay.buf[idx[b]];
                    int a_star = argmax_row(q_ns_online.data(), b * A, A);
                    float boot = tr.done ? 0.f : gamma * q_ns_target.data()[b * A + a_star];
                    tgt.data()[b * A + tr.a] = tr.r + boot;
                }
                auto loss = mul_scalar(sum(huber(sub(q, tgt))), 1.0f / batch);
                opt.zero_grad(); loss.backward(); clip_grads(params, 10.f); opt.step();
            }
            if (total_steps % target_sync == 0) target.copy_from(online);
        }

        recent_x.push_back(ep_max_x);
        if (recent_x.size() > 50) recent_x.pop_front();
        double avg_x = 0; for (int x : recent_x) avg_x += x; avg_x /= recent_x.size();
        if (ep_max_x > best_x) { best_x = ep_max_x; best.copy_from(online); best.save("mario_best.bin"); }

        if (ep % 10 == 0) {
            float eps = std::max(eps_end, eps_start - (eps_start - eps_end) * total_steps / eps_decay_steps);
            std::printf("ep %4d  ret %7.1f  max_x %4d  avg50_x %6.1f  best_x %4d  eps %.2f  steps %ld\n",
                        ep, ep_ret, ep_max_x, avg_x, best_x, eps, total_steps);
            std::fflush(stdout);
        }
    }
    std::printf("\nbest distance reached: x=%d (saved mario_best.bin)\n", best_x);
    return 0;
}
