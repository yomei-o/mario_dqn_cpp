#pragma once
// Generic DQN trainer, templated on the level Env (mario13::Env, mario14::Env, ...).
// The DQN core is level-agnostic; only the Env differs (start/win/reward), so we
// share one trainer instead of copy-pasting per level. Each level's train_*.cpp is
// then a ~3-line main. (train12.cpp predates this and keeps its own copy.)
//
// The Env must expose: STATE_DIM, ACTIONS, reset(), step(a,done), observation(),
// mario_x(), won(), area().  Warm-start: resume own 6-action ckpt -> a 6-action
// net (e.g. the 1-2 net) -> map a 5-action 1-1 net into 6 (NOOP zero-init).
#include "autograd.h"
#include "qnet.h"
#include "replay.h"
#include "mario_shared.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <deque>
#include <algorithm>
#include <chrono>
#include <thread>

#ifdef _WIN32
extern "C" __declspec(dllimport) unsigned int __stdcall timeBeginPeriod(unsigned int);
extern "C" __declspec(dllimport) unsigned int __stdcall timeEndPeriod(unsigned int);
#endif

namespace trn {

using namespace ag;
using namespace rl;

struct CpuThrottle {
    double util = 0.8; bool on = true;
    std::chrono::steady_clock::time_point mark;
    double busy_total = 0.0, slept_total = 0.0;
    CpuThrottle() {
        if (const char* e = std::getenv("MARIO_CPU")) util = std::atof(e);
        on = util > 0.0 && util < 1.0;
#ifdef _WIN32
        if (on) timeBeginPeriod(1);
#endif
        mark = std::chrono::steady_clock::now();
    }
#ifdef _WIN32
    ~CpuThrottle() { if (on) timeEndPeriod(1); }
#endif
    void maybe_sleep() {
        if (!on) return;
        auto now = std::chrono::steady_clock::now();
        double chunk = std::chrono::duration<double>(now - mark).count();
        if (chunk < 0.1) return;
        busy_total += chunk;
        double need = busy_total * (1.0 - util) / util - slept_total;
        if (need > 0.0) {
            auto s0 = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::chrono::duration<double>(need));
            slept_total += std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
        }
        mark = std::chrono::steady_clock::now();
    }
};

inline int argmax_row(const std::vector<float>& q, int off, int A) {
    int best = 0; for (int a = 1; a < A; ++a) if (q[off + a] > q[off + best]) best = a; return best;
}
inline int greedy_action(QNet& net, const std::vector<float>& s) {
    auto x = Tensor::from(s, {1, (int)s.size()}, false);
    return argmax_row(net.forward(x).data(), 0, net.adim);
}
inline void clip_grads(std::vector<Tensor>& ps, float max_norm) {
    double sq = 0; for (auto& p : ps) for (float g : p.grad()) sq += (double)g * g;
    float norm = (float)std::sqrt(sq);
    if (norm > max_norm) { float s = max_norm/(norm+1e-6f); for (auto& p : ps) for (float& g : p.grad()) g *= s; }
}
inline void ema_update(QNet& ema, QNet& online, float decay) {
    auto e = ema.params(), o = online.params();
    for (size_t i = 0; i < e.size(); ++i) {
        auto& ed = e[i].data(); auto& od = o[i].data();
        for (size_t j = 0; j < ed.size(); ++j) ed[j] = decay * ed[j] + (1.f - decay) * od[j];
    }
}
// Map a 5-action 1-1 net into this 6-action net (W3/b3 gain a NOOP column, zeroed).
inline bool warm_from_1_1(QNet& net, const std::string& path, int old_A) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    auto ps = net.params();
    auto rd_into = [&](Tensor& t) -> bool {
        int n = 0; if (std::fread(&n, 4, 1, f) != 1 || n != t.numel()) return false;
        return (int)std::fread(t.data().data(), sizeof(float), n, f) == n;
    };
    bool ok = rd_into(ps[0]) && rd_into(ps[1]) && rd_into(ps[2]) && rd_into(ps[3]);
    int H = net.hdim, A = net.adim, nW3 = 0;
    if (ok && std::fread(&nW3, 4, 1, f) == 1 && nW3 == H * old_A) {
        std::vector<float> oW3(nW3);
        if ((int)std::fread(oW3.data(), sizeof(float), nW3, f) == nW3) {
            auto& w3 = ps[4].data();
            for (int r = 0; r < H; ++r) for (int c = 0; c < A; ++c)
                w3[r*A+c] = (c < old_A) ? oW3[r*old_A+c] : 0.f;
            int nb3 = 0;
            if (std::fread(&nb3, 4, 1, f) == 1 && nb3 == old_A) {
                std::vector<float> ob3(nb3);
                if ((int)std::fread(ob3.data(), sizeof(float), nb3, f) == nb3) {
                    auto& b3 = ps[5].data();
                    for (int c = 0; c < A; ++c) b3[c] = (c < old_A) ? ob3[c] : 0.f;
                    std::fclose(f); return true;
                }
            }
        }
    }
    std::fclose(f); return false;
}

template <class Env>
long greedy_episode(QNet& net, Env& env, int* out_x = nullptr, bool* out_won = nullptr) {
    std::vector<float> s = env.reset();
    bool done = false; int mx = 0;
    int start_area = env.area(), max_area = start_area;
    while (!done) {
        int a = greedy_action(net, s);
        env.step(a, done); s = env.observation();
        mx = std::max(mx, env.mario_x());
        max_area = std::max(max_area, env.area());
    }
    if (out_x)   *out_x = mx;
    if (out_won) *out_won = env.won();
    return (long)(max_area - start_area) * 100000L + mx;
}

// Generic training entry. warm6 = a 6-action warm net (e.g. the 1-2 net);
// warm5 = a 5-action 1-1 net mapped up. Usage from a level main:
//   return trn::train_level<mario13::Env>(argc, argv, "1-3", "mario13_best.bin");
template <class Env>
int train_level(int argc, char** argv, const char* level, const char* default_out) {
    const char* rom = argc > 1 ? argv[1] : "Super Mario Bros (JU) (PRG 0).nes";
    int seed_val = argc > 2 ? std::atoi(argv[2]) : 0;
    std::string out_path  = argc > 3 ? argv[3] : default_out;
    std::string warm6     = argc > 4 ? argv[4] : "warmstarts/dqn12_1-2_x978_hid512.bin";
    float arg_eps = argc > 5 ? (float)std::atof(argv[5]) : -1.f;
    float arg_lr  = argc > 6 ? (float)std::atof(argv[6]) : -1.f;
    const char* warm5 = "warmstarts/bc_clear_x2370_hid512.bin";

    seed(seed_val);
    const int S = Env::STATE_DIM, A = Env::ACTIONS, HID = 512;
    const float gamma = 0.99f;
    const int batch = 32, warmup = 5000, target_sync = 2000, episodes = 8000, train_freq = 4;
    const float eps_end = 0.1f;

    QNet online(S, A, HID), target(S, A, HID), best(S, A, HID);
    Adam opt(online.params());
    auto params = online.params();
    Replay replay(100000);
    Env env;
    if (!env.init(rom)) { std::printf("env init failed\n"); return 1; }

    bool warm = online.load(out_path) || online.load(warm6) || warm_from_1_1(online, warm5, smb::N_ACTIONS);
    float eps_start = arg_eps >= 0.f ? arg_eps : (warm ? 0.4f : 1.0f);
    float eps_decay_steps = warm ? 200000.f : 300000.f;
    opt.lr = arg_lr > 0.f ? arg_lr : (warm ? 1.2e-4f : 2.5e-4f);
    target.copy_from(online); best.copy_from(online);
    QNet ema(S, A, HID); ema.copy_from(online);
    const float ema_decay = 0.999f;

    std::printf("== DQN x Super Mario Bros %s ==\n", level);
    std::printf("   seed=%d out=%s warm=%d | S=%d A=%d HID=%d | eps0=%.2f lr=%.1e\n",
                seed_val, out_path.c_str(), (int)warm, S, A, HID, eps_start, opt.lr);
    CpuThrottle throttle;
    std::printf("   cpu throttle: %s (util=%.2f)\n", throttle.on ? "on" : "off", throttle.util);
    { std::vector<float> s0 = env.reset();
      std::printf("   %s start ready: x=%d area=%d\n", level, env.mario_x(), env.area()); std::fflush(stdout); }

    long total_steps = 0; std::deque<int> recent_x;
    long best_metric = -1; int best_x = 0;
    if (warm) { int wx=0; bool ww=false; best_metric = greedy_episode(ema, env, &wx, &ww); best_x = wx;
        std::printf("   warm greedy: x=%d won=%d\n", wx, (int)ww); std::fflush(stdout); }
    bool have_win = false;

    for (int ep = 1; ep <= episodes; ++ep) {
        std::vector<float> s = env.reset();
        bool done = false; float ep_ret = 0; int ep_max_x = 0;
        while (!done) {
            float eps = std::max(eps_end, eps_start - (eps_start - eps_end) * total_steps / eps_decay_steps);
            int a = (ag::randf() < eps) ? (int)(ag::randf() * A) % A : greedy_action(online, s);
            bool d; float r = env.step(a, d);
            const std::vector<float>& ns = env.observation();
            ep_ret += r; ep_max_x = std::max(ep_max_x, env.mario_x());
            replay.push({s, ns, a, r, d});
            s = ns; done = d; ++total_steps;

            if ((int)replay.size() >= warmup && total_steps % train_freq == 0) {
                auto idx = replay.sample(batch);
                auto xs = Tensor::zeros({batch, S}, false), xns = Tensor::zeros({batch, S}, false);
                for (int b = 0; b < batch; ++b) {
                    const auto& tr = replay.buf[idx[b]];
                    for (int k = 0; k < S; ++k) { xs.data()[b*S+k]=tr.s[k]; xns.data()[b*S+k]=tr.ns[k]; }
                }
                auto q_ns_online = online.forward(xns);
                auto q_ns_target = target.forward(xns);
                auto q = online.forward(xs);
                auto tgt = Tensor::from(q.data(), {batch, A}, false);
                for (int b = 0; b < batch; ++b) {
                    const auto& tr = replay.buf[idx[b]];
                    int a_star = argmax_row(q_ns_online.data(), b*A, A);
                    float boot = tr.done ? 0.f : gamma * q_ns_target.data()[b*A + a_star];
                    tgt.data()[b*A + tr.a] = tr.r + boot;
                }
                auto loss = mul_scalar(sum(huber(sub(q, tgt))), 1.0f / batch);
                opt.zero_grad(); loss.backward();
                clip_grads(params, 10.f); opt.step();
                ema_update(ema, online, ema_decay);
            }
            if (total_steps % target_sync == 0) target.copy_from(online);
            throttle.maybe_sleep();
        }
        if (env.won()) have_win = true;

        recent_x.push_back(ep_max_x);
        if (recent_x.size() > 50) recent_x.pop_front();
        double avg_x = 0; for (int x : recent_x) avg_x += x; avg_x /= recent_x.size();
        if (ep % 10 == 0) { std::printf("  [hb] ep %d steps %ld replay %d last_x %d\n",
                                        ep, total_steps, (int)replay.size(), ep_max_x); std::fflush(stdout); }
        if (ep % 25 == 0 && (int)replay.size() >= warmup) {
            int gx=0; bool gw=false;
            long metric = greedy_episode(ema, env, &gx, &gw);
            if (metric > best_metric) { best_metric = metric; best_x = gx; best.copy_from(ema); best.save(out_path.c_str()); }
            ema.save((std::string(default_out) + ".latest_" + std::to_string(seed_val)).c_str());
            float eps = std::max(eps_end, eps_start - (eps_start - eps_end) * total_steps / eps_decay_steps);
            std::printf("ep %4d  ret %7.1f  train_max_x %4d  avg50 %6.1f  GREEDY_x %4d won %d  best_x %4d  eps %.2f  steps %ld  win=%d\n",
                        ep, ep_ret, ep_max_x, avg_x, gx, (int)gw, best_x, eps, total_steps, (int)have_win);
            std::fflush(stdout);
        }
    }
    std::printf("\nbest greedy x=%d (saved %s)\n", best_x, out_path.c_str());
    return 0;
}

}  // namespace trn
