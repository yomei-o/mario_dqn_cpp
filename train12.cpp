// DQN trainer for Super Mario Bros World 1-2. SEPARATE from the 1-1 trainer
// (mario_dqn.cpp) but reuses the same DQN core (QNet + Adam + replay + target +
// EMA) and the shared env core. Because the observation layout is identical
// across levels, we WARM-START from a 1-1 net: the "run right, jump obstacles /
// enemies" skill transfers, giving 1-2 a big head start. 1-2 adds the exit-pipe
// milestone (env's PIPE_BONUS) and captures a from-boot clear demo on a flag win.
//
// Usage: mario12_dqn [ROM] [seed] [out.bin] [warmstart.bin] [eps_start] [lr]
#include "autograd.h"
#include "qnet.h"
#include "replay.h"
#include "mario12.h"
#include <cstdio>
#include <cstring>
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

using namespace ag;
using namespace rl;

// CPU throttle (same self-calibrating design as mario_dqn.cpp; env MARIO_CPU).
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
    double sq = 0; for (auto& p : ps) for (float g : p.grad()) sq += (double)g * g;
    float norm = (float)std::sqrt(sq);
    if (norm > max_norm) { float s = max_norm / (norm + 1e-6f); for (auto& p : ps) for (float& g : p.grad()) g *= s; }
}
static void ema_update(QNet& ema, QNet& online, float decay) {
    auto e = ema.params(), o = online.params();
    for (size_t i = 0; i < e.size(); ++i) {
        auto& ed = e[i].data(); auto& od = o[i].data();
        for (size_t j = 0; j < ed.size(); ++j) ed[j] = decay * ed[j] + (1.f - decay) * od[j];
    }
}
// Warm-start a 6-action 1-2 net from a 5-action 1-1 net: W1/b1/W2/b2 are identical
// dims (copy), W3/b3 gain one column for A_NOOP -- copy the 5 old action columns and
// ZERO the new NOOP column so the transferred policy is unchanged at init (NOOP just
// starts neutral and trains up). Assumes params order W1,b1,W2,b2,W3,b3.
static bool warm_from_1_1(QNet& net, const std::string& path, int old_A) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    auto ps = net.params();
    auto rd_into = [&](Tensor& t) -> bool {
        int n = 0; if (std::fread(&n, 4, 1, f) != 1 || n != t.numel()) return false;
        return (int)std::fread(t.data().data(), sizeof(float), n, f) == n;
    };
    bool ok = rd_into(ps[0]) && rd_into(ps[1]) && rd_into(ps[2]) && rd_into(ps[3]);
    int H = net.hdim, A = net.adim;
    int nW3 = 0;
    if (ok && std::fread(&nW3, 4, 1, f) == 1 && nW3 == H * old_A) {
        std::vector<float> oW3(nW3);
        if ((int)std::fread(oW3.data(), sizeof(float), nW3, f) == nW3) {
            auto& w3 = ps[4].data();
            for (int r = 0; r < H; ++r) for (int c = 0; c < A; ++c)
                w3[r * A + c] = (c < old_A) ? oW3[r * old_A + c] : 0.f;
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

static bool save_actions(const std::string& path, const std::vector<uint8_t>& a) {
    FILE* f = std::fopen(path.c_str(), "wb"); if (!f) return false;
    uint32_t n = (uint32_t)a.size(); std::fwrite(&n, 4, 1, f);
    if (n) std::fwrite(a.data(), 1, n, f); std::fclose(f); return true;
}

// One fully-greedy rollout from the 1-2 start. Returns furthest area-progress as
// (area*100000 + max_x) so a rollout that entered the exit pipe always ranks above
// one that only went far in the first area. Reports won / entered-pipe / raw x.
static long greedy_episode(QNet& net, mario12::Env& env, int* out_x = nullptr,
                           bool* out_pipe = nullptr, bool* out_won = nullptr) {
    std::vector<float> s = env.reset();
    bool done = false; int mx = 0; bool pipe = false;
    int start_area = env.area(), max_area = start_area;
    while (!done) {
        int a = greedy_action(net, s);
        env.step(a, done);
        s = env.observation();
        mx = std::max(mx, env.mario_x());
        if (env.entered_pipe()) pipe = true;
        max_area = std::max(max_area, env.area());
    }
    if (out_x)    *out_x = mx;
    if (out_pipe) *out_pipe = pipe;
    if (out_won)  *out_won = env.won();
    return (long)(max_area - start_area) * 100000L + mx;
}

int main(int argc, char** argv) {
    const char* rom = argc > 1 ? argv[1] : "Super Mario Bros (JU) (PRG 0).nes";
    int seed_val = argc > 2 ? std::atoi(argv[2]) : 0;
    std::string out_path = argc > 3 ? argv[3] : "mario12_best.bin";
    std::string warm_path = argc > 4 ? argv[4] : "warmstarts/bc_clear_x2370_hid512.bin";
    float arg_eps = argc > 5 ? (float)std::atof(argv[5]) : -1.f;   // per-worker exploration
    float arg_lr  = argc > 6 ? (float)std::atof(argv[6]) : -1.f;

    seed(seed_val);
    const int S = mario12::Env::STATE_DIM, A = mario12::N_ACTIONS, HID = 512;
    const float gamma = 0.99f;
    const int batch = 32, warmup = 5000, target_sync = 2000, episodes = 8000, train_freq = 4;
    const float eps_end = 0.1f;

    QNet online(S, A, HID), target(S, A, HID), best(S, A, HID);
    Adam opt(online.params());
    auto params = online.params();
    Replay replay(100000);
    mario12::Env env;
    if (!env.init(rom)) { std::printf("env init failed\n"); return 1; }

    // Warm-start from a 1-1 net (identical obs layout): the run-right + jump skill
    // transfers. Prefer this worker's own 6-action checkpoint (resume); else map the
    // 5-action 1-1 net into 6 actions (NOOP column zero-init).
    bool warm = online.load(out_path) || warm_from_1_1(online, warm_path, smb::N_ACTIONS);
    float eps_start = arg_eps >= 0.f ? arg_eps : (warm ? 0.35f : 1.0f);  // explore 1-2's new terrain, keep transfer
    float eps_decay_steps = warm ? 200000.f : 300000.f;
    opt.lr = arg_lr > 0.f ? arg_lr : (warm ? 1.2e-4f : 2.5e-4f);         // gentle when warm (don't wreck the transfer)
    target.copy_from(online); best.copy_from(online);
    QNet ema(S, A, HID); ema.copy_from(online);
    const float ema_decay = 0.999f;

    std::printf("== DQN x Super Mario Bros 1-2 (underground) ==\n");
    std::printf("   seed=%d out=%s warm=%d(%s) | S=%d A=%d HID=%d | eps0=%.2f lr=%.1e\n",
                seed_val, out_path.c_str(), (int)warm, warm_path.c_str(), S, A, HID, eps_start, opt.lr);
    std::fflush(stdout);

    CpuThrottle throttle;
    std::printf("   cpu throttle: %s (util=%.2f)\n", throttle.on ? "on" : "off", throttle.util);

    // Prime the (expensive) 1-2 start once so the first episode isn't a surprise.
    { std::vector<float> s0 = env.reset();
      std::printf("   1-2 start ready: x=%d area=%d entry_demo=%d actions\n",
                  env.mario_x(), env.area(), env.entry_demo_len()); std::fflush(stdout); }

    long total_steps = 0;
    std::deque<int> recent_x;
    long best_metric = -1; int best_x = 0;
    if (warm) { bool wp=false, ww=false; int wx=0;
        best_metric = greedy_episode(ema, env, &wx, &wp, &ww); best_x = wx;
        std::printf("   warm greedy: x=%d pipe=%d won=%d\n", wx, (int)wp, (int)ww); std::fflush(stdout); }

    const std::string win_path = "demo_clear_1-2_" + std::to_string(seed_val) + ".bin";
    bool have_clear = false; int best_clear_len = 1 << 30;

    for (int ep = 1; ep <= episodes; ++ep) {
        std::vector<float> s = env.reset();
        bool done = false; float ep_ret = 0; int ep_max_x = 0; bool ep_pipe = false;
        std::vector<uint8_t> ep_actions;
        while (!done) {
            float eps = std::max(eps_end, eps_start - (eps_start - eps_end) * total_steps / eps_decay_steps);
            int a = (ag::randf() < eps) ? (int)(ag::randf() * A) % A : greedy_action(online, s);
            bool d; float r = env.step(a, d);
            ep_actions.push_back((uint8_t)a);
            const std::vector<float>& ns = env.observation();
            ep_ret += r; ep_max_x = std::max(ep_max_x, env.mario_x());
            if (env.entered_pipe()) ep_pipe = true;
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

        // Flag reached in 1-2: save this episode's 1-2 actions. A full from-boot
        // clear replays deterministically as demo_clear_1-1.bin ++ this file (the
        // 1-2 start is reached by clearing 1-1). Kept only when it's a new clear
        // that is shorter (more efficient) than the best so far.
        if (env.won() && (!have_clear || (int)ep_actions.size() < best_clear_len)) {
            best_clear_len = (int)ep_actions.size();
            save_actions(win_path, ep_actions);
            have_clear = true;
            std::printf("FLAG(1-2) ep %d  %d 1-2 actions -> %s\n", ep, (int)ep_actions.size(), win_path.c_str());
            std::fflush(stdout);
        }

        recent_x.push_back(ep_max_x);
        if (recent_x.size() > 50) recent_x.pop_front();
        double avg_x = 0; for (int x : recent_x) avg_x += x; avg_x /= recent_x.size();

        if (ep % 10 == 0) {
            std::printf("  [hb] ep %d steps %ld replay %d last_x %d pipe %d\n",
                        ep, total_steps, (int)replay.size(), ep_max_x, (int)ep_pipe);
            std::fflush(stdout);
        }
        if (ep % 25 == 0 && (int)replay.size() >= warmup) {
            int gx=0; bool gp=false, gw=false;
            long metric = greedy_episode(ema, env, &gx, &gp, &gw);
            if (metric > best_metric) {
                best_metric = metric; best_x = gx;
                best.copy_from(ema); best.save(out_path.c_str());
            }
            ema.save(("mario12_latest_" + std::to_string(seed_val) + ".bin").c_str());
            float eps = std::max(eps_end, eps_start - (eps_start - eps_end) * total_steps / eps_decay_steps);
            std::printf("ep %4d  ret %7.1f  train_max_x %4d  avg50 %6.1f  GREEDY_x %4d pipe %d won %d  best_x %4d  eps %.2f  steps %ld  clear=%d\n",
                        ep, ep_ret, ep_max_x, avg_x, gx, (int)gp, (int)gw, best_x, eps, total_steps, (int)have_clear);
            std::fflush(stdout);
        }
    }
    std::printf("\nbest greedy x=%d (saved %s)\n", best_x, out_path.c_str());
    return 0;
}
