#pragma once
// Shared plumbing of the Ape-X style trainers (mario_gpu for the 1-1 env,
// smb_train for the all-stage env): action-file I/O, greedy helpers, actor
// process spawn, episode files, the n-step replay and weight publishing.
// Env-specific helpers are templates; `prog(env)` is the progress measure
// (x for the 1-1 env; smbx::Env overrides it via ADL with sub-area progress).
#include "autograd.h"
#include "qnet.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <filesystem>
#include "dqn_learner.h"
#ifdef _WIN32
#include <process.h>
#else
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;
#endif

namespace apex {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static const int N_STEP = 5;
static const float GAMMA = 0.99f;
static const float REWARD_SCALE = 0.1f;  // env reward is ~[-50,+125]/step; keep Q near the margin's scale

template <class Env> int prog(const Env& env) { return env.mario_x(); }

inline double secs_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------- small helpers
inline bool save_actions(const std::string& path, const std::vector<uint8_t>& acts) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    uint32_t n = (uint32_t)acts.size();
    std::fwrite(&n, 4, 1, f);
    if (n) std::fwrite(acts.data(), 1, n, f);
    std::fclose(f);
    return true;
}
inline bool load_actions(const std::string& path, std::vector<uint8_t>& acts) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t n = 0;
    if (std::fread(&n, 4, 1, f) != 1) { std::fclose(f); return false; }
    acts.resize(n);
    size_t got = n ? std::fread(acts.data(), 1, n, f) : 0;
    std::fclose(f);
    return got == n;
}

inline int argmax(const float* q, int A) {
    int b = 0;
    for (int a = 1; a < A; ++a) if (q[a] > q[b]) b = a;
    return b;
}
inline int greedy_action(rl::QNet& net, const std::vector<float>& s) {
    auto x = ag::Tensor::from(s, {1, (int)s.size()}, false);
    return argmax(net.forward(x).data().data(), net.adim);
}

inline void qnet_get(rl::QNet& net, std::vector<std::vector<float>>& p) {
    auto ps = net.params();
    p.resize(ps.size());
    for (size_t i = 0; i < ps.size(); ++i) p[i] = ps[i].data();
}
inline void qnet_set(rl::QNet& net, const std::vector<std::vector<float>>& p) {
    auto ps = net.params();
    for (size_t i = 0; i < ps.size(); ++i) ps[i].data() = p[i];
}


// ---------------------------------------------------------------- process spawn
inline void spawn_self(const std::string& exe, const std::vector<std::string>& args) {
#ifdef _WIN32
    // _spawnv does not quote: wrap every argument (the ROM name has spaces).
    std::vector<std::string> q;
    q.push_back("\"" + exe + "\"");
    for (auto& a : args) q.push_back("\"" + a + "\"");
    std::vector<const char*> argv;
    for (auto& s : q) argv.push_back(s.c_str());
    argv.push_back(nullptr);
    if (_spawnv(_P_NOWAIT, exe.c_str(), argv.data()) == -1) std::printf("spawn failed: %s\n", exe.c_str());
#else
    std::vector<std::string> all{exe};
    all.insert(all.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (auto& s : all) argv.push_back(const_cast<char*>(s.c_str()));
    argv.push_back(nullptr);
    pid_t pid;
    if (posix_spawn(&pid, exe.c_str(), nullptr, nullptr, argv.data(), environ) != 0)
        std::printf("spawn failed: %s\n", exe.c_str());
#endif
}

// ---------------------------------------------------------------- episode files
// "MEP1" | i32 S, T, actor, start_ckpt, max_x, won | (T+1)*S f32 obs | T i32 act | T f32 rew
struct Episode {
    int S = 0, T = 0, actor = 0, start_ckpt = -1, max_x = 0, won = 0;
    std::vector<float> obs;
    std::vector<int> act;
    std::vector<float> rew;
};

inline bool write_episode(const std::string& path, const Episode& e) {
    std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    int hdr[6] = {e.S, e.T, e.actor, e.start_ckpt, e.max_x, e.won};
    std::fwrite("MEP1", 1, 4, f);
    std::fwrite(hdr, sizeof(int), 6, f);
    std::fwrite(e.obs.data(), sizeof(float), e.obs.size(), f);
    std::fwrite(e.act.data(), sizeof(int), e.act.size(), f);
    std::fwrite(e.rew.data(), sizeof(float), e.rew.size(), f);
    std::fclose(f);
    std::error_code ec;
    fs::rename(tmp, path, ec);
    return !ec;
}

inline bool read_episode(const std::string& path, Episode& e) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char mg[4]; int hdr[6];
    bool ok = std::fread(mg, 1, 4, f) == 4 && std::memcmp(mg, "MEP1", 4) == 0 &&
              std::fread(hdr, sizeof(int), 6, f) == 6;
    if (ok) {
        e.S = hdr[0]; e.T = hdr[1]; e.actor = hdr[2]; e.start_ckpt = hdr[3]; e.max_x = hdr[4]; e.won = hdr[5];
        e.obs.resize((size_t)(e.T + 1) * e.S); e.act.resize(e.T); e.rew.resize(e.T);
        ok = std::fread(e.obs.data(), sizeof(float), e.obs.size(), f) == e.obs.size() &&
             std::fread(e.act.data(), sizeof(int), e.act.size(), f) == e.act.size() &&
             std::fread(e.rew.data(), sizeof(float), e.rew.size(), f) == e.rew.size();
    }
    std::fclose(f);
    return ok;
}

// ---------------------------------------------------------------- replay (n-step)
struct NStepReplay {
    int S; size_t cap, head = 0, size = 0;
    std::vector<float> s, ns, ret, gn;
    std::vector<int> a;
    NStepReplay(int S_, size_t cap_) : S(S_), cap(cap_) {
        s.resize(cap * S); ns.resize(cap * S); ret.resize(cap); gn.resize(cap); a.resize(cap);
    }
    // Every episode ends with done (death / flag / stall / time limit), so the
    // n-step window is cut at the end and the bootstrap dropped there.
    void add_episode(const Episode& e) {
        for (int t = 0; t < e.T; ++t) {
            int m = std::min(N_STEP, e.T - t);
            float r = 0.f, g = 1.f;
            for (int k = 0; k < m; ++k) { r += g * e.rew[t + k] * REWARD_SCALE; g *= GAMMA; }
            bool terminal = (t + m == e.T);
            size_t i = head;
            std::copy_n(&e.obs[(size_t)t * S], S, &s[i * S]);
            std::copy_n(&e.obs[(size_t)(t + m) * S], S, &ns[i * S]);
            a[i] = e.act[t]; ret[i] = r; gn[i] = terminal ? 0.f : g;
            head = (head + 1) % cap;
            size = std::min(size + 1, cap);
        }
    }
    void fill_row(size_t i, rl::Batch& b, int row, float demo) const {
        std::copy_n(&s[i * S], S, &b.s[(size_t)row * S]);
        std::copy_n(&ns[i * S], S, &b.ns[(size_t)row * S]);
        b.a[row] = a[i]; b.ret[row] = ret[i]; b.gn[row] = gn[i]; b.demo[row] = demo;
    }
};


// Greedy run from the level start. Returns max x; fills the action sequence.
template <class Net, class Env>
int greedy_run(Net& net, Env& env, bool& won, std::vector<uint8_t>* acts = nullptr) {
    std::vector<float> s = env.reset();
    bool done = false; int mx = 0;
    if (acts) acts->clear();
    while (!done) {
        int a = greedy_action(net, s);
        env.step(a, done);
        if (acts) acts->push_back((uint8_t)a);
        s = env.observation();
        mx = std::max(mx, prog(env));
    }
    won = env.won();
    return mx;
}

// Replay an action sequence from the start; returns the step index at which the
// flag was reached (+1 = sequence length up to the win), or -1 if it never won.
template <class Env>
int replay_win_len(Env& env, const std::vector<uint8_t>& acts, int* max_x = nullptr) {
    env.reset();
    bool done = false; int mx = 0;
    for (size_t t = 0; t < acts.size(); ++t) {
        env.step(acts[t], done);
        mx = std::max(mx, prog(env));
        if (done) { if (max_x) *max_x = mx; return env.won() ? (int)t + 1 : -1; }
    }
    if (max_x) *max_x = mx;
    return -1;
}

// Run a from-start action sequence and record it as an Episode (for the demo buffer).
template <class Env>
Episode demo_episode(Env& env, const std::vector<uint8_t>& acts) {
    Episode e; e.S = Env::STATE_DIM;
    std::vector<float> s = env.reset();
    e.obs.insert(e.obs.end(), s.begin(), s.end());
    bool done = false;
    for (size_t t = 0; t < acts.size() && !done; ++t) {
        float r = env.step(acts[t], done);
        const auto& ns = env.observation();
        e.obs.insert(e.obs.end(), ns.begin(), ns.end());
        e.act.push_back(acts[t]); e.rew.push_back(r);
        e.max_x = std::max(e.max_x, prog(env));
    }
    e.T = (int)e.act.size(); e.won = env.won();
    return e;
}

template <class Net>
void publish(Net& net, int ver, const std::string& dir) {
    std::string tmp = dir + "/weights.tmp", dst = dir + "/weights.bin";
    net.save(tmp);
    for (int tries = 0; tries < 50; ++tries) {   // Windows: an actor may hold the file open for a moment
        std::error_code ec;
        fs::rename(tmp, dst, ec);
        if (!ec) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (FILE* f = std::fopen((dir + "/weights.ver").c_str(), "w")) { std::fprintf(f, "%d\n", ver); std::fclose(f); }
}

}  // namespace apex
