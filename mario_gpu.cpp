// GPU DQfD trainer for Super Mario Bros 1-1 (Ape-X layout: CPU actors + one learner).
//
// The NES core is a global singleton and the emulator -- not the net -- is the
// bottleneck, so experience comes from N actor PROCESSES (one emulator each,
// eps-greedy with their own epsilon, half the episodes started from curriculum
// checkpoints along a clear demo). A single learner process ingests their
// episodes, turns them into n-step transitions, and trains the Q-net in big
// minibatches on the GPU (dqn_learner_cuda.cu; CPU fallback = same math). The
// loss is DQfD: Double-DQN TD + a large-margin term that pins the demo's action
// above the others on the demo states, so the greedy policy starts as the demo
// and RL teaches it to recover from off-demo states.
//
// Actors and learner talk through files in gpu_run/ (portable: works the same
// on Kaggle/Linux and Windows): the learner publishes weights.bin, actors drop
// finished episodes as ep/*.ep. The learner checks the greedy run from the level
// start (the honest metric) every eval period and saves:
//   mario_gpu_best.bin   farthest greedy net,  mario_gpu_clear.bin  first net whose greedy run clears 1-1
//   gpu_run/greedy_clear.bin  that run's action sequence (play it with `mario_gpu record`)
//
//   mario_gpu searchdemo ROM out.bin              build a clean clear demo by search (run + deliberate jumps)
//   mario_gpu cleandemo ROM in.bin out.bin        canonicalize a clear demo (non-essential actions -> run)
//   mario_gpu train ROM demo.bin [actors] [minutes] [warm.bin]
//   mario_gpu eval ROM net.bin                    greedy run from the start
//   mario_gpu diagnose ROM net.bin [step] [demo]  why the greedy run stops: REWARD / OBSERVATION / LEARNING
//   mario_gpu record ROM net.bin out.bin          MRUN video of the greedy run (tools/make_gif.sh)
//   mario_gpu parity                              CPU vs CUDA learner on identical batches
//   mario_gpu bench ROM                           emulator steps/sec
// Env: MARIO_LEARNER=cpu forces the CPU learner; MARIO_RATIO = replay ratio (default 16);
//      MARIO_LR = RL learning rate (default 1e-4); MARIO_EVAL_SEC = greedy eval period (default 30).
#include "autograd.h"
#include "qnet.h"
#include "mario.h"
#include "nes.h"
#include "dqn_learner.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <deque>
#include <algorithm>
#include <utility>
#include <tuple>
#include <map>
#include <chrono>
#include <thread>
#include <filesystem>
#ifdef _WIN32
#include <process.h>
#else
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static const int HID = 512;              // same width as the deployed nets (wasm.cpp)
static const int N_STEP = 5;
static const float GAMMA = 0.99f;
static const float REWARD_SCALE = 0.1f;  // env reward is ~[-50,+125]/step; keep Q near the margin's scale
static const char* RUN_DIR = "gpu_run";

static double secs_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------- small helpers
static bool save_actions(const std::string& path, const std::vector<uint8_t>& acts) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    uint32_t n = (uint32_t)acts.size();
    std::fwrite(&n, 4, 1, f);
    if (n) std::fwrite(acts.data(), 1, n, f);
    std::fclose(f);
    return true;
}
static bool load_actions(const std::string& path, std::vector<uint8_t>& acts) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t n = 0;
    if (std::fread(&n, 4, 1, f) != 1) { std::fclose(f); return false; }
    acts.resize(n);
    size_t got = n ? std::fread(acts.data(), 1, n, f) : 0;
    std::fclose(f);
    return got == n;
}

static int argmax(const float* q, int A) {
    int b = 0;
    for (int a = 1; a < A; ++a) if (q[a] > q[b]) b = a;
    return b;
}
static int greedy_action(rl::QNet& net, const std::vector<float>& s) {
    auto x = ag::Tensor::from(s, {1, (int)s.size()}, false);
    return argmax(net.forward(x).data().data(), net.adim);
}

static void qnet_get(rl::QNet& net, std::vector<std::vector<float>>& p) {
    auto ps = net.params();
    p.resize(ps.size());
    for (size_t i = 0; i < ps.size(); ++i) p[i] = ps[i].data();
}
static void qnet_set(rl::QNet& net, const std::vector<std::vector<float>>& p) {
    auto ps = net.params();
    for (size_t i = 0; i < ps.size(); ++i) ps[i].data() = p[i];
}

// Greedy run from the level start. Returns max x; fills the action sequence.
static int greedy_run(rl::QNet& net, mario::Env& env, bool& won, std::vector<uint8_t>* acts = nullptr) {
    std::vector<float> s = env.reset();
    bool done = false; int mx = 0;
    if (acts) acts->clear();
    while (!done) {
        int a = greedy_action(net, s);
        env.step(a, done);
        if (acts) acts->push_back((uint8_t)a);
        s = env.observation();
        mx = std::max(mx, env.mario_x());
    }
    won = env.won();
    return mx;
}

// Replay an action sequence from the start; returns the step index at which the
// flag was reached (+1 = sequence length up to the win), or -1 if it never won.
static int replay_win_len(mario::Env& env, const std::vector<uint8_t>& acts, int* max_x = nullptr) {
    env.reset();
    bool done = false; int mx = 0;
    for (size_t t = 0; t < acts.size(); ++t) {
        env.step(acts[t], done);
        mx = std::max(mx, env.mario_x());
        if (done) { if (max_x) *max_x = mx; return env.won() ? (int)t + 1 : -1; }
    }
    if (max_x) *max_x = mx;
    return -1;
}

// ---------------------------------------------------------------- process spawn
static void spawn_self(const std::string& exe, const std::vector<std::string>& args) {
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

static bool write_episode(const std::string& path, const Episode& e) {
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

static bool read_episode(const std::string& path, Episode& e) {
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

// Run a from-start action sequence and record it as an Episode (for the demo buffer).
static Episode demo_episode(mario::Env& env, const std::vector<uint8_t>& acts) {
    Episode e; e.S = mario::Env::STATE_DIM;
    std::vector<float> s = env.reset();
    e.obs.insert(e.obs.end(), s.begin(), s.end());
    bool done = false;
    for (size_t t = 0; t < acts.size() && !done; ++t) {
        float r = env.step(acts[t], done);
        const auto& ns = env.observation();
        e.obs.insert(e.obs.end(), ns.begin(), ns.end());
        e.act.push_back(acts[t]); e.rew.push_back(r);
        e.max_x = std::max(e.max_x, env.mario_x());
    }
    e.T = (int)e.act.size(); e.won = env.won();
    return e;
}

// ---------------------------------------------------------------- modes
static int bench(const char* rom) {
    mario::Env env;
    if (!env.init(rom)) return 1;
    env.reset();
    auto t0 = Clock::now();
    long steps = 0;
    while (secs_since(t0) < 5.0) {
        bool done = false;
        env.reset();
        while (!done) { env.step(ag::randf() < 0.3f ? (int)(ag::randf() * mario::N_ACTIONS) % mario::N_ACTIONS : mario::A_RUNRIGHT, done); ++steps; }
    }
    double dt = secs_since(t0);
    std::printf("emulator: %.0f env steps/s (%.0f frames/s)\n", steps / dt, steps * mario::Env::FRAME_SKIP / dt);
    return 0;
}

// Canonicalize a clear demo: every action that is not essential to the clear is
// replaced by plain run-right (the replay is deterministic, so "essential" is
// exactly "replacing it breaks the clear"). Random exploration leaves jumps and
// hops scattered over flat ground; those give contradicting labels for similar
// states, which is what kept behavior cloning from fitting the demo.
static int cleandemo(const char* rom, const char* in, const char* out) {
    mario::Env env;
    if (!env.init(rom)) return 1;
    std::vector<uint8_t> acts;
    if (!load_actions(in, acts)) { std::printf("cannot read %s\n", in); return 1; }
    int L = replay_win_len(env, acts);
    if (L < 0) { std::printf("%s does not clear 1-1\n", in); return 1; }
    acts.resize(L);
    auto count_run = [&]() { int c = 0; for (auto a : acts) c += a == mario::A_RUNRIGHT; return c; };
    std::printf("demo %s: %d actions to the flag, %d already run-right\n", in, L, count_run());
    auto t0 = Clock::now();
    for (int pass = 0; pass < 2; ++pass) {
        int changed = 0;
        for (size_t i = 0; i < acts.size(); ++i) {
            if (acts[i] == mario::A_RUNRIGHT) continue;
            auto trial = acts;
            trial[i] = (uint8_t)mario::A_RUNRIGHT;
            int l = replay_win_len(env, trial);
            if (l > 0) { trial.resize(l); acts.swap(trial); ++changed; }
        }
        std::printf("  pass %d: %d actions -> run-right, length %zu, %d run-right (%.0fs)\n",
                    pass, changed, acts.size(), count_run(), secs_since(t0));
        std::fflush(stdout);
        if (!changed) break;
    }
    int mx = 0;
    if (replay_win_len(env, acts, &mx) < 0) { std::printf("internal error: cleaned demo no longer clears\n"); return 1; }
    save_actions(out, acts);
    std::printf("wrote %s (%zu actions, clears at x=%d)\n", out, acts.size(), mx);
    return 0;
}

// Build a clean, "human-style" clear demo by search: hold RUN, and when a run
// dies or gets stuck, try a running jump (RUNJUMP held d steps) started s steps
// before the last progress point, latest start first; commit the one that gets
// farthest within a short horizon (+100 if still alive). If no jump improves,
// undo the previous jump and forbid it (backtracking). The result is RUN
// everywhere except deliberate jumps in front of obstacles -- a policy that is a
// function of what the agent sees, unlike a random-exploration capture.
static int searchdemo(const char* rom, const char* out) {
    const int R = mario::A_RUNRIGHT;
#if MARIO_ACTIONS == 3
    const int J = mario::A_RUNJUMP;
#else
    const int J = mario::A_RIGHT_AB;
#endif
    const int durs[] = {2, 4, 6, 9, 13, 18};
    const int horizon = 60;
    mario::Env env;
    if (!env.init(rom)) return 1;
    env.reset();
    using Snap = mario::Env::Snapshot;
    std::vector<uint8_t> C;                 // committed actions
    std::vector<Snap> S(1);                 // S[i] = state before C[i]
    env.snapshot(S[0]);
    struct Decision { size_t base; int s, d; };
    std::vector<Decision> stack;
    std::vector<std::tuple<size_t, int, int>> tabu;
    auto t0 = Clock::now();
    long sim = 0;
    for (int iter = 0; iter < 2000; ++iter) {
        size_t L = C.size();
        env.restore(S[L]);
        // Base rollout: RUN until the episode ends; remember every state and the last progress step.
        std::vector<Snap> run(1);
        env.snapshot(run[0]);
        bool done = false;
        int mx = env.mario_x(), last_prog = 0;
        while (!done) {
            env.step(R, done); ++sim;
            run.emplace_back(); env.snapshot(run.back());
            if (env.mario_x() > mx) { mx = env.mario_x(); last_prog = (int)run.size() - 1; }
        }
        if (env.won()) {
            C.insert(C.end(), run.size() - 1, (uint8_t)R);
            int L2 = replay_win_len(env, C);
            if (L2 < 0) { std::printf("internal error: search result does not replay\n"); return 1; }
            C.resize(L2);
            save_actions(out, C);
            std::printf("CLEAR: %zu actions (%d jumps) -> %s  [%.0fs, %ld sim steps]\n",
                        C.size(), (int)stack.size(), out, secs_since(t0), sim);
            return 0;
        }
        int D = (int)run.size() - 1;
        int death_x = mx;
        auto try_jump = [&](int s, int d) -> double {
            if (s < 0 || s >= (int)run.size() || d < 1) return -1;
            if (std::find(tabu.begin(), tabu.end(), std::make_tuple(L, s, d)) != tabu.end()) return -1;
            env.restore(run[s]);
            bool dn = false; int m = 0;
            for (int k = 0; k < d && !dn; ++k) { env.step(J, dn); ++sim; m = std::max(m, env.mario_x()); }
            for (int h = 0; h < horizon && !dn; ++h) { env.step(R, dn); ++sim; m = std::max(m, env.mario_x()); }
            return env.won() ? 1e9 : m + (dn ? 0 : 100);
        };
        const int pass = death_x + 164;              // got past the obstacle with room to spare
        int best_s = -1, best_d = 0; double best_score = death_x;
        int s_hi = std::min(D - 1, last_prog + 5), s_lo = std::max(0, last_prog - 40);
        for (int s = s_hi; s >= s_lo; --s) {
            for (int d : durs) {
                double score = try_jump(s, d);
                if (score > best_score) { best_score = score; best_s = s; best_d = d; }
            }
            if (best_score >= pass) break;           // latest start that clears it
        }
        // The latest working start is knife-edge: one step later fails. A net
        // imitating the demo is off by a step now and then, so move to the (s, d)
        // whose neighbours (s+-1, d+-1) also pass -- a demo that tolerates that.
        if (best_score >= pass && best_score < 1e9) {
            std::map<std::pair<int, int>, bool> ok;
            auto passes = [&](int s, int d) {
                auto k = std::make_pair(s, d);
                auto it = ok.find(k);
                if (it != ok.end()) return it->second;
                bool r = try_jump(s, d) >= pass;
                ok[k] = r;
                return r;
            };
            int s0 = best_s, d0 = best_d, best_rob = -1;
            for (int s = s0; s >= std::max(0, s0 - 6); --s)
                for (int d = std::max(1, d0 - 2); d <= d0 + 2; ++d) {
                    if (!passes(s, d)) continue;
                    int rob = 0;
                    for (int ds = -1; ds <= 1; ++ds)
                        for (int dd = -1; dd <= 1; ++dd) rob += passes(s + ds, d + dd);
                    if (rob > best_rob) { best_rob = rob; best_s = s; best_d = d; }
                }
            std::printf("    robust pick: start %d->%d hold %d->%d (%d/9 neighbours pass)\n", s0, best_s, d0, best_d, best_rob);
        }
        if (best_s < 0) {
            if (stack.empty()) { std::printf("search failed at x=%d\n", death_x); return 1; }
            Decision dd = stack.back(); stack.pop_back();
            tabu.emplace_back(dd.base, dd.s, dd.d);
            C.resize(dd.base); S.resize(dd.base + 1);
            std::printf("  backtrack: undo jump @%zu+%d (d=%d) [stuck at x=%d]\n", dd.base, dd.s, dd.d, death_x);
            std::fflush(stdout);
            continue;
        }
        stack.push_back({L, best_s, best_d});
        env.restore(S[L]);
        auto commit = [&](int a) { bool dn; env.step(a, dn); C.push_back((uint8_t)a); S.emplace_back(); env.snapshot(S.back()); };
        for (int k = 0; k < best_s; ++k) commit(R);
        for (int k = 0; k < best_d; ++k) commit(J);
        std::printf("  jump %2zu: step %4zu hold %2d  (run alone died/stuck at x=%d, now reaches %s%d)  [%.0fs]\n",
                    stack.size(), L + best_s, best_d, death_x, best_score >= 1e9 ? "the flag " : "",
                    (int)(best_score >= 1e9 ? 3161 : best_score), secs_since(t0));
        std::fflush(stdout);
    }
    std::printf("search gave up\n");
    return 1;
}

// Print the wall-probe features along a demo (sanity check of the obs layout).
static int wallprobe(const char* rom, const char* demo) {
    mario::Env env;
    if (!env.init(rom)) return 1;
    std::vector<uint8_t> acts;
    if (!load_actions(demo, acts)) return 1;
    env.reset();
    const int wb = 20 + 2 * mario::Env::TILE_N;
    bool done = false;
    for (size_t t = 0; t < acts.size() && !done; ++t) {
        env.step(acts[t], done);
        const auto& o = env.observation();
        std::string bits;
        for (int k = 0; k < mario::Env::WALL_N; ++k) bits += o[wb + k] > 0.5f ? '#' : '.';
        if (bits.find('#') != std::string::npos)
            std::printf("t=%3zu x=%4d y=%3d wall %s\n", t, env.mario_x(), (int)nes::ram(0x00CE), bits.c_str());
    }
    return 0;
}

// Why does the greedy run stop here? Separates the three usual suspects at one
// state of the greedy run (default: just before its last forward progress):
//   REWARD       some action leads to a clearly better OUTCOME (alive & farther /
//                the flag) but a LOWER return -> the reward prefers the worse play.
//   OBSERVATION  the best action differs from what the net does, and the obs is
//                (near-)identical to another state where the opposite action was
//                taken -> the net cannot tell the two apart.
//   LEARNING     the reward ranks the right action first and the obs is distinct,
//                but the net's Q still prefers another action -> needs more
//                training / the actors have not experienced it.
// "True" action values come from search: after the first action, continue with
// the net, plain run, or run + one running jump (start x hold grid), and keep the
// best discounted return (same reward and gamma as training).
//   mario_gpu diagnose ROM net.bin [step] [demo.bin]
static int diagnose(const char* rom, const char* weights, int step_arg, const char* demo_path) {
    const int S = mario::Env::STATE_DIM, A = mario::N_ACTIONS;
    const int R = mario::A_RUNRIGHT;
#if MARIO_ACTIONS == 3
    const int J = mario::A_RUNJUMP;
#else
    const int J = mario::A_RIGHT_AB;
#endif
    const int H = 120;
    rl::QNet net(S, A, HID);
    if (!net.load(weights) && !net.load_expand(weights)) { std::printf("cannot load %s\n", weights); return 1; }
    mario::Env env;
    if (!env.init(rom)) return 1;
    using Snap = mario::Env::Snapshot;

    // Greedy run, remembering the state before every step.
    std::vector<Snap> snaps;
    std::vector<std::vector<float>> obs;
    std::vector<int> act, xs;
    std::vector<float> s = env.reset();
    bool done = false;
    while (!done) {
        snaps.emplace_back(); env.snapshot(snaps.back());
        obs.push_back(s);
        int a = greedy_action(net, s);
        env.step(a, done);
        act.push_back(a); xs.push_back(env.mario_x());
        s = env.observation();
    }
    int T = (int)act.size();
    bool won = env.won();
    int P = (int)(std::max_element(xs.begin(), xs.end()) - xs.begin());
    std::printf("greedy run: %d steps, max x=%d at step %d, %s\n", T, xs[P], P,
                won ? "WON" : (T >= mario::Env::MAX_STEPS ? "time limit" : "ended (death or stall)"));
    if (won && step_arg < 0) { std::printf("the greedy run clears; pass a step to inspect one anyway\n"); return 0; }
    // Stuck (no progress for many steps after the max) -> look right after it
    // stopped; otherwise (death) look a little before the end.
    bool stalled = !won && T - 1 - P >= 30;
    int t0 = step_arg >= 0 ? std::min(step_arg, T - 1) : stalled ? std::min(T - 1, P + 1) : std::max(0, P - 2);
    const char* an[] = {"run", "runjump", "hop", "a3", "a4", "a5"};
    std::printf("greedy tail (step x action):");
    for (int t = std::max(0, t0 - 8); t <= std::min(T - 1, t0 + 4); ++t)
        std::printf("%s %d:%d:%s", t == t0 ? " [" : " ", t, xs[t], an[act[t]]);
    std::printf("   <- inspecting step %d\n", t0);
    {
        std::string wb;
        for (int k = 20 + 2 * mario::Env::TILE_N; k < S; ++k) wb += obs[t0][k] > 0.5f ? '#' : '.';
        std::printf("state at step %d: x=%d y=%.0f vx=%.2f air=%.2f wall-probe %s\n", t0, t0 > 0 ? xs[t0 - 1] : 0,
                    obs[t0][1] * 255, obs[t0][2], obs[t0][3], wb.c_str());
    }

    // Search the value of each first action.
    struct Out { double G = -1e30; int maxx = 0; bool alive = false, won = false; };
    auto outcome_key = [](const Out& o) { return o.won ? 1e9 : (o.alive ? 1e5 : 0) + o.maxx; };
    auto rollout = [&](int a0, int cont, int js, int jd) {
        env.restore(snaps[t0]);
        Out o; o.G = 0; double g = 1;
        bool dn = false;
        std::vector<float> cur = env.observation();
        for (int k = 0; k < H && !dn; ++k) {
            int a = a0;
            if (k > 0) {
                if (cont == 0) a = greedy_action(net, cur);
                else if (cont == 1) a = R;
                else a = (k - 1 >= js && k - 1 < js + jd) ? J : R;
            }
            float r = env.step(a, dn);
            o.G += g * r; g *= GAMMA;
            o.maxx = std::max(o.maxx, env.mario_x());
            cur = env.observation();
        }
        o.won = env.won(); o.alive = !dn || o.won;
        return o;
    };
    std::vector<Out> bestG(A), bestO(A);   // per first action: best return / best outcome
    for (int a0 = 0; a0 < A; ++a0) {
        auto consider = [&](const Out& o) {
            if (o.G > bestG[a0].G) bestG[a0] = o;
            if (outcome_key(o) > outcome_key(bestO[a0]) || bestO[a0].G < -1e29) bestO[a0] = o;
        };
        consider(rollout(a0, 0, 0, 0));
        consider(rollout(a0, 1, 0, 0));
        for (int js = 0; js <= 18; js += 3)
            for (int jd : {2, 4, 6, 9, 13}) consider(rollout(a0, 2, js, jd));
    }
    std::vector<float> q(A);
    {
        auto x = ag::Tensor::from(obs[t0], {1, S}, false);
        auto qq = net.forward(x).data();
        for (int a = 0; a < A; ++a) q[a] = qq[a] / REWARD_SCALE;
    }
    int ag_ = argmax(q.data(), A);
    std::printf("\n action    net Q(x10)   search: best return (x reached, %s)   best outcome (return)\n", "alive?");
    for (int a = 0; a < A; ++a)
        std::printf(" %-8s %9.1f %s  %9.1f (x=%4d %-5s)             x=%4d %-5s (%.1f)\n", an[a], q[a], a == ag_ ? "*" : " ",
                    bestG[a].G, bestG[a].maxx, bestG[a].won ? "FLAG" : bestG[a].alive ? "alive" : "end",
                    bestO[a].maxx, bestO[a].won ? "FLAG" : bestO[a].alive ? "alive" : "end", bestO[a].G);

    int a_ret = 0, a_out = 0;
    for (int a = 1; a < A; ++a) {
        if (bestG[a].G > bestG[a_ret].G) a_ret = a;
        if (outcome_key(bestO[a]) > outcome_key(bestO[a_out])) a_out = a;
    }
    // REWARD: the best achievable outcome scores a lower return than some worse play.
    double best_ret_overall = bestG[a_ret].G;
    const Out& top = bestO[a_out];
    bool better_outcome = outcome_key(top) > outcome_key(bestG[a_ret]) + 64;
    std::printf("\n");
    if (better_outcome && top.G < best_ret_overall - 1.0) {
        std::printf("VERDICT: REWARD -- '%s' reaches x=%d %s but returns %.1f, while the return-maximizing play "
                    "(%s, x=%d %s) returns %.1f. The reward ranks the worse play higher.\n",
                    an[a_out], top.maxx, top.won ? "(flag)" : top.alive ? "alive" : "", top.G,
                    an[a_ret], bestG[a_ret].maxx, bestG[a_ret].alive ? "alive" : "ended", best_ret_overall);
        return 0;
    }
    if (bestG[ag_].G >= best_ret_overall - 1.0) {
        std::printf("VERDICT: NOT HERE -- the net's action '%s' is already (near) the best under the reward at step %d. "
                    "The mistake is earlier: rerun with a smaller step (e.g. %d).\n", an[ag_], t0, std::max(0, t0 - 4));
        return 0;
    }
    // OBSERVATION: a (near-)identical obs elsewhere was answered with another action.
    std::vector<std::pair<std::vector<float>, int>> ref;
    for (int t = 0; t + 10 < P; ++t) ref.push_back({obs[t], act[t]});
    std::vector<uint8_t> demo;
    if (demo_path && load_actions(demo_path, demo)) {
        Episode de = demo_episode(env, demo);
        for (int t = 0; t < de.T; ++t)
            ref.push_back({std::vector<float>(de.obs.begin() + (size_t)t * S, de.obs.begin() + (size_t)(t + 1) * S), de.act[t]});
    }
    int alias = -1; float alias_d = 1e9f;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i].second == a_ret) continue;
        float d = 0;
        for (int k = 0; k < S; ++k) d = std::max(d, std::fabs(ref[i].first[k] - obs[t0][k]));
        if (d < alias_d) { alias_d = d; alias = (int)i; }
    }
    if (alias >= 0) {
        // Where the nearest state answered differently differs from this one, by
        // obs group -- if only "where am I" features (mario x/y) separate them, the
        // net has to memorize positions: effectively an observation gap.
        const int TN = mario::Env::TILE_N;
        struct G { const char* name; int lo, hi; } groups[] = {
            {"mario x/y/vx/air", 0, 4}, {"enemies", 4, 19}, {"terrain grid", 19, 19 + TN}, {"power", 19 + TN, 20 + TN},
            {"?-blocks", 20 + TN, 20 + 2 * TN}, {"wall probe", 20 + 2 * TN, S}};
        std::printf("nearest state that took another action ('%s', %s): max |diff| by group:",
                    an[ref[alias].second], alias < (P > 10 ? P - 10 : 0) ? "greedy run" : "demo");
        for (auto& g : groups) {
            float d = 0;
            for (int k = g.lo; k < g.hi; ++k) d = std::max(d, std::fabs(ref[alias].first[k] - obs[t0][k]));
            std::printf("  %s %.2f", g.name, d);
        }
        std::printf("\n");
    }
    if (alias >= 0 && alias_d < 0.05f) {
        std::printf("VERDICT: OBSERVATION -- best action here is '%s', but reference state #%d (%s) has an obs within "
                    "%.3f (max abs diff) and took '%s'. The net cannot separate them; add the missing feature.\n",
                    an[a_ret], alias, alias < (int)(P > 10 ? P - 10 : 0) ? "greedy run" : "demo", alias_d,
                    an[ref[alias].second]);
        return 0;
    }
    std::printf("VERDICT: LEARNING/EXPLORATION -- the reward ranks '%s' best (%.1f vs %.1f for the net's '%s') and the obs "
                "is distinct (nearest conflicting state %.3f away), but the net's Q prefers '%s'. Needs more training "
                "or more experience from here (curriculum / actors).\n",
                an[a_ret], best_ret_overall, bestG[ag_].G, an[ag_], alias_d, an[ag_]);
    return 0;
}

static int evalnet(const char* rom, const char* weights) {
    rl::QNet net(mario::Env::STATE_DIM, mario::N_ACTIONS, HID);
    if (!net.load(weights) && !net.load_expand(weights)) { std::printf("cannot load %s\n", weights); return 1; }
    mario::Env env;
    if (!env.init(rom)) return 1;
    bool won = false;
    std::vector<uint8_t> acts;
    int x = greedy_run(net, env, won, &acts);
    std::printf("greedy x=%d won=%d steps=%zu\n", x, (int)won, acts.size());
    return won ? 0 : 2;
}

static int record(const char* rom, const char* weights, const char* out_path) {
    rl::QNet net(mario::Env::STATE_DIM, mario::N_ACTIONS, HID);
    if (!net.load(weights) && !net.load_expand(weights)) { std::printf("cannot load %s\n", weights); return 1; }
    mario::Env env;
    if (!env.init(rom)) return 1;
    std::vector<float> s = env.reset();
    FILE* f = std::fopen(out_path, "wb");
    if (!f) return 1;
    uint32_t w = nes::WIDTH, h = nes::HEIGHT, nframes = 0;
    std::fwrite("MRUN", 1, 4, f);
    std::fwrite(&nframes, 4, 1, f); std::fwrite(&w, 4, 1, f); std::fwrite(&h, 4, 1, f);
    std::vector<uint8_t> rgb(w * h * 3);
    auto put_frame = [&]() {
        const uint32_t* px = nes::pixels();
        for (uint32_t i = 0; i < w * h; ++i) {
            rgb[i * 3 + 0] = (px[i] >> 16) & 0xFF; rgb[i * 3 + 1] = (px[i] >> 8) & 0xFF; rgb[i * 3 + 2] = px[i] & 0xFF;
        }
        std::fwrite(rgb.data(), 1, rgb.size(), f);
        ++nframes;
    };
    bool done = false;
    while (!done) {
        env.step(greedy_action(net, s), done);
        s = env.observation();
        put_frame();
    }
    // The env counts x>3160 as the win, which can fire while Mario is still in
    // the air on his way to the pole. Keep the emulator running hands-off for a
    // few seconds so the video shows the pole grab, and confirm it really
    // happened (float state 3 = sliding down the flagpole).
    bool pole = false;
    if (env.won())
        for (int t = 0; t < 120; ++t) {
            nes::set_buttons(0, 0);
            for (int k = 0; k < mario::Env::FRAME_SKIP; ++k) nes::step_frame();
            pole = pole || nes::ram(0x001D) == 0x03;
            put_frame();
        }
    std::fseek(f, 4, SEEK_SET);
    std::fwrite(&nframes, 4, 1, f);
    std::fclose(f);
    std::printf("recorded %u frames (x=%d won=%d flagpole_grabbed=%d) -> %s\n",
                nframes, env.mario_x(), (int)env.won(), (int)pole, out_path);
    return 0;
}

// CPU and CUDA learners from the same weights on the same batches must agree.
static int parity() {
    const int S = mario::Env::STATE_DIM, A = mario::N_ACTIONS, B = 64;
    auto gpu = rl::make_cuda_learner(S, A, HID);
    if (!gpu) { std::printf("no CUDA learner in this build / no GPU\n"); return 1; }
    auto cpu = rl::make_cpu_learner(S, A, HID);
    ag::seed(7);
    rl::QNet net(S, A, HID);
    std::vector<std::vector<float>> p;
    qnet_get(net, p);
    cpu->set_params(p); gpu->set_params(p);
    rl::Batch b; b.resize(B, S);
    float worst = 0.f;
    for (int it = 0; it < 20; ++it) {
        for (auto& v : b.s) v = ag::randf() < 0.3f ? ag::randf() : 0.f;
        for (auto& v : b.ns) v = ag::randf() < 0.3f ? ag::randf() : 0.f;
        for (int r = 0; r < B; ++r) {
            b.a[r] = (int)(ag::randf() * A) % A; b.ret[r] = ag::randf() * 4 - 2;
            b.gn[r] = ag::randf() < 0.1f ? 0.f : 0.95f; b.demo[r] = ag::randf() < 0.3f;
        }
        float lc = cpu->train(b), lg = gpu->train(b);
        if (it % 5 == 0) cpu->sync_target(), gpu->sync_target();
        std::vector<std::vector<float>> pc, pg;
        cpu->get_params(pc); gpu->get_params(pg);
        float md = 0.f;
        for (size_t i = 0; i < pc.size(); ++i)
            for (size_t j = 0; j < pc[i].size(); ++j) md = std::max(md, std::fabs(pc[i][j] - pg[i][j]));
        worst = std::max(worst, md);
        std::printf("step %2d  loss cpu %.6f gpu %.6f  max|dW| %.2e\n", it, lc, lg, md);
    }
    std::vector<float> qc(B * A), qg(B * A);
    cpu->forward(b.s.data(), B, qc.data()); gpu->forward(b.s.data(), B, qg.data());
    float mq = 0.f;
    for (int i = 0; i < B * A; ++i) mq = std::max(mq, std::fabs(qc[i] - qg[i]));
    std::printf("forward max|dQ| %.2e\n", mq);
    bool ok = worst < 1e-3f && mq < 1e-3f;
    std::printf("parity %s\n", ok ? "OK" : "MISMATCH");
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------- actor
static int actor(const char* rom, int id, int n_actors) {
    std::string dir = RUN_DIR;
    std::string log = dir + "/actor_" + std::to_string(id) + ".log";
    std::freopen(log.c_str(), "w", stdout);
    ag::seed(1000 + id);
    const int S = mario::Env::STATE_DIM, A = mario::N_ACTIONS;
    // Ape-X epsilon spread: actor 0 explores most, the last one is nearly greedy.
    float eps = n_actors > 1 ? std::pow(0.4f, 1.f + 7.f * id / (n_actors - 1)) : 0.1f;
    const float ckpt_prob = 0.5f;
    mario::Env env;
    if (!env.init(rom)) return 1;
    if (env.load_demo((dir + "/demo.bin").c_str())) env.build_curriculum(24);
    rl::QNet net(S, A, HID);
    std::printf("actor %d/%d eps=%.4f checkpoints=%d\n", id, n_actors, eps, env.num_checkpoints());
    std::fflush(stdout);

    std::string wpath = dir + "/weights.bin", vpath = dir + "/weights.ver";
    int have_ver = -1, seq = 0;
    auto t_last = Clock::now();
    for (;;) {
        if (fs::exists(dir + "/stop")) break;
        // Pick up the newest published weights (the learner bumps weights.ver).
        int ver = -1;
        if (FILE* f = std::fopen(vpath.c_str(), "r")) { if (std::fscanf(f, "%d", &ver) != 1) ver = -1; std::fclose(f); }
        if (ver != have_ver && ver >= 0) {
            if (net.load(wpath)) { have_ver = ver; t_last = Clock::now(); }
        } else if (secs_since(t_last) > 300) {
            std::printf("no new weights for 300s -> learner gone, exiting\n");
            break;
        }
        if (have_ver < 0) { std::this_thread::sleep_for(std::chrono::milliseconds(200)); continue; }

        Episode e; e.S = S; e.actor = id;
        std::vector<float> s;
        if (env.num_checkpoints() > 0 && ag::randf() < ckpt_prob) {
            e.start_ckpt = (int)(ag::randf() * env.num_checkpoints()) % env.num_checkpoints();
            s = env.reset_to_checkpoint(e.start_ckpt);
        } else {
            s = env.reset();
        }
        e.obs.insert(e.obs.end(), s.begin(), s.end());
        bool done = false;
        while (!done) {
            int a = ag::randf() < eps ? (int)(ag::randf() * A) % A : greedy_action(net, s);
            float r = env.step(a, done);
            s = env.observation();
            e.obs.insert(e.obs.end(), s.begin(), s.end());
            e.act.push_back(a); e.rew.push_back(r);
            e.max_x = std::max(e.max_x, env.mario_x());
        }
        e.T = (int)e.act.size(); e.won = env.won();
        write_episode(dir + "/ep/a" + std::to_string(id) + "_" + std::to_string(seq++) + ".ep", e);
    }
    return 0;
}

// ---------------------------------------------------------------- learner
static void publish(rl::QNet& net, int ver) {
    std::string dir = RUN_DIR;
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

static int train(const char* exe, const char* rom, const char* demo_path, int n_actors, double minutes, const char* warm) {
    const int S = mario::Env::STATE_DIM, A = mario::N_ACTIONS;
    const int B = 256, target_sync = 2000, publish_every = 200, pretrain_steps = 30000;
    const float demo_frac = 0.25f;
    double eval_period = 30.0;
    if (const char* e = std::getenv("MARIO_EVAL_SEC")) eval_period = std::atof(e);
    float ratio = 16.f;
    if (const char* e = std::getenv("MARIO_RATIO")) ratio = (float)std::atof(e);

    std::string dir = RUN_DIR;
    fs::create_directories(dir + "/ep");
    fs::remove(dir + "/stop");
    for (auto& f : fs::directory_iterator(dir + "/ep")) fs::remove(f.path());

    std::unique_ptr<rl::Learner> L;
    const char* want = std::getenv("MARIO_LEARNER");
    if (!(want && std::strcmp(want, "cpu") == 0)) L = rl::make_cuda_learner(S, A, HID);
    if (!L) L = rl::make_cpu_learner(S, A, HID);
    L->lr = 1e-4f;
    if (const char* e = std::getenv("MARIO_LR")) L->lr = (float)std::atof(e);

    ag::seed(1);
    rl::QNet net(S, A, HID);
    bool warmed = false;
    if (warm && *warm) {
        if ((warmed = net.load(warm) || net.load_expand(warm))) std::printf("   warm start: %s (demo pre-training skipped)\n", warm);
        else std::printf("   (could not load %s, starting fresh)\n", warm);
    }
    std::vector<std::vector<float>> P;
    qnet_get(net, P);
    L->set_params(P);

    mario::Env env;
    if (!env.init(rom)) return 1;
    std::vector<uint8_t> demo_acts;
    if (!load_actions(demo_path, demo_acts)) { std::printf("cannot read demo %s\n", demo_path); return 1; }
    Episode de = demo_episode(env, demo_acts);
    std::printf("== mario_gpu: DQfD, learner=%s, %d actors, %.0f min ==\n", L->name(), n_actors, minutes);
    std::printf("   state=%d actions=%d hidden=%d batch=%d n_step=%d ratio=%.1f | demo %s: %d steps, x=%d won=%d\n",
                S, A, HID, B, N_STEP, ratio, demo_path, de.T, de.max_x, de.won);
    std::fflush(stdout);
    save_actions(dir + "/demo.bin", demo_acts);   // actors build their curriculum from it

    NStepReplay demo_buf(S, de.T + 1), rep(S, 400000);
    demo_buf.add_episode(de);
    rl::Batch batch; batch.resize(B, S);
    auto sample = [&](int n_demo) {
        for (int r = 0; r < B; ++r) {
            if (r < n_demo || rep.size == 0)
                demo_buf.fill_row((size_t)(ag::randf() * demo_buf.size) % demo_buf.size, batch, r, 1.f);
            else
                rep.fill_row((size_t)(ag::randf() * rep.size) % rep.size, batch, r, 0.f);
        }
    };

    auto t0 = Clock::now();
    int best_x = -1, ver = 0;
    bool cleared = false;
    auto evaluate = [&](const char* tag) {
        L->get_params(P); qnet_set(net, P);
        bool won = false;
        std::vector<uint8_t> acts;
        int x = greedy_run(net, env, won, &acts);
        if (x > best_x) { best_x = x; net.save("mario_gpu_best.bin"); }
        if (won && !cleared) {
            cleared = true;
            net.save("mario_gpu_clear.bin");
            save_actions(dir + "/greedy_clear.bin", acts);
            std::printf("*** GREEDY CLEARED 1-1 from the start (%zu steps, %.1f min) -> mario_gpu_clear.bin ***\n",
                        acts.size(), secs_since(t0) / 60);
        }
        std::printf("[%7.1fs] %-8s greedy_x %4d %s best %d\n", secs_since(t0), tag, x, won ? "WON" : "   ", best_x);
        std::fflush(stdout);
        return won;
    };

    // Demo-state accuracy (greedy action == demo action) and label conflicts
    // (identical observations the demo answered differently -- unfittable).
    auto demo_acc = [&]() {
        std::vector<float> q((size_t)de.T * A);
        L->forward(de.obs.data(), de.T, q.data());
        int ok = 0, first_bad = -1;
        for (int t = 0; t < de.T; ++t) {
            if (argmax(&q[(size_t)t * A], A) == de.act[t]) ++ok;
            else if (first_bad < 0) first_bad = t;
        }
        return std::make_pair(ok, first_bad);
    };
    {
        int conflicts = 0;
        for (int i = 0; i < de.T; ++i)
            for (int j = i + 1; j < de.T; ++j)
                if (de.act[i] != de.act[j] &&
                    std::equal(&de.obs[(size_t)i * S], &de.obs[(size_t)(i + 1) * S], &de.obs[(size_t)j * S]))
                    ++conflicts;
        std::printf("   demo label conflicts (same obs, different action): %d\n", conflicts);
    }

    // DQfD pre-training on the demo alone until the net reproduces every demo
    // action (then the greedy run from the start IS the demo, deterministically).
    // Skipped on a warm start: its high LR would overwrite the loaded policy.
    const float lr_rl = L->lr;
    L->lr = 5e-4f;
    for (int i = 1; i <= (warmed ? 0 : pretrain_steps); ++i) {
        sample(B);
        float l = L->train(batch);
        if (i % target_sync == 0) L->sync_target();
        if (i % 500 == 0) {
            auto [ok, bad] = demo_acc();
            std::printf("   pretrain %5d loss %.4f demo acc %d/%d first miss @%d\n", i, l, ok, de.T, bad);
            std::fflush(stdout);
            if (ok == de.T && i >= 2000) break;
        }
    }
    L->lr = lr_rl;
    L->sync_target();
    evaluate("pretrain");

    L->get_params(P); qnet_set(net, P);
    publish(net, ver++);
    for (int i = 0; i < n_actors; ++i)
        spawn_self(exe, {"actor", rom, std::to_string(i), std::to_string(n_actors)});

    long updates = 0, steps_in = 0, eps_in = 0;
    std::deque<int> recent_x;
    int actor_wins = 0;
    double loss_acc = 0; int loss_n = 0;
    auto t_eval = Clock::now(), t_log = Clock::now();
    while (secs_since(t0) < minutes * 60 && !cleared) {
        // Ingest finished episodes.
        std::vector<fs::path> files;
        for (auto& f : fs::directory_iterator(dir + "/ep"))
            if (f.path().extension() == ".ep") files.push_back(f.path());
        for (auto& f : files) {
            Episode e;
            if (read_episode(f.string(), e) && e.S == S) {
                rep.add_episode(e);
                steps_in += e.T; ++eps_in;
                actor_wins += e.won;
                recent_x.push_back(e.max_x);
                if (recent_x.size() > 100) recent_x.pop_front();
            }
            std::error_code ec;
            fs::remove(f, ec);
        }
        // Train while within the replay ratio (samples drawn per env step ingested).
        int did = 0;
        while (rep.size >= 2000 && (double)updates * B < ratio * steps_in && did < 500) {
            sample((int)(B * demo_frac));
            loss_acc += L->train(batch); ++loss_n;
            ++updates; ++did;
            if (updates % target_sync == 0) L->sync_target();
            if (updates % publish_every == 0) { L->get_params(P); qnet_set(net, P); publish(net, ver++); }
        }
        if (!did) std::this_thread::sleep_for(std::chrono::milliseconds(50));

        if (secs_since(t_log) > 10) {
            double ax = 0; for (int x : recent_x) ax += x; if (!recent_x.empty()) ax /= recent_x.size();
            double el = secs_since(t0);
            std::printf("[%7.1fs] steps %ld (%.0f/s) eps %ld actor_wins %d avg_x %.0f | updates %ld (%.1f/s) loss %.4f replay %zu\n",
                        el, steps_in, steps_in / el, eps_in, actor_wins, ax, updates, updates / el,
                        loss_n ? loss_acc / loss_n : 0.0, rep.size);
            std::fflush(stdout);
            loss_acc = 0; loss_n = 0; t_log = Clock::now();
        }
        if (secs_since(t_eval) > eval_period && updates > 0) { evaluate("eval"); t_eval = Clock::now(); }
    }
    if (FILE* f = std::fopen((dir + "/stop").c_str(), "w")) std::fclose(f);
    std::printf("done: best greedy x=%d cleared=%d (%.1f min, %ld env steps, %ld updates)\n",
                best_x, (int)cleared, secs_since(t0) / 60, steps_in, updates);
    return cleared ? 0 : 2;
}

int main(int argc, char** argv) {
    auto arg = [&](int i, const char* d) { return argc > i ? argv[i] : d; };
    std::string mode = arg(1, "");
    const char* rom_default = "Super Mario Bros (JU) (PRG 0).nes";
    if (mode == "bench") return bench(arg(2, rom_default));
    if (mode == "cleandemo" && argc >= 5) return cleandemo(argv[2], argv[3], argv[4]);
    if (mode == "diagnose" && argc >= 4)
        return diagnose(argv[2], argv[3], std::atoi(arg(4, "-1")), arg(5, "warmstarts/demo_robust_1-1_3act.bin"));
    if (mode == "wallprobe" && argc >= 4) return wallprobe(argv[2], argv[3]);
    if (mode == "searchdemo" && argc >= 4) return searchdemo(argv[2], argv[3]);
    if (mode == "eval" && argc >= 4) return evalnet(argv[2], argv[3]);
    if (mode == "record" && argc >= 5) return record(argv[2], argv[3], argv[4]);
    if (mode == "parity") return parity();
    if (mode == "actor" && argc >= 5) return actor(argv[2], std::atoi(argv[3]), std::atoi(argv[4]));
    if (mode == "train" && argc >= 4)
        return train(argv[0], argv[2], argv[3], std::atoi(arg(4, "3")), std::atof(arg(5, "60")), arg(6, ""));
    std::printf("usage: mario_gpu cleandemo|train|eval|record|parity|bench ... (see header)\n");
    return 1;
}
