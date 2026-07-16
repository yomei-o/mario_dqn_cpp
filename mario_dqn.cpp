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
#include <cstdlib>
#include <string>
#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>

#ifdef _WIN32
// Raise the system timer resolution to 1ms so std::this_thread::sleep_for is
// accurate (Windows' default ~15.6ms granularity otherwise makes short sleeps
// overshoot, over-throttling the trainer). Declared here to avoid pulling in
// the whole <windows.h>; winmm is linked in CMakeLists.
extern "C" __declspec(dllimport) unsigned int __stdcall timeBeginPeriod(unsigned int);
extern "C" __declspec(dllimport) unsigned int __stdcall timeEndPeriod(unsigned int);
#endif

using namespace ag;
using namespace rl;

// CPU throttle: keep long training from pinning a core at 100% so the machine
// stays usable. Targets a fraction of wall-clock as busy time (default 0.8 =>
// ~80% CPU); override with env var MARIO_CPU (e.g. 0.5, or 1.0 to disable).
// Self-calibrating via CUMULATIVE busy/slept accounting: after each ~100ms chunk
// of work it sleeps the running deficit between the target sleep total and what
// it has actually slept. Because it measures ACTUAL sleep (Windows' ~15ms timer
// granularity makes short sleeps overshoot) and corrects against the cumulative
// target, the average busy fraction converges to `util` regardless of machine
// speed or OS sleep rounding.
struct CpuThrottle {
    double util = 0.8;
    bool on = true;
    std::chrono::steady_clock::time_point mark;   // start of the current busy accrual
    double busy_total = 0.0, slept_total = 0.0;   // cumulative seconds
    CpuThrottle() {
        if (const char* e = std::getenv("MARIO_CPU")) util = std::atof(e);
        on = util > 0.0 && util < 1.0;
#ifdef _WIN32
        if (on) timeBeginPeriod(1);   // accurate short sleeps (restored in dtor)
#endif
        mark = std::chrono::steady_clock::now();
    }
#ifdef _WIN32
    ~CpuThrottle() { if (on) timeEndPeriod(1); }
#endif
    // Call once per env step. No-op until ~100ms of busy time has accrued, so the
    // resulting sleep is comfortably larger than the OS timer granularity.
    void maybe_sleep() {
        if (!on) return;
        auto now = std::chrono::steady_clock::now();
        double chunk = std::chrono::duration<double>(now - mark).count();
        if (chunk < 0.1) return;                       // keep accruing (don't reset mark)
        busy_total += chunk;
        double target_slept = busy_total * (1.0 - util) / util;
        double need = target_slept - slept_total;      // cumulative deficit (self-correcting)
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

// One fully-greedy rollout from the level start (honest skill metric). Returns
// the max level-x reached; if out_score is given, also reports the final score.
static int greedy_episode(QNet& net, mario::Env& env, int* out_score = nullptr, int* out_power = nullptr) {
    std::vector<float> s = env.reset();
    bool done = false; int mx = 0, mp = 0;
    while (!done) {
        int a = greedy_action(net, s);
        env.step(a, done);
        s = env.observation();
        mx = std::max(mx, env.mario_x());
        mp = std::max(mp, env.power());        // highest power reached (1=grabbed a mushroom, 2=fire)
    }
    if (out_score) *out_score = env.score();
    if (out_power) *out_power = mp;
    return mx;
}

static void clip_grads(std::vector<Tensor>& ps, float max_norm) {
    double sq = 0;
    for (auto& p : ps) for (float g : p.grad()) sq += (double)g * g;
    float norm = (float)std::sqrt(sq);
    if (norm > max_norm) { float s = max_norm / (norm + 1e-6f); for (auto& p : ps) for (float& g : p.grad()) g *= s; }
}

// Action-sequence file I/O (u32 count + count action bytes). Shared by the demo
// tooling and the win-capture path.
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

// Replay a saved FROM-BOOT action sequence (e.g. a captured winning run) and
// record it to web/run.bin for the browser viewer. env.reset() + stepping the
// sequence reproduces the exact trajectory (deterministic emulator), so a demo
// that reached the flagpole plays back the full level clear. Same MRUN format
// as record(): "MRUN", u32 nframes, u32 w, u32 h, then nframes*w*h*3 RGB bytes.
static int recorddemo(const char* rom, const char* actions_path, const char* out_path) {
    mario::Env env;
    if (!env.init(rom)) return 1;
    std::vector<uint8_t> acts;
    if (!load_actions(actions_path, acts)) { std::printf("cannot load actions %s\n", actions_path); return 1; }
    env.reset();
    FILE* f = std::fopen(out_path, "wb");
    if (!f) { std::printf("cannot open %s\n", out_path); return 1; }
    uint32_t w = nes::WIDTH, h = nes::HEIGHT, nframes = 0;
    std::fwrite("MRUN", 1, 4, f);
    std::fwrite(&nframes, 4, 1, f); std::fwrite(&w, 4, 1, f); std::fwrite(&h, 4, 1, f);
    std::vector<uint8_t> rgb(w * h * 3);
    bool done = false;
    for (size_t t = 0; t < acts.size() && !done; ++t) {
        env.step(acts[t], done);
        const uint32_t* px = nes::pixels();
        for (uint32_t i = 0; i < w * h; ++i) {
            rgb[i * 3 + 0] = (px[i] >> 16) & 0xFF;
            rgb[i * 3 + 1] = (px[i] >> 8) & 0xFF;
            rgb[i * 3 + 2] = px[i] & 0xFF;
        }
        std::fwrite(rgb.data(), 1, rgb.size(), f);
        ++nframes;
    }
    std::fseek(f, 4, SEEK_SET);
    std::fwrite(&nframes, 4, 1, f);
    std::fclose(f);
    std::printf("recorded %u frames from demo (final x=%d, won=%d) -> %s\n",
                nframes, env.mario_x(), (int)env.won(), out_path);
    return 0;
}

// Record a greedy episode of the trained agent to web/run.bin so the browser
// viewer can play it back. Format: "MRUN", u32 nframes, u32 w, u32 h, then
// nframes * w*h*3 bytes RGB (one NES frame per agent step).
static int record(const char* rom, const char* weights, const char* out_path) {
    const int S = mario::Env::STATE_DIM, A = mario::N_ACTIONS;
    QNet net(S, A, 256);
    if (!net.load_expand(weights)) std::printf("(warning) could not load %s; using random net\n", weights);
    mario::Env env;
    if (!env.init(rom)) return 1;
    env.reset();

    FILE* f = std::fopen(out_path, "wb");
    if (!f) { std::printf("cannot open %s\n", out_path); return 1; }
    uint32_t w = nes::WIDTH, h = nes::HEIGHT, nframes = 0;
    std::fwrite("MRUN", 1, 4, f);
    std::fwrite(&nframes, 4, 1, f); std::fwrite(&w, 4, 1, f); std::fwrite(&h, 4, 1, f);

    std::vector<uint8_t> rgb(w * h * 3);
    bool done = false;
    for (int t = 0; t < 1200 && !done; ++t) {
        int a = greedy_action(net, env.observation());
        env.step(a, done);
        const uint32_t* px = nes::pixels();
        for (uint32_t i = 0; i < w * h; ++i) {
            rgb[i * 3 + 0] = (px[i] >> 16) & 0xFF;
            rgb[i * 3 + 1] = (px[i] >> 8) & 0xFF;
            rgb[i * 3 + 2] = px[i] & 0xFF;
        }
        std::fwrite(rgb.data(), 1, rgb.size(), f);
        ++nframes;
    }
    std::fseek(f, 4, SEEK_SET);           // patch nframes
    std::fwrite(&nframes, 4, 1, f);
    std::fclose(f);
    std::printf("recorded %u frames (final x=%d) -> %s\n", nframes, env.mario_x(), out_path);
    return 0;
}

// Diagnostic: verify SMB score/coins/power-state RAM by playing greedily and
// printing those values whenever they change (stomps/coins/power-ups add score).
static int scoretest(const char* rom, const char* weights) {
    auto score_of = []() {
        int s = 0;
        for (int i = 0; i < 6; ++i) s = s * 10 + nes::ram(0x07DD + i);   // 6 BCD digits
        return s;
    };
    const int S = mario::Env::STATE_DIM, A = mario::N_ACTIONS;
    QNet net(S, A, 256);
    bool have = net.load_expand(weights);
    mario::Env env;
    if (!env.init(rom)) return 1;
    std::vector<float> s = env.reset();
    int last_score = score_of(), last_coins = nes::ram(0x075E), last_pow = nes::ram(0x0756);
    std::printf("start: score=%d coins=%d power=%d (0=small 1=super 2=fire)\n", last_score, last_coins, last_pow);
    bool done = false;
    for (int t = 0; t < 600 && !done; ++t) {
        int a = have ? greedy_action(net, s) : mario::A_RIGHT_B;
        env.step(a, done);
        s = env.observation();
        int sc = score_of(), co = nes::ram(0x075E), pw = nes::ram(0x0756);
        if (sc != last_score || co != last_coins || pw != last_pow) {
            std::printf("  x=%4d  score=%d(+%d)  coins=%d  power=%d\n",
                        env.mario_x(), sc, sc - last_score, co, pw);
            last_score = sc; last_coins = co; last_pow = pw;
        }
    }
    std::printf("end: x=%d score=%d coins=%d power=%d\n", env.mario_x(), last_score, last_coins, last_pow);
    return 0;
}

// Diagnostic: confirm the emulator is deterministic across resets IN-PROCESS
// (load-bearing for curriculum-by-input-replay). Run the same fixed action
// sequence 3x in one process; all runs must land on the same x.
static int dettest(const char* rom) {
    mario::Env env;
    if (!env.init(rom)) return 1;
    for (int trial = 0; trial < 3; ++trial) {
        env.reset();
        bool done = false;
        for (int t = 0; t < 250 && !done; ++t) {
            int a = (t * 7 + 3) % mario::N_ACTIONS;   // fixed pseudo-pattern, no RNG
            env.step(a, done);
        }
        std::printf("trial %d: final x=%d\n", trial, env.mario_x());
    }
    return 0;
}

// Diagnostic: verify an in-memory snapshot reproduces the ground-truth replay.
// Restore a checkpoint and replay-to-the-same-point must (a) start at the same
// x and (b) evolve identically under a fixed action sequence. If they match, the
// snapshot captured all state the curriculum relies on.
static int snaptest(const char* rom) {
    mario::Env env;
    if (!env.init(rom)) return 1;
    if (!env.load_demo("demo.bin")) { std::printf("no demo.bin (run gendemo first)\n"); return 1; }
    env.build_curriculum(8);
    int nck = env.num_checkpoints();
    if (nck == 0) { std::printf("no checkpoints built\n"); return 1; }
    int k = nck - 1;
    int idx = env.demo_len() - 8;   // matches build_curriculum's last target (hi)
    auto rollout = [&](bool snapshot) {
        int x0;
        if (snapshot) { env.reset_to_checkpoint(k); }
        else          { env.reset_from(idx); }
        x0 = env.mario_x();
        bool done = false; int last = x0;
        for (int t = 0; t < 200 && !done; ++t) {
            int a = (t * 5 + 2) % mario::N_ACTIONS;   // fixed pattern
            env.step(a, done);
            last = env.mario_x();
        }
        return std::pair<int,int>(x0, last);
    };
    auto gt = rollout(false);   // ground truth: boot + replay
    auto sn = rollout(true);    // snapshot restore
    std::printf("checkpoint x: replay=%d snapshot=%d  %s\n", gt.first, sn.first,
                gt.first == sn.first ? "OK" : "MISMATCH");
    std::printf("after 200 fixed steps: replay x=%d snapshot x=%d  %s\n", gt.second, sn.second,
                gt.second == sn.second ? "OK (snapshot is complete)" : "MISMATCH (snapshot incomplete!)");
    return (gt == sn) ? 0 : 2;
}

// Diagnostic: dump SMB's background tile buffer + Mario's position so we can
// verify the tile-window geometry (which rows are ground, where Mario sits).
static int tilescan(const char* rom) {
    mario::Env env;
    if (!env.init(rom)) return 1;
    env.reset();
    bool done;
    for (int t = 0; t < 20; ++t) env.step(mario::A_RIGHT_B, done);  // walk in a bit
    int lx = nes::ram(0x006D) * 256 + nes::ram(0x0086);
    int px = nes::ram(0x0086), py = nes::ram(0x00CE);
    std::printf("level_x=%d  px(0x86)=%d  py(0xCE)=%d\n", lx, px, py);
    std::printf("player row=(py-32)/16=%d  player col=(lx/16)%%16=%d  page=%d\n",
                (py - 32) / 16, (lx / 16) % 16, (lx / 16 / 16) % 2);
    for (int row = 0; row < 13; ++row) {
        std::printf("row%2d: ", row);
        for (int c = 0; c < 8; ++c) {
            int col = lx / 16 + c;
            int page = (col / 16) % 2, colp = col % 16;
            int v = nes::ram(0x0500 + page * 0xD0 + row * 16 + colp);
            std::printf("%02X ", v);
        }
        std::printf("\n");
    }
    return 0;
}

// Generate a demonstration: run the trained net greedily from the level start
// and save the action sequence to `out` (u32 count + count action bytes). This
// demo is replayed by the curriculum to fast-forward episodes to the frontier.
static int gendemo(const char* rom, const char* weights, const char* out) {
    const int S = mario::Env::STATE_DIM, A = mario::N_ACTIONS;
    QNet net(S, A, 256);
    if (!net.load_expand(weights)) { std::printf("could not load %s\n", weights); return 1; }
    mario::Env env;
    if (!env.init(rom)) return 1;
    std::vector<float> s = env.reset();
    std::vector<uint8_t> actions;
    bool done = false;
    for (int t = 0; t < 2000 && !done; ++t) {
        int a = greedy_action(net, s);
        env.step(a, done);
        s = env.observation();
        actions.push_back((uint8_t)a);
    }
    FILE* f = std::fopen(out, "wb");
    if (!f) { std::printf("cannot open %s\n", out); return 1; }
    uint32_t n = (uint32_t)actions.size();
    std::fwrite(&n, 4, 1, f);
    std::fwrite(actions.data(), 1, n, f);
    std::fclose(f);
    std::printf("demo: %u actions, final x=%d -> %s\n", n, env.mario_x(), out);
    return 0;
}

// Evaluate a checkpoint: greedy rollout from the start, print distance + score.
// Used to pick the winner among parallel best-of-N workers' checkpoints.
static int evalnet(const char* rom, const char* weights) {
    const int S = mario::Env::STATE_DIM, A = mario::N_ACTIONS;
    QNet net(S, A, 256);
    if (!net.load_expand(weights)) { std::printf("%s: load failed\n", weights); return 1; }
    mario::Env env;
    if (!env.init(rom)) return 1;
    int sc = 0;
    int x = greedy_episode(net, env, &sc);
    std::printf("%s: greedy x=%d score=%d (metric=%d)\n", weights, x, sc, x + sc);
    return 0;
}

static int envtest(const char* rom) {
    mario::Env env;
    if (!env.init(rom)) return 1;
    env.reset();
    std::printf("after reset: x=%d\n", env.mario_x());
    for (int t = 0; t < 40; ++t) {
        bool done; float r = env.step(mario::A_RIGHT_B, done);   // run right
        const auto& o = env.observation();
        // obs[4..18] are the 5 enemy slots (active, rel_x, rel_y). Print any active one.
        std::string en;
        for (int i = 0; i < 5; ++i)
            if (o[4 + i * 3] > 0.5f)
                en += " enemy" + std::to_string(i) + "(dx=" + std::to_string(o[4+i*3+1]).substr(0,5) +
                      ",dy=" + std::to_string(o[4+i*3+2]).substr(0,5) + ")";
        // Terrain window (obs[19..]): TILE_ROWS chars per column ahead, '#'=solid '.'=open.
        std::string tiles = " tiles=";
        int base = 19;
        for (int c = 0; c < mario::Env::TILE_COLS; ++c) {
            for (int rr = 0; rr < mario::Env::TILE_ROWS; ++rr)
                tiles += o[base + c * mario::Env::TILE_ROWS + rr] > 0.5f ? '#' : '.';
            tiles += '/';
        }
        std::printf("  step %2d: x=%d r=%.1f done=%d%s%s\n", t, env.mario_x(), r, done,
                    en.empty() ? "" : en.c_str(), tiles.c_str());
        if (done) { std::printf("  episode ended at step %d (x=%d)\n", t, env.mario_x()); break; }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "envtest") == 0)
        return envtest(argc > 2 ? argv[2] : "Super Mario Bros (JU) (PRG 0).nes");
    if (argc > 1 && std::strcmp(argv[1], "tilescan") == 0)
        return tilescan(argc > 2 ? argv[2] : "Super Mario Bros (JU) (PRG 0).nes");
    if (argc > 1 && std::strcmp(argv[1], "dettest") == 0)
        return dettest(argc > 2 ? argv[2] : "Super Mario Bros (JU) (PRG 0).nes");
    if (argc > 1 && std::strcmp(argv[1], "snaptest") == 0)
        return snaptest(argc > 2 ? argv[2] : "Super Mario Bros (JU) (PRG 0).nes");
    if (argc > 1 && std::strcmp(argv[1], "scoretest") == 0)
        return scoretest(argc > 2 ? argv[2] : "Super Mario Bros (JU) (PRG 0).nes",
                         argc > 3 ? argv[3] : "mario_best.bin");
    if (argc > 1 && std::strcmp(argv[1], "gendemo") == 0)
        return gendemo(argc > 2 ? argv[2] : "Super Mario Bros (JU) (PRG 0).nes",
                       argc > 3 ? argv[3] : "mario_best.bin",
                       argc > 4 ? argv[4] : "demo.bin");
    if (argc > 1 && std::strcmp(argv[1], "record") == 0)
        return record(argc > 2 ? argv[2] : "Super Mario Bros (JU) (PRG 0).nes",
                      argc > 3 ? argv[3] : "mario_best.bin",
                      argc > 4 ? argv[4] : "web/run.bin");
    if (argc > 1 && std::strcmp(argv[1], "recorddemo") == 0)   // replay a saved action seq (e.g. a captured win)
        return recorddemo(argc > 2 ? argv[2] : "Super Mario Bros (JU) (PRG 0).nes",
                          argc > 3 ? argv[3] : "demo_win_0.bin",
                          argc > 4 ? argv[4] : "web/run.bin");
    if (argc > 1 && std::strcmp(argv[1], "eval") == 0)
        return evalnet(argc > 2 ? argv[2] : "Super Mario Bros (JU) (PRG 0).nes",
                       argc > 3 ? argv[3] : "mario_best.bin");

    // Training. Usage: mario_dqn [ROM] [seed] [out.bin]
    // Parallelism: launch N processes with distinct seeds and out paths (each
    // gets its own emulator on its own core -- the core is a global singleton, so
    // one process = one core), then pick the best checkpoint with `eval`.
    // Usage: mario_dqn [ROM] [seed] [out.bin] [eps_start] [lr] [curriculum_prob]
    // The optional hyperparameters let parallel workers span the preserve<->explore
    // spectrum (gentle low-eps/low-lr keeps the warm-started policy; aggressive
    // high-eps explores past the frontier); best-of-N keeps whichever wins.
    const char* rom = argc > 1 ? argv[1] : "Super Mario Bros (JU) (PRG 0).nes";
    int seed_val = argc > 2 ? std::atoi(argv[2]) : 0;
    std::string out_path = argc > 3 ? argv[3] : "mario_best.bin";
    float arg_eps = argc > 4 ? (float)std::atof(argv[4]) : -1.f;
    float arg_lr  = argc > 5 ? (float)std::atof(argv[5]) : -1.f;
    float arg_cp  = argc > 6 ? (float)std::atof(argv[6]) : -1.f;

    seed(seed_val);
    const int S = mario::Env::STATE_DIM, A = mario::N_ACTIONS;
    const float gamma = 0.99f;
    const int batch = 32, warmup = 5000, target_sync = 2000, episodes = 12000;
    const int train_freq = 4;   // gradient step every N env steps (Atari-DQN style; ~Nx faster wall-clock)
    const float eps_end = 0.1f;

    QNet online(S, A, 256), target(S, A, 256), best(S, A, 256);
    Adam opt(online.params(), 2.5e-4f);
    auto params = online.params();
    Replay replay(100000);
    mario::Env env;
    if (!env.init(rom)) return 1;

    // Warm start from this worker's own checkpoint if present, else the shared
    // base. A loaded policy is GOOD but its Q-values are calibrated to the old
    // reward scale; with a new reward + high LR + high exploration the first
    // updates destroyed it (greedy-from-start collapsed 2017 -> ~400). So fine
    // -tune GENTLY: low LR + minimal exploration to adapt values without wrecking
    // the policy. Fresh (no checkpoint) training keeps the aggressive schedule.
    bool warm = online.load_expand(out_path.c_str()) || online.load_expand("mario_best.bin");   // expands 89->90 dims
    float eps_start = arg_eps >= 0.f ? arg_eps : (warm ? 0.1f : 1.0f);
    float eps_decay_steps = warm ? 100000.f : 250000.f;
    opt.lr = arg_lr > 0.f ? arg_lr : (warm ? 1.0e-4f : 2.5e-4f);
    target.copy_from(online); best.copy_from(online);

    // Curriculum: if a demonstration exists, precompute snapshots across the back
    // half of it, then start most episodes at one of those checkpoints (near the
    // frontier) so the agent densely practices the still-unsolved final stretch
    // instead of replaying the easy start each time. Snapshots restore instantly.
    bool have_demo = env.load_demo("demo.bin");
    if (have_demo) env.build_curriculum(24);   // finer, whole-level coverage (was 16, back-half only)
    int n_ckpt = env.num_checkpoints();
    bool curriculum = n_ckpt > 0;
    // Anneal the curriculum: start balanced (learn the hard frontier) and shift
    // toward full-start runs so the whole level gets re-consolidated into one
    // chainable policy (heavy curriculum alone made it forget the early game).
    float cp_start = arg_cp >= 0.f ? arg_cp : 0.5f;
    const float cp_end = 0.2f, cp_anneal = 150000.f;

    std::printf("== DQN x Super Mario Bros 1-1 (RAM features, real NES) ==\n");
    std::printf("   seed=%d out=%s | eps_start=%.2f lr=%.1e cp_start=%.2f\n",
                seed_val, out_path.c_str(), eps_start, opt.lr, cp_start);
    std::printf("   state_dim=%d actions=%d | batch=%d target_sync=%d | warm=%d curriculum=%d(demo=%d ckpts=%d)\n",
                S, A, batch, target_sync, (int)warm, (int)curriculum, env.demo_len(), n_ckpt);
    if (curriculum)
        std::printf("   checkpoint x-range: %d .. %d\n",
                    env.checkpoint_x(0), env.checkpoint_x(n_ckpt - 1));

    CpuThrottle throttle;   // ~80% CPU by default (env MARIO_CPU to override)
    std::printf("   cpu throttle: %s (target util=%.2f, env MARIO_CPU)\n",
                throttle.on ? "on" : "off", throttle.util);
    std::fflush(stdout);    // surface config immediately when logging to a file

    long total_steps = 0;
    std::deque<int> recent_x;   // last-50 episode max distances
    // Checkpoint by a distance+score composite so that, among policies reaching a
    // similar distance, the higher-scoring one (more coins/power-ups) is kept --
    // while distance still dominates, keeping the flag as the primary objective.
    int best_x = 0, best_score = 0, best_metric = -1;
    if (warm) { best_x = greedy_episode(online, env, &best_score); best_metric = best_x + best_score; }
    std::printf("   warm-start greedy x=%d score=%d\n", best_x, best_score);

    // Win-capture: when a training episode reaches the flagpole, save the FROM-BOOT
    // action sequence that produced it (demo prefix up to its start checkpoint +
    // this episode's actions). That sequence is a full level-clear demo -- playable
    // with `recorddemo`, and promotable to demo.bin to push the curriculum frontier
    // to the flag. Keep only the most autonomous win (smallest replayed prefix), so
    // the captured demo trends toward a from-start clear as the agent improves.
    const std::string win_path = "demo_win_" + std::to_string(seed_val) + ".bin";
    int best_win_prefix = 1 << 30;

    // Restrict curriculum starts to EARLY checkpoints (env MARIO_CKPT_XMAX). Phase
    // ii forces dense practice at the early ? -block / mushroom cluster (x~700) so
    // the agent actually attempts block-hits (a rare precise maneuver it otherwise
    // never discovers) -- the reward for a coin/mushroom can only be learned once
    // it's been hit. Unset = all checkpoints (phase-i whole-level behavior).
    int ckpt_xmax = -1;
    if (const char* e = std::getenv("MARIO_CKPT_XMAX")) ckpt_xmax = std::atoi(e);
    std::vector<int> eligible_ckpts;
    for (int k = 0; k < n_ckpt; ++k)
        if (ckpt_xmax < 0 || env.checkpoint_x(k) <= ckpt_xmax) eligible_ckpts.push_back(k);
    if (curriculum)
        std::printf("   curriculum starts: %d/%d checkpoints eligible (MARIO_CKPT_XMAX=%d)\n",
                    (int)eligible_ckpts.size(), n_ckpt, ckpt_xmax);
    std::fflush(stdout);

    for (int ep = 1; ep <= episodes; ++ep) {
        // Curriculum: a decreasing fraction of episodes start at a checkpoint near
        // the frontier; the rest start from x=0 so the full policy stays practiced.
        float curriculum_prob = cp_start - (cp_start - cp_end) * std::min(1.f, total_steps / cp_anneal);
        std::vector<float> s;
        int ep_ckpt = -1;                 // checkpoint this episode started from (-1 = level start)
        if (curriculum && !eligible_ckpts.empty() && ag::randf() < curriculum_prob) {
            ep_ckpt = eligible_ckpts[(int)(ag::randf() * eligible_ckpts.size()) % (int)eligible_ckpts.size()];
            s = env.reset_to_checkpoint(ep_ckpt);
        } else {
            s = env.reset();
        }
        bool done = false;
        float ep_ret = 0; int ep_max_x = 0;
        std::vector<uint8_t> ep_actions;  // this episode's actions (for win-capture)
        while (!done) {
            float eps = std::max(eps_end, eps_start - (eps_start - eps_end) * total_steps / eps_decay_steps);
            int a = (ag::randf() < eps) ? (int)(ag::randf() * A) % A : greedy_action(online, s);
            std::vector<float> s_next_holder;
            bool d;
            float r = env.step(a, d);
            ep_actions.push_back((uint8_t)a);
            const std::vector<float>& ns = env.observation();
            ep_ret += r; ep_max_x = std::max(ep_max_x, env.mario_x());
            replay.push({s, ns, a, r, d});
            s = ns; done = d;
            ++total_steps;

            if ((int)replay.size() >= warmup && total_steps % train_freq == 0) {
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
            throttle.maybe_sleep();   // yield CPU so the machine stays usable
        }

        // Flagpole reached: stitch a from-boot clear sequence and save it if it is
        // more autonomous (smaller demo prefix) than any previous capture.
        if (env.won()) {
            int prefix = (ep_ckpt >= 0) ? env.checkpoint_demo_len(ep_ckpt) : 0;
            if (prefix < best_win_prefix) {
                best_win_prefix = prefix;
                std::vector<uint8_t> seq;
                if (prefix > 0) {
                    const auto& d = env.demo_actions();
                    seq.assign(d.begin(), d.begin() + std::min<size_t>(prefix, d.size()));
                }
                seq.insert(seq.end(), ep_actions.begin(), ep_actions.end());
                save_actions(win_path, seq);
                std::printf("FLAG ep %d  from_ckpt %d (prefix %d) + %d self actions = %d total -> %s\n",
                            ep, ep_ckpt, prefix, (int)ep_actions.size(), (int)seq.size(), win_path.c_str());
                std::fflush(stdout);
            }
        }

        recent_x.push_back(ep_max_x);
        if (recent_x.size() > 50) recent_x.pop_front();
        double avg_x = 0; for (int x : recent_x) avg_x += x; avg_x /= recent_x.size();

        if (ep % 10 == 0) {   // lightweight heartbeat so progress is visible pre-warmup
            std::printf("  [hb] ep %d  total_steps %ld  replay %d  last_max_x %d\n",
                        ep, total_steps, (int)replay.size(), ep_max_x);
            std::fflush(stdout);
        }

        // Checkpoint by GREEDY skill (not by exploration luck), like CartPole/Othello.
        if (ep % 25 == 0 && (int)replay.size() >= warmup) {
            int gs = 0, gp = 0;
            int gx = greedy_episode(online, env, &gs, &gp);
            int metric = gx + gs;
            if (metric > best_metric) {
                best_metric = metric; best_x = gx; best_score = gs;
                best.copy_from(online); best.save(out_path.c_str());
            }
            float eps = std::max(eps_end, eps_start - (eps_start - eps_end) * total_steps / eps_decay_steps);
            std::printf("ep %4d  ret %7.1f  train_max_x %4d  avg50 %6.1f  GREEDY_x %4d  score %4d  power %d  best_greedy %4d  best_score %4d  eps %.2f  steps %ld\n",
                        ep, ep_ret, ep_max_x, avg_x, gx, gs, gp, best_x, best_score, eps, total_steps);
            std::fflush(stdout);
        }
    }
    std::printf("\nbest greedy: x=%d score=%d (saved %s)\n", best_x, best_score, out_path.c_str());
    return 0;
}
