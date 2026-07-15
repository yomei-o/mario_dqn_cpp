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

// One fully-greedy rollout; returns the max level-x reached (honest skill metric).
static int greedy_episode(QNet& net, mario::Env& env) {
    std::vector<float> s = env.reset();
    bool done = false; int mx = 0;
    while (!done) {
        int a = greedy_action(net, s);
        env.step(a, done);
        s = env.observation();
        mx = std::max(mx, env.mario_x());
    }
    return mx;
}

static void clip_grads(std::vector<Tensor>& ps, float max_norm) {
    double sq = 0;
    for (auto& p : ps) for (float g : p.grad()) sq += (double)g * g;
    float norm = (float)std::sqrt(sq);
    if (norm > max_norm) { float s = max_norm / (norm + 1e-6f); for (auto& p : ps) for (float& g : p.grad()) g *= s; }
}

// Record a greedy episode of the trained agent to web/run.bin so the browser
// viewer can play it back. Format: "MRUN", u32 nframes, u32 w, u32 h, then
// nframes * w*h*3 bytes RGB (one NES frame per agent step).
static int record(const char* rom, const char* weights, const char* out_path) {
    const int S = mario::Env::STATE_DIM, A = mario::N_ACTIONS;
    QNet net(S, A, 256);
    if (!net.load(weights)) std::printf("(warning) could not load %s; using random net\n", weights);
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
    if (argc > 1 && std::strcmp(argv[1], "record") == 0)
        return record(argc > 2 ? argv[2] : "Super Mario Bros (JU) (PRG 0).nes",
                      argc > 3 ? argv[3] : "mario_best.bin",
                      argc > 4 ? argv[4] : "web/run.bin");
    const char* rom = argc > 1 ? argv[1] : "Super Mario Bros (JU) (PRG 0).nes";

    seed(0);
    const int S = mario::Env::STATE_DIM, A = mario::N_ACTIONS;
    const float gamma = 0.99f;
    const int batch = 32, warmup = 5000, target_sync = 2000, episodes = 12000;
    const int train_freq = 4;   // gradient step every N env steps (Atari-DQN style; ~Nx faster wall-clock)
    const float eps_start = 1.0f, eps_end = 0.05f, eps_decay_steps = 200000.f;

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
            int gx = greedy_episode(online, env);
            if (gx > best_x) { best_x = gx; best.copy_from(online); best.save("mario_best.bin"); }
            float eps = std::max(eps_end, eps_start - (eps_start - eps_end) * total_steps / eps_decay_steps);
            std::printf("ep %4d  ret %7.1f  train_max_x %4d  avg50 %6.1f  GREEDY_x %4d  best_greedy %4d  eps %.2f  steps %ld\n",
                        ep, ep_ret, ep_max_x, avg_x, gx, best_x, eps, total_steps);
            std::fflush(stdout);
        }
    }
    std::printf("\nbest greedy distance: x=%d (saved mario_best.bin)\n", best_x);
    return 0;
}
