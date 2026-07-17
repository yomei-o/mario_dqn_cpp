// Record a 1-2 agent run to an MRUN file (RGB frames) for tools/make_gif.sh.
// Separate exe from the trainer so it runs while workers train. Two modes:
//   (default)  mario12_rec [ROM] [net.bin] [out.bin]
//              -- record the net's GREEDY run.
//   search     mario12_rec search [ROM] [net.bin] [out.bin] [trials] [eps] [seed]
//              -- run the net with eps-greedy NOISE for `trials` rollouts from the
//                 1-2 start, keep the FURTHEST-reaching action sequence, then replay
//                 it deterministically and record it. Reproduces a "deep exploration"
//                 run (greedy alone plateaus far short of what exploration reaches).
// One frame per env-step (== the 1-1 recorder cadence).
#include "qnet.h"
#include "mario12.h"
#include "nes.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

using namespace rl;

static int argmax_row(const std::vector<float>& q, int A) {
    int best = 0; for (int a = 1; a < A; ++a) if (q[a] > q[best]) best = a; return best;
}
static int greedy_a(QNet& net, const std::vector<float>& s, int A) {
    auto x = ag::Tensor::from(s, {1, (int)s.size()}, false);
    return argmax_row(net.forward(x).data(), A);
}

// Replay a fixed action sequence from the 1-2 start, dumping one frame per step.
static void record_seq(mario12::Env& env, const std::vector<uint8_t>& acts, const char* out_path) {
    env.reset();
    FILE* f = std::fopen(out_path, "wb");
    if (!f) { std::printf("cannot open %s\n", out_path); return; }
    uint32_t w = nes::WIDTH, h = nes::HEIGHT, nframes = 0;
    std::fwrite("MRUN", 1, 4, f);
    std::fwrite(&nframes, 4, 1, f); std::fwrite(&w, 4, 1, f); std::fwrite(&h, 4, 1, f);
    std::vector<uint8_t> rgb(w * h * 3);
    bool done = false; int mx = 0; bool pipe = false, won = false;
    for (size_t t = 0; t < acts.size() && !done; ++t) {
        env.step(acts[t], done);
        if (env.entered_pipe()) pipe = true;
        if (env.won()) won = true;
        mx = std::max(mx, env.mario_x());
        const uint32_t* px = nes::pixels();
        for (uint32_t i = 0; i < w * h; ++i) {
            rgb[i*3+0] = (px[i] >> 16) & 0xFF; rgb[i*3+1] = (px[i] >> 8) & 0xFF; rgb[i*3+2] = px[i] & 0xFF;
        }
        std::fwrite(rgb.data(), 1, rgb.size(), f);
        ++nframes;
    }
    std::fseek(f, 4, SEEK_SET); std::fwrite(&nframes, 4, 1, f); std::fclose(f);
    std::printf("recorded %u frames  max_x=%d pipe=%d won=%d -> %s\n", nframes, mx, (int)pipe, (int)won, out_path);
}

int main(int argc, char** argv) {
    bool search = argc > 1 && std::strcmp(argv[1], "search") == 0;
    int ai = search ? 2 : 1;
    const char* rom      = argc > ai   ? argv[ai]   : "Super Mario Bros (JU) (PRG 0).nes";
    const char* net_path = argc > ai+1 ? argv[ai+1] : "mario12_best.bin";
    const char* out_path = argc > ai+2 ? argv[ai+2] : "web/run_1-2.bin";
    int   trials = search && argc > ai+3 ? std::atoi(argv[ai+3]) : 400;
    float eps    = search && argc > ai+4 ? (float)std::atof(argv[ai+4]) : 0.15f;
    int   seedv  = search && argc > ai+5 ? std::atoi(argv[ai+5]) : 1;

    const int S = mario12::Env::STATE_DIM, A = mario12::N_ACTIONS, HID = 512;
    QNet net(S, A, HID);
    if (!net.load(net_path)) { std::printf("cannot load net: %s\n", net_path); return 1; }
    mario12::Env env;
    if (!env.init(rom)) { std::printf("env init failed\n"); return 1; }

    if (!search) {
        // greedy record: build the greedy action sequence, then record it.
        std::vector<uint8_t> acts;
        std::vector<float> s = env.reset();
        bool done = false;
        for (int t = 0; t < mario12::Env::MAX_STEPS && !done; ++t) {
            int a = greedy_a(net, s, A); acts.push_back((uint8_t)a);
            env.step(a, done); s = env.observation();
        }
        record_seq(env, acts, out_path);
        return 0;
    }

    // search: eps-greedy rollouts, keep the furthest (prefer pipe/flag).
    ag::seed(seedv);
    std::vector<uint8_t> best_acts; long best_metric = -1; int best_x = 0;
    for (int tr = 0; tr < trials; ++tr) {
        std::vector<float> s = env.reset();
        int start_area = env.area(), max_area = start_area, mx = 0;
        bool done = false; std::vector<uint8_t> acts;
        while (!done) {
            int a = (ag::randf() < eps) ? (int)(ag::randf() * A) % A : greedy_a(net, s, A);
            acts.push_back((uint8_t)a);
            env.step(a, done); s = env.observation();
            mx = std::max(mx, env.mario_x());
            max_area = std::max(max_area, env.area());
        }
        long metric = (long)(max_area - start_area) * 100000L + mx;
        if (metric > best_metric) {
            best_metric = metric; best_x = mx; best_acts = acts;
            std::printf("  trial %d: new best max_x=%d area+%d (len %d)\n",
                        tr, mx, max_area - start_area, (int)acts.size());
            std::fflush(stdout);
        }
    }
    std::printf(">> search done: best max_x=%d over %d trials\n", best_x, trials);
    record_seq(env, best_acts, out_path);
    return 0;
}
