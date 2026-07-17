#pragma once
// Generic greedy-run recorder for a warp-started level Env. Records one frame per
// env-step to an MRUN file (RGB) for tools/make_gif.sh. Shared by record13/record14.
// Env must expose STATE_DIM, ACTIONS, init, reset, step, observation, mario_x,
// won, area.  Usage: return rec::record_level<mario13::Env>(argc, argv, "run.bin");
#include "qnet.h"
#include "nes.h"
#include "autograd.h"
#include <cstdio>
#include <vector>
#include <algorithm>

namespace rec {

template <class Env>
int record_level(int argc, char** argv, const char* default_out) {
    const char* rom      = argc > 1 ? argv[1] : "Super Mario Bros (JU) (PRG 0).nes";
    const char* net_path = argc > 2 ? argv[2] : "mario_best.bin";
    const char* out_path = argc > 3 ? argv[3] : default_out;

    const int S = Env::STATE_DIM, A = Env::ACTIONS, HID = 512;
    rl::QNet net(S, A, HID);
    if (!net.load(net_path)) { std::printf("cannot load net: %s\n", net_path); return 1; }
    Env env;
    if (!env.init(rom)) { std::printf("env init failed\n"); return 1; }
    std::vector<float> s = env.reset();

    FILE* f = std::fopen(out_path, "wb");
    if (!f) { std::printf("cannot open %s\n", out_path); return 1; }
    uint32_t w = nes::WIDTH, h = nes::HEIGHT, nframes = 0;
    std::fwrite("MRUN", 1, 4, f);
    std::fwrite(&nframes, 4, 1, f); std::fwrite(&w, 4, 1, f); std::fwrite(&h, 4, 1, f);

    std::vector<uint8_t> rgb(w * h * 3);
    bool done = false; int mx = 0; bool won = false;
    for (int t = 0; t < Env::MAX_STEPS && !done; ++t) {
        auto x = ag::Tensor::from(s, {1, S}, false);
        auto out = net.forward(x);            // keep the Tensor alive (data() refs its storage)
        const auto& q = out.data();
        int a = 0; for (int i = 1; i < A; ++i) if (q[i] > q[a]) a = i;
        env.step(a, done);
        s = env.observation();
        if (env.won()) won = true;
        mx = std::max(mx, env.mario_x());
        const uint32_t* px = nes::pixels();
        for (uint32_t i = 0; i < w * h; ++i) {
            rgb[i*3+0]=(px[i]>>16)&0xFF; rgb[i*3+1]=(px[i]>>8)&0xFF; rgb[i*3+2]=px[i]&0xFF;
        }
        std::fwrite(rgb.data(), 1, rgb.size(), f);
        ++nframes;
    }
    std::fseek(f, 4, SEEK_SET); std::fwrite(&nframes, 4, 1, f); std::fclose(f);
    std::printf("recorded %u frames  max_x=%d won=%d -> %s\n", nframes, mx, (int)won, out_path);
    return 0;
}

}  // namespace rec
