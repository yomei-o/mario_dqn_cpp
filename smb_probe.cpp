// Sanity tool for smb_env: warp to a stage, run right for a while, and print
// what the agent sees (the egocentric board) plus the raw tile ids around Mario.
//   smb_probe ROM 1-2 [steps]
#include "smb_env.h"
#include "mario_shared.h"
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>
#include <cstring>
#include <cctype>

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: smb_probe ROM W-S [steps]\n"); return 1; }
    smbx::Level lv;
    if (!smbx::parse_level(argv[2], lv)) { std::printf("bad level %s\n", argv[2]); return 1; }
    // 3rd arg: a step count of plain runs, or a macro string like R26J9R60
    // (R run, J runjump, H hop, W wait, B back, X backjump; letter + count).
    std::vector<int> plan;
    std::string spec = argc > 3 ? argv[3] : "30";
    if (spec.size() > 4 && spec.compare(spec.size() - 4, 4, ".bin") == 0) {   // action file
        FILE* f = std::fopen(spec.c_str(), "rb");
        uint32_t n = 0;
        if (f && std::fread(&n, 4, 1, f) == 1) {
            std::vector<uint8_t> a(n);
            if (std::fread(a.data(), 1, n, f) == n) plan.assign(a.begin(), a.end());
        }
        if (f) std::fclose(f);
    } else if (!spec.empty() && std::isdigit((unsigned char)spec[0])) plan.assign(std::atoi(spec.c_str()), smbx::A_RUN);
    else for (size_t i = 0; i < spec.size();) {
        const char* letters = "RJHWBX";
        const char* p = std::strchr(letters, spec[i]);
        size_t j = i + 1; int n = 0;
        while (j < spec.size() && std::isdigit((unsigned char)spec[j])) n = n * 10 + (spec[j++] - '0');
        if (p) plan.insert(plan.end(), std::max(1, n), (int)(p - letters));
        i = j;
    }
    int steps = (int)plan.size();
    smbx::Env env;
    if (!env.init(argv[1])) return 1;
    env.set_level(lv);
    env.reset();
    std::printf("level %s: world=%d stage=%d area_index=%d area_type=%d x=%d y=%d lives=%d\n",
                lv.name().c_str(), smb::world_num() + 1, smb::RAM(0x075C) + 1, smb::area_index(),
                smb::area_type(), smb::level_x(), smb::player_y(), smb::lives());
    bool done = false;
    int t = 0;
    for (; t < steps && !done; ++t) env.step(plan[t], done);
    std::printf("after %d steps: x=%d y=%d progress=%d power=%d done=%d won=%d\n", t, env.mario_x(),
                smb::player_y(), env.progress(), env.power(), (int)done, (int)env.won());

    const auto& o = env.observation();
    using E = smbx::Env;
    const char* ch = "#?EPM";
    for (int r = 0; r < E::GRID_R; ++r) {
        std::printf("row%2d ", r);
        for (int c = 0; c < E::GRID_W; ++c) {
            char out = '.';
            for (int k = E::GRID_C - 1; k >= 0; --k)
                if (o[(size_t)(k * E::GRID_R + r) * E::GRID_W + c] > 0.5f) out = ch[k];
            std::printf("%c", out);
        }
        std::printf("   ");
        int col0 = smb::level_x() / 16 - E::GRID_BEHIND;
        for (int c = 0; c < E::GRID_W; ++c) std::printf("%02X ", smb::tile_at((col0 + c) * 16 + 8, 32 + r * 16 + 8));
        std::printf("\n");
    }
    std::printf("scalars:");
    for (int k = 0; k < E::N_SCALAR; ++k) std::printf(" %.2f", o[E::GRID_N + k]);
    std::printf("\n");
    return 0;
}
