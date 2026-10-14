// All-stage trainer on smbx::Env: one net, several stages, the same rules.
//
//   smb_train searchdemo ROM W-S out.bin           clean clear demo for one stage by search
//   smb_train train ROM --levels 1-1,1-2 [--eval 2-1] [--demos demos] [--actors 16]
//                       [--minutes 60] [--warm net.bin]
//   smb_train eval ROM net.bin W-S[,W-S...]          greedy run per stage
//   smb_train record ROM net.bin W-S out.bin         MRUN video of the greedy run (tools/make_gif.sh)
//   smb_train recorddemo ROM W-S demo.bin out.bin    MRUN video of a demo (action file)
//   smb_train actor ...                               (internal, spawned by train)
//
// Same Ape-X layout and DQfD loss as mario_gpu (see there and apex_common.hpp);
// what changes is the env: every stage is reached by warp, observed as an
// egocentric board, and scored by one reward. Demos are per stage
// (<demos>/smb_<W-S>.bin); actors cycle through the training stages; the
// learner evaluates the greedy run on every training stage AND on held-out
// --eval stages (never trained on) -- the second number is the one that says
// whether the net learned rules rather than routes.
#include "smb_env.h"
#include "mario_shared.h"
#include "apex_common.hpp"
#include "conv_net.h"
#include <map>
#include <deque>
#include <sstream>
#include <set>

using namespace apex;
using smbx::Env;
using smbx::Level;

// Q-model: the MLP (rl::QNet) or the conv net (rl::ConvQ, conv_net.h). A new
// net is conv when SMB_NET=conv; a saved net's kind and width are read back
// from its file. Hidden width: SMB_HID (default 512).
struct QModel {
    virtual ~QModel() {}
    virtual bool conv() const = 0;
    virtual int hidden() const = 0;
    virtual int act(const std::vector<float>& s) = 0;
    virtual bool load(const std::string& path) = 0;
    virtual bool save(const std::string& path) = 0;
    virtual void get(std::vector<std::vector<float>>& p) = 0;
    virtual void set(const std::vector<std::vector<float>>& p) = 0;
};
struct MlpModel : QModel {
    rl::QNet net;
    MlpModel(int H) : net(Env::STATE_DIM, smbx::N_ACTIONS, H) {}
    bool conv() const override { return false; }
    int hidden() const override { return net.hdim; }
    int act(const std::vector<float>& s) override { return greedy_action(net, s); }
    bool load(const std::string& p) override { return net.load(p) || net.load_expand(p); }
    bool save(const std::string& p) override { return net.save(p); }
    void get(std::vector<std::vector<float>>& p) override { qnet_get(net, p); }
    void set(const std::vector<std::vector<float>>& p) override { qnet_set(net, p); }
};
struct ConvModel : QModel {
    rl::ConvQ net;
    ConvModel(int H) : net(smbx::N_ACTIONS, H) { net.init_random(1); }
    bool conv() const override { return true; }
    int hidden() const override { return net.arch.H; }
    int act(const std::vector<float>& s) override {
        float q[16];
        net.forward(s.data(), 1, q);
        return argmax(q, smbx::N_ACTIONS);
    }
    bool load(const std::string& p) override { return net.load(p); }
    bool save(const std::string& p) override { return net.save(p); }
    void get(std::vector<std::vector<float>>& p) override { net.get_params(p); }
    void set(const std::vector<std::vector<float>>& p) override { net.set_params(p); }
};
inline int greedy_action(QModel& m, const std::vector<float>& s) { return m.act(s); }

static const char* RUN_DIR = "smb_run";
static int hid_default() { const char* e = std::getenv("SMB_HID"); return e ? std::atoi(e) : 512; }
static std::unique_ptr<QModel> new_model(bool conv, int H) {
    if (conv) return std::unique_ptr<QModel>(new ConvModel(H));
    return std::unique_ptr<QModel>(new MlpModel(H));
}
// Model matching a weights file (kind + width), or a fresh default if unreadable.
static std::unique_ptr<QModel> model_for(const char* path) {
    int H = 0;
    if (rl::is_conv_file(path, &H) && H > 0) return new_model(true, H);
    FILE* f = std::fopen(path, "rb");
    int n = 0;
    if (f) { if (std::fread(&n, sizeof(int), 1, f) != 1) n = 0; std::fclose(f); }
    if (n > 0 && n % Env::STATE_DIM == 0) return new_model(false, n / Env::STATE_DIM);
    const char* k = std::getenv("SMB_NET");
    return new_model(k && std::strcmp(k, "conv") == 0, hid_default());
}

static std::vector<Level> parse_levels(const std::string& csv) {
    std::vector<Level> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty()) { Level lv; if (smbx::parse_level(item, lv)) out.push_back(lv); else std::printf("bad level '%s'\n", item.c_str()); }
    return out;
}

static std::string demo_path(const std::string& dir, const Level& lv) { return dir + "/smb_" + lv.name() + ".bin"; }

// ---------------------------------------------------------------- search demo
// Hold RUN; when the run ends (death / stuck), try interventions started s
// steps before the last progress point, cheapest kind first:
//   1. running jump held d steps (latest working start, then moved to the
//      (s, d) whose +-1 neighbours also pass, so an imitating net has slack)
//   2. wait k / back k / wait-then-jump / hop  (enemies to wait out, low ceilings)
// and, while still small, "stop under a ?-block, hop into it, wait for the
// item, run" wherever a ?-block is overhead -- a power-up scores +600 in the
// search, so the demo takes the mushroom whenever it can.
struct Cand { std::vector<uint8_t> acts; };

static int searchdemo(const char* rom, const Level& lv, const char* out) {
    using smbx::A_RUN; using smbx::A_RUNJUMP; using smbx::A_HOP; using smbx::A_WAIT; using smbx::A_BACK;
    const int horizon = 60;
    Env env;
    if (!env.init(rom)) return 1;
    env.set_level(lv);
    env.reset();
    using Snap = Env::Snapshot;
    std::vector<uint8_t> C;
    std::vector<Snap> S(1);
    env.snapshot(S[0]);
    struct Decision { size_t base; int s; std::vector<uint8_t> m; };
    std::set<int> tried_blocks;
    std::vector<Decision> stack;
    std::vector<std::pair<size_t, std::vector<uint8_t>>> tabu;
    auto t0 = Clock::now();
    long sim = 0;
    auto rep = [](int a, int n) { return std::vector<uint8_t>(n, (uint8_t)a); };
    auto cat = [](std::vector<uint8_t> a, const std::vector<uint8_t>& b) { a.insert(a.end(), b.begin(), b.end()); return a; };

    for (int iter = 0; iter < 3000; ++iter) {
        size_t L = C.size();
        env.restore(S[L]);
        int pw0 = env.power();
        std::vector<Snap> run(1);
        env.snapshot(run[0]);
        bool done = false;
        int best_p = env.progress(), last_prog = 0;
        while (!done) {
            env.step(A_RUN, done); ++sim;
            run.emplace_back(); env.snapshot(run.back());
            if (env.progress() > best_p) { best_p = env.progress(); last_prog = (int)run.size() - 1; }
        }
        if (env.won()) {
            C.insert(C.end(), run.size() - 1, (uint8_t)A_RUN);
            int L2 = replay_win_len(env, C);
            if (L2 < 0) { std::printf("internal error: search result does not replay\n"); return 1; }
            C.resize(L2);
            save_actions(out, C);
            env.reset();
            bool d2 = false; int maxpw = 0;
            for (auto a : C) { env.step(a, d2); maxpw = std::max(maxpw, env.power()); }
            std::printf("CLEAR %s: %zu actions, %zu decisions, power-up %s -> %s  [%.0fs, %ld sim steps]\n",
                        lv.name().c_str(), C.size(), stack.size(), maxpw ? "taken" : "not taken", out, secs_since(t0), sim);
            return 0;
        }
        int D = (int)run.size() - 1;
        const double death_p = best_p;
        const double pass = 1e6;            // safely past (see score_of)

        // A candidate is judged by whether it gets Mario SAFELY past the point
        // where running alone ended: after the macro, keep running until he
        // stands on the ground again beyond that point (+24px). That scores
        // 1e6 + progress (+-600 per power level gained/lost); whatever comes
        // after the landing is the next decision's job. Unsafe candidates score
        // their best progress, no survival bonus (it rewarded hopping in place).
        // While small, a safe candidate is followed for up to 40 more run steps:
        // if that catches a power-up (alive), it earns +600 and those run steps
        // are committed with it (grab_run), so the next decision cannot steer
        // away from the mushroom before it is caught.
        int grab_run = 0;
        auto score_of = [&](int s, const std::vector<uint8_t>& macro) -> double {
            grab_run = 0;
            for (auto& tb : tabu) if (tb.first == L + s && tb.second == macro) return -1;
            env.restore(run[s]);
            bool dn = false;
            int m = 0;
            for (size_t k = 0; k < macro.size() && !dn; ++k) { env.step(macro[k], dn); ++sim; m = std::max(m, env.progress()); }
            bool safe = false;
            int ran = 0;
            for (int h = 0; h < horizon && !dn && !safe; ++h) {
                if (env.power() > pw0) break;
                if (smb::float_state() == 0 && env.progress() >= death_p + 24) { safe = true; break; }
                env.step(A_RUN, dn); ++sim; ++ran; m = std::max(m, env.progress());
            }
            if (env.won()) return 1e9;
            if (safe && pw0 == 0 && env.power() == pw0)
                for (int h = 0; h < 40 && !dn && env.power() == pw0; ++h) { env.step(A_RUN, dn); ++sim; ++ran; }
            if (env.power() > pw0 && !dn) {           // caught a power-up, still alive
                grab_run = ran;
                return 1e6 + std::max(m, env.progress()) + 600.0 * (env.power() - pw0);
            }
            if (safe && env.power() < pw0) return 5e5 + m;   // got past but took a hit
            return safe ? 1e6 + m : m + 600.0 * (env.power() - pw0);
        };
        int best_s = -1; std::vector<uint8_t> best_m; double best = death_p + 8;
        int s_hi = std::min(D - 1, last_prog + 5), s_lo = std::max(0, last_prog - 40);
        int best_grab = 0;
        auto consider = [&](int s, const std::vector<uint8_t>& m) {
            double sc = score_of(s, m);
            if (sc > best) { best = sc; best_s = s; best_m = m; best_grab = grab_run; }
        };

        // Power-up attempts: a power-up ?-block (metatile 0xC1; coin blocks are 0xC0)
        // within reach above, ahead or a little behind. Walk/run to it, stop, hop
        // into it, wait for the item, then run (the mushroom slides right and is
        // caught on the way). Each block is attempted once per small-Mario visit.
        if (pw0 == 0) {
            for (int s = 0; s <= std::min(D - 1, last_prog + 5); s += 3) {
                env.restore(run[s]);
                if (smb::float_state() != 0) continue;
                int lx = env.mario_x(), py = smb::player_y(), bx = -1;
                for (int dx = -128; dx <= 96 && bx < 0; dx += 16)
                    for (int up = 2; up <= 4 && bx < 0; ++up)
                        if (smb::tile_at(lx + 8 + dx, py + 16 - up * 16 + 8) == 0xC1) bx = (lx + 8 + dx) / 16;
                if (bx < 0 || tried_blocks.count(bx * 100000 + (int)(L + s) / 30)) continue;
                tried_blocks.insert(bx * 100000 + (int)(L + s) / 30);
                int dir = bx * 16 + 8 >= lx + 8 ? A_RUN : A_BACK;
                for (int m : {0, 4, 8, 12, 16, 20})
                    for (int wait : {0, 15, 30})
                        consider(s, cat(cat(cat(rep(dir, m), rep(A_WAIT, 4)), rep(A_HOP, 8)), rep(A_WAIT, wait)));
            }
        }
        // 1. jumps (latest start first)
        const int durs[] = {2, 4, 6, 9, 13, 18, 24};
        for (int s = s_hi; s >= s_lo && best < pass; --s)
            for (int d : durs) consider(s, rep(A_RUNJUMP, d));
        // robustify a plain jump pick
        if (best >= pass && best < 1e9 && !best_m.empty() && best_m[0] == A_RUNJUMP &&
            std::all_of(best_m.begin(), best_m.end(), [](uint8_t a) { return a == A_RUNJUMP; })) {
            int s0 = best_s, d0 = (int)best_m.size(), best_rob = -1;
            std::map<std::pair<int, int>, bool> ok;
            auto passes = [&](int s, int d) {
                if (s < 0 || s >= (int)run.size() || d < 1) return false;
                auto k = std::make_pair(s, d);
                auto it = ok.find(k);
                if (it != ok.end()) return it->second;
                return ok[k] = score_of(s, rep(A_RUNJUMP, d)) >= pass;
            };
            for (int s = s0; s >= std::max(0, s0 - 6); --s)
                for (int d = std::max(1, d0 - 2); d <= d0 + 2; ++d) {
                    if (!passes(s, d)) continue;
                    int rob = 0;
                    for (int ds = -1; ds <= 1; ++ds) for (int dd = -1; dd <= 1; ++dd) rob += passes(s + ds, d + dd);
                    if (rob > best_rob) { best_rob = rob; best_s = s; best_m = rep(A_RUNJUMP, d); }
                }
            score_of(best_s, best_m); best_grab = grab_run;   // re-derive the committed run-on
        }
        // 2. waiting / backing off / hops, if no jump got past
        if (best < pass)
            for (int s = s_hi; s >= s_lo; s -= 2) {
                for (int k : {4, 8, 16, 24, 32}) consider(s, rep(A_WAIT, k));
                for (int k : {4, 8, 16}) consider(s, rep(A_BACK, k));
                for (int k : {4, 8, 16}) for (int d : {4, 9, 13}) consider(s, cat(rep(A_WAIT, k), rep(A_RUNJUMP, d)));
                for (int d : {2, 4, 6}) consider(s, rep(A_HOP, d));
            }

        // 3. fine-grained timing right before the stuck point (moving lifts,
        //    fire bars, enemies under a ceiling): wait 2..48 then run / jump / back-jump.
        if (best < pass)
            for (int s = std::max(0, last_prog - 12); s <= std::min(D - 1, last_prog + 2); ++s)
                for (int k = 2; k <= 48; k += 2) {
                    consider(s, rep(A_WAIT, k));
                    for (int d : {4, 9}) consider(s, cat(rep(A_WAIT, k), rep(A_RUNJUMP, d)));
                    consider(s, cat(rep(A_BACK, k / 2), rep(A_RUNJUMP, 9)));
                }

        if (best_s < 0) {
            if (stack.empty()) { std::printf("search failed on %s at progress %.0f\n", lv.name().c_str(), death_p); return 1; }
            Decision dd = stack.back(); stack.pop_back();
            tabu.emplace_back(dd.base + dd.s, dd.m);
            {   // where it got stuck, for `smb_probe ROM W-S <out>.stuck.bin`
                std::vector<uint8_t> upto = C;
                upto.insert(upto.end(), last_prog, (uint8_t)A_RUN);
                save_actions(std::string(out) + ".stuck.bin", upto);
            }
            C.resize(dd.base); S.resize(dd.base + 1);
            std::printf("  backtrack: undo decision at step %zu [stuck at %.0f]\n", dd.base, death_p);
            std::fflush(stdout);
            continue;
        }
        std::vector<uint8_t> macro = rep(A_RUN, best_s);
        macro.insert(macro.end(), best_m.begin(), best_m.end());
        macro.insert(macro.end(), best_grab, (uint8_t)A_RUN);   // run on to the power-up
        stack.push_back({L, best_s, best_m});
        env.restore(S[L]);
        for (auto a : macro) { bool dn; env.step(a, dn); C.push_back(a); S.emplace_back(); env.snapshot(S.back()); }
        std::string ms;
        for (size_t i = best_s; i < macro.size(); ++i) ms += "RJHWBX"[macro[i]];
        std::printf("  %s %3zu: step %4zu %-14s (run alone ended at %.0f, now %s%.0f)%s [%.0fs]\n", lv.name().c_str(), stack.size(),
                    L + best_s, ms.c_str(), death_p, best >= 1e9 ? "the flag " : best >= 1e6 ? "safe " : best >= 5e5 ? "hit " : "", best >= 1e9 ? 0.0 : best >= 1e6 ? best - 1e6 : best >= 5e5 ? best - 5e5 : best,
                    env.power() > pw0 ? " POWER-UP" : "", secs_since(t0));
        std::fflush(stdout);
    }
    std::printf("search gave up on %s\n", lv.name().c_str());
    return 1;
}

// ---------------------------------------------------------------- eval / record

static int evalnet(const char* rom, const char* weights, const std::vector<Level>& levels) {
    auto netp = model_for(weights);
    QModel& net = *netp;
    if (!net.load(weights)) { std::printf("cannot load %s\n", weights); return 1; }
    Env env;
    if (!env.init(rom)) return 1;
    int cleared = 0;
    for (auto& lv : levels) {
        env.set_level(lv);
        bool won = false;
        std::vector<uint8_t> acts;
        int p = greedy_run(net, env, won, &acts);
        cleared += won;
        std::printf("%s greedy progress=%d won=%d steps=%zu power=%d\n", lv.name().c_str(), p, (int)won, acts.size(), env.power());
    }
    std::printf("cleared %d/%zu\n", cleared, levels.size());
    return 0;
}

// MRUN video of one stage, driven by a net (greedy) or, with `is_demo`, by a
// saved action sequence (a searchdemo result). 120 hands-off steps follow a
// win so the flag / castle sequence is in the clip.
static int record(const char* rom, const char* src, const Level& lv, const char* out_path, bool is_demo) {
    std::unique_ptr<QModel> netp;
    std::vector<uint8_t> demo;
    if (is_demo) {
        if (!load_actions(src, demo)) { std::printf("cannot load demo %s\n", src); return 1; }
    } else {
        netp = model_for(src);
        if (!netp->load(src)) { std::printf("cannot load %s\n", src); return 1; }
    }
    Env env;
    if (!env.init(rom)) return 1;
    env.set_level(lv);
    std::vector<float> s = env.reset();
    FILE* f = std::fopen(out_path, "wb");
    if (!f) return 1;
    uint32_t w = nes::WIDTH, h = nes::HEIGHT, nframes = 0;
    std::fwrite("MRUN", 1, 4, f);
    std::fwrite(&nframes, 4, 1, f); std::fwrite(&w, 4, 1, f); std::fwrite(&h, 4, 1, f);
    std::vector<uint8_t> rgb(w * h * 3);
    auto put = [&]() {
        const uint32_t* px = nes::pixels();
        for (uint32_t i = 0; i < w * h; ++i) { rgb[i*3] = (px[i] >> 16) & 0xFF; rgb[i*3+1] = (px[i] >> 8) & 0xFF; rgb[i*3+2] = px[i] & 0xFF; }
        std::fwrite(rgb.data(), 1, rgb.size(), f); ++nframes;
    };
    bool done = false;
    size_t t = 0;
    while (!done) {
        int a = is_demo ? (t < demo.size() ? demo[t] : smbx::A_WAIT) : greedy_action(*netp, s);
        ++t;
        env.step(a, done); s = env.observation(); put();
    }
    if (env.won())
        for (int t = 0; t < 120; ++t) { nes::set_buttons(0, 0); for (int k = 0; k < Env::FRAME_SKIP; ++k) nes::step_frame(); put(); }
    std::fseek(f, 4, SEEK_SET); std::fwrite(&nframes, 4, 1, f); std::fclose(f);
    std::printf("recorded %u frames (%s progress=%d won=%d power=%d) -> %s\n", nframes, lv.name().c_str(),
                env.progress(), (int)env.won(), env.power(), out_path);
    return 0;
}

// ---------------------------------------------------------------- parity
// CPU and CUDA conv learners from the same weights on the same batches must agree.
static int parity_conv() {
    const int A = smbx::N_ACTIONS, H = 128, B = 32;
    auto gpu = rl::make_conv_cuda_learner(A, H);
    if (!gpu) { std::printf("no CUDA conv learner in this build / no GPU\n"); return 1; }
    auto cpu = rl::make_conv_cpu_learner(A, H);
    rl::ConvQ init(A, H);
    init.init_random(7);
    std::vector<std::vector<float>> p;
    init.get_params(p);
    cpu->set_params(p); gpu->set_params(p);
    const int S = rl::ConvArch::IN;
    rl::Batch b; b.resize(B, S);
    ag::seed(3);
    float worst = 0.f;
    for (int it = 0; it < 10; ++it) {
        for (auto& v : b.s) v = ag::randf() < 0.15f ? 1.f : 0.f;
        for (auto& v : b.ns) v = ag::randf() < 0.15f ? 1.f : 0.f;
        for (int r = 0; r < B; ++r) {
            b.a[r] = (int)(ag::randf() * A) % A; b.ret[r] = ag::randf() * 4 - 2;
            b.gn[r] = ag::randf() < 0.1f ? 0.f : 0.95f; b.demo[r] = ag::randf() < 0.3f;
        }
        float lc = cpu->train(b), lg = gpu->train(b);
        std::vector<std::vector<float>> pc, pg;
        cpu->get_params(pc); gpu->get_params(pg);
        float md = 0.f;
        for (size_t i = 0; i < pc.size(); ++i)
            for (size_t j = 0; j < pc[i].size(); ++j) md = std::max(md, std::fabs(pc[i][j] - pg[i][j]));
        worst = std::max(worst, md);
        std::printf("step %d  loss cpu %.6f gpu %.6f  max|dW| %.2e\n", it, lc, lg, md);
    }
    std::vector<float> qc(B * A), qg(B * A), qi(B * A);
    cpu->forward(b.s.data(), B, qc.data()); gpu->forward(b.s.data(), B, qg.data());
    rl::ConvQ inf(A, H);
    std::vector<std::vector<float>> pc; cpu->get_params(pc); inf.set_params(pc);
    inf.forward(b.s.data(), B, qi.data());
    float mq = 0.f, mi = 0.f;
    for (int i = 0; i < B * A; ++i) { mq = std::max(mq, std::fabs(qc[i] - qg[i])); mi = std::max(mi, std::fabs(qc[i] - qi[i])); }
    std::printf("forward max|dQ| cpu-gpu %.2e  cpu-inference %.2e\n", mq, mi);
    bool ok = worst < 1e-3f && mq < 1e-3f && mi < 1e-5f;
    std::printf("conv parity %s\n", ok ? "OK" : "MISMATCH");
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------- actor
static int actor(const char* rom, int id, int n_actors, const std::string& levels_csv, const std::string& demos) {
    std::string dir = RUN_DIR;
    std::freopen((dir + "/actor_" + std::to_string(id) + ".log").c_str(), "w", stdout);
    ag::seed(1000 + id);
    const int S = Env::STATE_DIM, A = smbx::N_ACTIONS;
    float eps = n_actors > 1 ? std::pow(0.4f, 1.f + 7.f * id / (n_actors - 1)) : 0.1f;
    std::vector<Level> levels = parse_levels(levels_csv);
    std::vector<Env> envs(levels.size());
    for (size_t i = 0; i < levels.size(); ++i) {
        envs[i].init(rom);
        envs[i].set_level(levels[i]);
        std::vector<uint8_t> d;
        if (load_actions(demo_path(demos, levels[i]), d)) envs[i].build_checkpoints(d, 24);
        std::printf("actor %d: %s checkpoints=%d\n", id, levels[i].name().c_str(), envs[i].num_checkpoints());
    }
    std::printf("actor %d/%d eps=%.4f\n", id, n_actors, eps);
    std::fflush(stdout);
    std::string wpath = dir + "/weights.bin", vpath = dir + "/weights.ver";
    while (!fs::exists(wpath)) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto netp = model_for(wpath.c_str());
    QModel& net = *netp;
    int have_ver = -1, seq = 0;
    auto t_last = Clock::now();
    for (long ep = id;; ++ep) {
        if (fs::exists(dir + "/stop")) break;
        int ver = -1;
        if (FILE* f = std::fopen(vpath.c_str(), "r")) { if (std::fscanf(f, "%d", &ver) != 1) ver = -1; std::fclose(f); }
        if (ver != have_ver && ver >= 0) { if (net.load(wpath)) { have_ver = ver; t_last = Clock::now(); } }
        else if (secs_since(t_last) > 300) { std::printf("no new weights for 300s -> exiting\n"); break; }
        if (have_ver < 0) { std::this_thread::sleep_for(std::chrono::milliseconds(200)); continue; }

        Env& env = envs[ep % envs.size()];
        Episode e; e.S = S; e.actor = id;
        std::vector<float> s;
        if (env.num_checkpoints() > 0 && ag::randf() < 0.5f) {
            e.start_ckpt = (int)(ag::randf() * env.num_checkpoints()) % env.num_checkpoints();
            s = env.reset_to_checkpoint(e.start_ckpt);
        } else s = env.reset();
        e.obs.insert(e.obs.end(), s.begin(), s.end());
        bool done = false;
        while (!done) {
            int a = ag::randf() < eps ? (int)(ag::randf() * A) % A : greedy_action(net, s);
            float r = env.step(a, done);
            s = env.observation();
            e.obs.insert(e.obs.end(), s.begin(), s.end());
            e.act.push_back(a); e.rew.push_back(r);
            e.max_x = std::max(e.max_x, env.progress());
        }
        e.T = (int)e.act.size(); e.won = env.won();
        write_episode(dir + "/ep/a" + std::to_string(id) + "_" + std::to_string(seq++) + ".ep", e);
    }
    return 0;
}

// ---------------------------------------------------------------- learner
static int train(const char* exe, const char* rom, const std::vector<Level>& levels, const std::vector<Level>& held,
                 const std::string& levels_csv, const std::string& demos, int n_actors, double minutes, const char* warm) {
    const int S = Env::STATE_DIM, A = smbx::N_ACTIONS;
    const char* kind = std::getenv("SMB_NET");
    auto netp = warm && *warm ? model_for(warm) : new_model(kind && std::strcmp(kind, "conv") == 0, hid_default());
    QModel& net = *netp;
    const int HID = net.hidden();
    const int B = 256, target_sync = 2000, publish_every = 200, pretrain_steps = 30000;
    const float demo_frac = 0.25f;
    double eval_period = 60.0;
    float ratio = 16.f;
    if (const char* e = std::getenv("MARIO_RATIO")) ratio = (float)std::atof(e);
    if (const char* e = std::getenv("MARIO_EVAL_SEC")) eval_period = std::atof(e);

    std::string dir = RUN_DIR;
    fs::create_directories(dir + "/ep");
    fs::remove(dir + "/stop");
    for (auto& f : fs::directory_iterator(dir + "/ep")) fs::remove(f.path());

    std::unique_ptr<rl::Learner> L;
    const char* want = std::getenv("MARIO_LEARNER");
    bool use_gpu = !(want && std::strcmp(want, "cpu") == 0);
    if (net.conv()) {
        if (use_gpu) L = rl::make_conv_cuda_learner(A, HID);
        if (!L) L = rl::make_conv_cpu_learner(A, HID);
    } else {
        if (use_gpu) L = rl::make_cuda_learner(S, A, HID);
        if (!L) L = rl::make_cpu_learner(S, A, HID);
    }
    L->lr = 1e-4f;
    if (const char* e = std::getenv("MARIO_LR")) L->lr = (float)std::atof(e);

    ag::seed(1);
    if (!net.conv()) ag::seed(1);
    bool warmed = warm && *warm && net.load(warm);
    std::vector<std::vector<float>> P;
    net.get(P);
    L->set_params(P);

    Env env;
    if (!env.init(rom)) return 1;
    std::vector<Episode> demo_eps;
    size_t demo_steps = 0;
    for (auto& lv : levels) {
        std::vector<uint8_t> acts;
        if (!load_actions(demo_path(demos, lv), acts)) { std::printf("   (no demo for %s)\n", lv.name().c_str()); continue; }
        env.set_level(lv);
        demo_eps.push_back(demo_episode(env, acts));
        demo_steps += demo_eps.back().T;
        std::printf("   demo %s: %d steps, won=%d\n", lv.name().c_str(), demo_eps.back().T, demo_eps.back().won);
    }
    std::printf("== smb_train: DQfD on %zu stages (+%zu held out), learner=%s, %d actors, %.0f min%s ==\n",
                levels.size(), held.size(), L->name(), n_actors, minutes, warmed ? ", warm start" : "");
    std::printf("   state=%d actions=%d hidden=%d batch=%d ratio=%.1f\n", S, A, HID, B, ratio);
    std::fflush(stdout);

    NStepReplay demo_buf(S, demo_steps + 1), rep(S, 600000);
    for (auto& e : demo_eps) demo_buf.add_episode(e);
    rl::Batch batch; batch.resize(B, S);
    auto sample = [&](int n_demo) {
        for (int r = 0; r < B; ++r) {
            if ((r < n_demo || rep.size == 0) && demo_buf.size)
                demo_buf.fill_row((size_t)(ag::randf() * demo_buf.size) % demo_buf.size, batch, r, 1.f);
            else
                rep.fill_row((size_t)(ag::randf() * rep.size) % rep.size, batch, r, 0.f);
        }
    };

    auto t0 = Clock::now();
    std::map<std::string, int> best_p;
    int best_cleared = -1;
    auto evaluate = [&](const char* tag) {
        L->get_params(P); net.set(P);
        int cl = 0, cl_held = 0;
        std::string line;
        for (int pass = 0; pass < 2; ++pass)
            for (auto& lv : pass ? held : levels) {
                env.set_level(lv);
                bool won = false;
                int p = greedy_run(net, env, won, nullptr);
                (pass ? cl_held : cl) += won;
                best_p[lv.name()] = std::max(best_p[lv.name()], p);
                char buf[96];
                std::snprintf(buf, sizeof buf, " %s%s:%d%s", pass ? "*" : "", lv.name().c_str(), p, won ? "W" : (env.power() ? "+" : ""));
                line += buf;
            }
        if (cl > best_cleared) { best_cleared = cl; net.save("smb_best.bin"); }
        net.save("smb_latest.bin");
        std::printf("[%7.1fs] %-8s cleared %d/%zu held-out %d/%zu |%s\n", secs_since(t0), tag, cl, levels.size(),
                    cl_held, held.size(), line.c_str());
        std::fflush(stdout);
        return cl == (int)levels.size();
    };

    if (!warmed && demo_buf.size) {
        const float lr_rl = L->lr;
        L->lr = 5e-4f;
        for (int i = 1; i <= pretrain_steps; ++i) {
            sample(B);
            float l = L->train(batch);
            if (i % target_sync == 0) L->sync_target();
            if (i % 5000 == 0) { std::printf("   pretrain %5d loss %.4f\n", i, l); std::fflush(stdout); }
        }
        L->lr = lr_rl;
    }
    L->sync_target();
    evaluate("start");

    L->get_params(P); net.set(P);
    int ver = 0;
    publish(net, ver++, dir);
    for (int i = 0; i < n_actors; ++i)
        spawn_self(exe, {"actor", rom, std::to_string(i), std::to_string(n_actors), levels_csv, demos});

    long updates = 0, steps_in = 0, eps_in = 0;
    int actor_wins = 0;
    double loss_acc = 0; int loss_n = 0;
    auto t_eval = Clock::now(), t_log = Clock::now();
    while (secs_since(t0) < minutes * 60) {
        std::vector<fs::path> files;
        for (auto& f : fs::directory_iterator(dir + "/ep"))
            if (f.path().extension() == ".ep") files.push_back(f.path());
        for (auto& f : files) {
            Episode e;
            if (read_episode(f.string(), e) && e.S == S) { rep.add_episode(e); steps_in += e.T; ++eps_in; actor_wins += e.won; }
            std::error_code ec; fs::remove(f, ec);
        }
        int did = 0;
        while (rep.size >= 2000 && (double)updates * B < ratio * steps_in && did < 500) {
            sample((int)(B * demo_frac));
            loss_acc += L->train(batch); ++loss_n;
            ++updates; ++did;
            if (updates % target_sync == 0) L->sync_target();
            if (updates % publish_every == 0) { L->get_params(P); net.set(P); publish(net, ver++, dir); }
        }
        if (!did) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (secs_since(t_log) > 30) {
            double el = secs_since(t0);
            std::printf("[%7.1fs] steps %ld (%.0f/s) eps %ld actor_wins %d | updates %ld loss %.4f replay %zu\n",
                        el, steps_in, steps_in / el, eps_in, actor_wins, updates, loss_n ? loss_acc / loss_n : 0.0, rep.size);
            std::fflush(stdout);
            loss_acc = 0; loss_n = 0; t_log = Clock::now();
        }
        if (secs_since(t_eval) > eval_period && updates > 0) {
            bool all = evaluate("eval");
            t_eval = Clock::now();
            if (all && std::getenv("SMB_STOP_ON_ALL")) break;
        }
    }
    if (FILE* f = std::fopen((dir + "/stop").c_str(), "w")) std::fclose(f);
    evaluate("final");
    std::printf("done (%.1f min, %ld env steps, %ld updates); best net smb_best.bin, last smb_latest.bin\n",
                secs_since(t0) / 60, steps_in, updates);
    return 0;
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "searchdemo" && argc >= 5) {
        Level lv;
        if (!smbx::parse_level(argv[3], lv)) return 1;
        return searchdemo(argv[2], lv, argv[4]);
    }
    if (mode == "eval" && argc >= 5) return evalnet(argv[2], argv[3], parse_levels(argv[4]));
    if (mode == "recorddemo" && argc >= 6) {
        Level lv;
        if (!smbx::parse_level(argv[3], lv)) return 1;
        return record(argv[2], argv[4], lv, argv[5], true);
    }
    if (mode == "record" && argc >= 6) {
        Level lv;
        if (!smbx::parse_level(argv[4], lv)) return 1;
        return record(argv[2], argv[3], lv, argv[5], false);
    }
    if (mode == "parity") return parity_conv();
    if (mode == "actor" && argc >= 7) return actor(argv[2], std::atoi(argv[3]), std::atoi(argv[4]), argv[5], argv[6]);
    if (mode == "train" && argc >= 3) {
        std::string levels = "1-1", held, demos = "demos", warm;
        int actors = 16; double minutes = 60;
        for (int i = 3; i + 1 < argc; i += 2) {
            std::string k = argv[i], v = argv[i + 1];
            if (k == "--levels") levels = v; else if (k == "--eval") held = v; else if (k == "--demos") demos = v;
            else if (k == "--actors") actors = std::atoi(v.c_str()); else if (k == "--minutes") minutes = std::atof(v.c_str());
            else if (k == "--warm") warm = v; else std::printf("unknown option %s\n", k.c_str());
        }
        return train(argv[0], argv[2], parse_levels(levels), parse_levels(held), levels, demos, actors, minutes, warm.c_str());
    }
    std::printf("usage: smb_train searchdemo|train|eval|record ... (see header)\n");
    return 1;
}
