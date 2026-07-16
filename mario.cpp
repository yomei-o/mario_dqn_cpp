#include "mario.h"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace mario {

// --- Super Mario Bros RAM map (well-documented addresses) -------------------
static inline int RAM(uint16_t a) { return nes::ram(a); }
static int level_x() { return RAM(0x006D) * 256 + RAM(0x0086); }   // x in level
static int player_y() { return RAM(0x00CE); }                       // y on screen
static int8_t x_speed() { return (int8_t)RAM(0x0057); }             // signed
static int player_state() { return RAM(0x000E); }                   // 0x08 normal, 0x06/0x0B dead/dying
static int float_state() { return RAM(0x001D); }                    // 0 ground, 1 air, 3 climbing
static int lives() { return RAM(0x075A); }
static int vert_page() { return RAM(0x00B5); }                      // >1 => fell into a pit
static int power_state() { return RAM(0x0756); }                    // 0 small, 1 super, 2 fire
static int coin_count() { return RAM(0x075E); }                     // BCD coin counter
static int score_val() {                                            // 6 score digits ($07DD..$07E2)
    int s = 0;
    for (int i = 0; i < 6; ++i) s = s * 10 + RAM(0x07DD + i);
    return s;
}
// Enemy i level-x (page-aware): x-page at 0x006E+i, x-within-page at 0x0087+i.
static int enemy_level_x(int i) { return RAM(0x006E + i) * 256 + RAM(0x0087 + i); }

// Read one background metatile from SMB's 2-screen tile buffer ($0500-$069F).
// x = absolute level pixel x, y = on-screen pixel y. Returns 0 for sky/empty
// (passable) and nonzero for solid ground/blocks/pipes. Formula is the standard
// SMB tilemap indexing (13 rows of 16 cols per screen, two screens buffered).
static int tile_at(int x_px, int y_px) {
    if (y_px < 32) return 0;                 // top HUD rows / sky above playfield
    int row = (y_px - 32) / 16;
    if (row < 0 || row > 12) return 0;       // below the playfield
    int col = x_px / 16;
    int page = (col / 16) % 2;
    int colp = col % 16;
    return RAM(0x0500 + page * 0xD0 + row * 16 + colp);
}

// Button mask for each discrete action.
static uint8_t action_buttons(int a) {
    using namespace nes;
    switch (a) {
        case A_RIGHT:    return RIGHT;
        case A_RIGHT_A:  return RIGHT | A;
        case A_RIGHT_B:  return RIGHT | B;
        case A_RIGHT_AB: return RIGHT | A | B;
        case A_A:        return A;
    }
    return 0;
}

bool Env::init(const char* rom_path) {
    FILE* f = std::fopen(rom_path, "rb");
    if (!f) { std::printf("cannot open ROM: %s\n", rom_path); return false; }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    rom_.resize(n);
    if (std::fread(rom_.data(), 1, n, f) != (size_t)n) { std::fclose(f); return false; }
    std::fclose(f);
    obs_.assign(STATE_DIM, 0.f);
    return true;
}

bool Env::is_dead() const {
    int st = player_state();
    return st == 0x06 || st == 0x0B || vert_page() > 1 || lives() < start_lives_;
}

bool Env::is_win() const {
    // 1-1 flagpole/level-end: player enters "end of level" states, or x passes ~3160.
    return float_state() == 0x03 || level_x() > 3160;
}

// Re-power the console and drive the title screen until Mario is in play.
void Env::boot() {
    nes::load_bytes(rom_.data(), (int)rom_.size());   // re-power the console
    auto run = [&](int frames, uint8_t btn) {
        nes::set_buttons(0, btn);
        for (int i = 0; i < frames; ++i) nes::step_frame();
    };
    run(40, 0);                    // boot to title
    run(8, nes::START);            // press START
    run(4, 0);
    // advance until Mario is actually in normal play (level loaded)
    nes::set_buttons(0, 0);
    for (int i = 0; i < 300; ++i) {
        nes::step_frame();
        if (player_state() == 0x08 && level_x() > 0) break;
    }
}

// Raw advance: hold `a`'s buttons for FRAME_SKIP frames. No reward/done/obs
// bookkeeping -- used to fast-forward through a demonstration prefix.
void Env::apply_action_frames(int a) {
    nes::set_buttons(0, action_buttons(a));
    for (int i = 0; i < FRAME_SKIP; ++i) nes::step_frame();
}

const std::vector<float>& Env::reset() { return reset_from(0); }

const std::vector<float>& Env::reset_from(int start_idx) {
    if (start_idx == 0) {
        // Fast level-start reset: restore a cached post-boot snapshot instead of
        // re-running ~350 title-screen frames on every episode. boot() dominated
        // the from-start episode time, so this roughly doubles trials/sec. The
        // emulator is deterministic, so the cached start state == a fresh boot.
        if (have_start_state_) nes::load_state(start_state_);
        else { boot(); nes::save_state(start_state_); have_start_state_ = true; }
    } else {
        boot();
        int n = std::min(start_idx, (int)demo_.size());
        for (int i = 0; i < n; ++i) apply_action_frames(demo_[i]);   // fast-forward to checkpoint
    }
    start_lives_ = lives();
    prev_x_ = level_x();
    max_x_ = prev_x_;
    steps_ = 0;
    stall_ = 0;
    prev_score_ = score_val();
    prev_power_ = power_state();
    won_ = false;
    build_obs();
    return obs_;
}

bool Env::load_demo(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    uint32_t n = 0;
    if (std::fread(&n, 4, 1, f) != 1) { std::fclose(f); return false; }
    demo_.resize(n);
    size_t got = n ? std::fread(demo_.data(), 1, n, f) : 0;
    std::fclose(f);
    return got == n;
}

void Env::build_curriculum(int k_points) {
    ckpt_states_.clear();
    ckpt_x_.clear();
    ckpt_target_.clear();
    int L = (int)demo_.size();
    if (L < 40 || k_points < 1) return;
    // Spread checkpoints across the WHOLE demo (not just the back half). When the
    // demo only reached the frontier, back-half focus made sense; now the demo is
    // a full-level clear, so covering early+mid+late gives the agent dense practice
    // in EVERY region. That is what lets a from-start greedy run chain end-to-end:
    // mid-level (where from-start episodes die before reaching, and old back-half
    // checkpoints started after) was the un-practiced gap that broke chaining.
    int lo = std::max(1, L / 12), hi = std::max(1, L - 8);   // ~8%..~99%: full level, short of the fatal tail
    // Target action counts (after how many demo actions to snapshot), ascending.
    std::vector<int> targets;
    for (int j = 0; j < k_points; ++j) {
        int idx = (k_points == 1) ? hi : lo + (hi - lo) * j / (k_points - 1);
        if (targets.empty() || idx > targets.back()) targets.push_back(idx);
    }
    // One forward pass: replay the demo and snapshot when each target is reached.
    boot();
    size_t t = 0;
    for (int i = 0; i < hi && t < targets.size(); ++i) {
        apply_action_frames(demo_[i]);
        if (i + 1 == targets[t]) {
            std::vector<uint8_t> st;
            nes::save_state(st);
            ckpt_states_.push_back(std::move(st));
            ckpt_x_.push_back(level_x());
            ckpt_target_.push_back(targets[t]);   // demo actions replayed to reach here (== i+1)
            ++t;
        }
    }
}

const std::vector<float>& Env::reset_to_checkpoint(int k) {
    nes::load_state(ckpt_states_[k]);          // instant restore (no replay)
    start_lives_ = lives();
    prev_x_ = level_x();
    max_x_ = prev_x_;
    steps_ = 0;
    stall_ = 0;
    prev_score_ = score_val();
    prev_power_ = power_state();
    won_ = false;
    build_obs();
    return obs_;
}

float Env::step(int action, bool& done) {
    // Snapshot enemy state BEFORE advancing, so we can detect a stomp this step:
    // an enemy that was active and just ahead of / below Mario, gone after the
    // frames with a score gain, while Mario was above it = defeated by a stomp
    // (vs merely scrolling off-screen behind, which we exclude by the x window).
    int pa[5], prx[5], pey[5], ppy = player_y(), psc = score_val();
    for (int i = 0; i < 5; ++i) {
        pa[i]  = RAM(0x000F + i) != 0;
        prx[i] = enemy_level_x(i) - level_x();
        pey[i] = RAM(0x00CF + i);
    }

    uint8_t btn = action_buttons(action);
    nes::set_buttons(0, btn);
    for (int i = 0; i < FRAME_SKIP; ++i) nes::step_frame();
    ++steps_;

    // Stomp = Mario landed on an enemy: score went up this step, and an enemy that
    // was active and just ahead sat at/below Mario (he was poised above it). We do
    // NOT require the enemy to have vanished -- a stomped Goomba lingers a few
    // frames in a "flattened" state, so requiring instant removal missed real
    // stomps. (A coin grabbed while directly above an enemy is a rare false hit and
    // still a fine "good" demo, so acceptable.)
    stomped_ = false;
    if (score_val() > psc)
        for (int i = 0; i < 5; ++i)
            if (pa[i] && prx[i] >= -24 && prx[i] <= 56 && ppy <= pey[i] + 4) { stomped_ = true; break; }

    int x = level_x();
    float dx = (float)(x - prev_x_);
    prev_x_ = x;

    // Stagnation tracking: reset the stall counter whenever Mario reaches new
    // ground, otherwise let it grow. This lets us both penalize dithering and
    // cut the episode short when the agent is stuck against a pipe/pit.
    if (x > max_x_) { max_x_ = x; stall_ = 0; } else { ++stall_; }

    // --- PROGRESS-FIRST + STOMP-only reward ---------------------------------
    // The user's insight: rewarding enemy STOMPS (not mushrooms/coins) is the clean
    // signal. A stomp is ON the forward line (jump onto the Goomba while advancing)
    // so it doesn't fight progress the way grabbing a mushroom does (that needs a
    // detour: hit the block from below, wait for it to emerge, chase it) -- and its
    // reward is IMMEDIATE (jump -> land -> score this step), so credit assignment is
    // easy, unlike the delayed multi-step mushroom/block payoff. So: strong progress
    // + an immediate stomp bonus, and NO item (mushroom/coin) reward at all.
    // Progress reward, ANTI-RUSH (user's insight: the progress reward + time cost
    // + discount lure the agent into sprinting headlong into hazards and dying).
    // Reward forward progress only up to a WALKING pace (cap dx at 8): sprinting
    // faster earns no extra, so there's no incentive to recklessly barrel forward.
    // Time penalty softened (0.1 -> 0.04) so it isn't rushed. Death (-50) still
    // makes caution pay.
    float dense = std::max(-12.f, std::min(12.f, std::min(dx, 8.f) - 0.04f));
    float r = dense;
    if (stomped_) r += 12.f;                    // defeated an enemy -> clears the path (risky for small Mario)
    else {
        // Jump-over / evade (user's idea): an enemy that was just ahead is now
        // BEHIND, still alive, while Mario is airborne -> he safely cleared it.
        // Rewarded MORE than a stomp: for small Mario, jumping over is easier and
        // safer (a mis-stomp is death), so we steer toward the safe evade. Outcome
        // -based + tied to forward progress (enemy must go ahead->behind) -> not
        // farmable.
        if (float_state() == 0x01)
            for (int i = 0; i < 5; ++i)
                if (pa[i] && RAM(0x000F + i) != 0 && prx[i] >= 0 && prx[i] <= 48 &&
                    (enemy_level_x(i) - x) < 0) { r += 25.f; break; }
    }
    prev_score_ = score_val();                  // kept for logging (not rewarded)
    prev_power_ = power_state();                 // kept for the obs power feature (not rewarded)

    done = false;
    if (is_dead()) { r = dense - 50.f; done = true; }           // pit/enemy death: clearly bad
    else if (is_win()) { r += 100.f; done = true; won_ = true; } // flagpole: strong, dominant reward
    else if (stall_ >= STALL_LIMIT) { r = dense - 10.f; done = true; }  // stuck: give up this life
    else if (steps_ >= MAX_STEPS) { done = true; }

    build_obs();
    return r;
}

int Env::mario_x() const { return level_x(); }
int Env::score() const { return score_val(); }
int Env::coins() const { return coin_count(); }
int Env::power() const { return power_state(); }

void Env::build_obs() {
    obs_.assign(STATE_DIM, 0.f);
    int px = RAM(0x0086), py = player_y();
    int lx = level_x();
    obs_[0] = px / 255.f;
    obs_[1] = py / 255.f;
    obs_[2] = std::max(-1.f, std::min(1.f, x_speed() / 48.f));
    obs_[3] = float_state() / 3.f;
    // 5 enemy slots: active flag + relative position to Mario. Relative x uses
    // page-aware LEVEL coordinates so an enemy ahead reads positive (a naive
    // low-byte subtraction flips sign across a 256px page boundary).
    for (int i = 0; i < 5; ++i) {
        int active = RAM(0x000F + i) != 0;
        int rel_x = enemy_level_x(i) - lx;
        int ey = RAM(0x00CF + i);
        obs_[4 + i * 3 + 0] = (float)active;
        obs_[4 + i * 3 + 1] = active ? std::max(-1.f, std::min(1.f, rel_x / 128.f)) : 0.f;
        obs_[4 + i * 3 + 2] = active ? std::max(-1.f, std::min(1.f, (ey - py) / 128.f)) : 0.f;
    }
    // Terrain occupancy grid: TILE_COLS columns starting at Mario's and going
    // right, sampled at FIXED screen rows TILE_ROW0..TILE_ROW0+TILE_ROWS-1.
    // 1 = solid, 0 = passable. Reading a column top->bottom gives the vertical
    // profile ahead: flat ground = "00000 11", a pit = all zeros (jump the gap),
    // a pipe = solid cells stacking up from the ground (jump the obstacle). Using
    // fixed rows (vs Mario-relative) keeps the ground/landing visible mid-jump.
    // Two grids over the same lookahead window (TILE_COLS x TILE_ROWS ahead of
    // Mario, fixed screen rows): (1) occupancy -- 1 = solid; (2) ? -blocks -- 1 =
    // an active question/coin block (SMB overworld metatiles 0xC0..0xC3; become
    // 0x54 once hit). The ? -block grid lets the agent SEE which blocks above it
    // are worth jumping into for coins / a mushroom (phase ii: items/score).
    int base = 19;
    int qbase = 19 + TILE_N + 1;                      // ? -block grid, right after the power feature
    for (int c = 0; c < TILE_COLS; ++c) {
        int tx = lx + c * 16 + 8;                     // center of column c ahead
        for (int r = 0; r < TILE_ROWS; ++r) {
            int ty = 32 + (TILE_ROW0 + r) * 16 + 8;   // center of fixed screen row
            int v = tile_at(tx, ty);
            obs_[base + c * TILE_ROWS + r]  = v != 0 ? 1.f : 0.f;
            obs_[qbase + c * TILE_ROWS + r] = (v >= 0xC0 && v <= 0xC3) ? 1.f : 0.f;
        }
    }
    // Power state (0 small / 0.5 super / 1 fire): lets the agent behave more
    // boldly when big (a hit costs power, not a life) and value keeping it.
    obs_[19 + TILE_N] = std::min(1.f, power_state() / 2.f);
}

}  // namespace mario
