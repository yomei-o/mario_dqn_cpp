// CPU side of the conv Q-network: inference (ConvQ, used by actors and
// evaluation) and the CPU training backend -- the reference the CUDA backend
// (conv_learner_cuda.cu) is checked against with `smb_train parity`.
#include "conv_net.h"
#include "cpu_dense.h"
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>

namespace rl {

using CA = ConvArch;

// Shared forward pass; keeps the intermediates the backward pass needs.
struct ConvActs {
    std::vector<float> col1, a1, col2, a2, z, h, q;
};

static void conv_forward(const ConvArch& ar, const float* P, const float* x, int B, ConvActs& t) {
    const int R1 = B * CA::H1 * CA::W1, R2 = B * CA::H2 * CA::W2;
    t.col1.resize((size_t)R1 * CA::K1);
    for (int b = 0; b < B; ++b)
        for (int oh = 0; oh < CA::H1; ++oh)
            for (int ow = 0; ow < CA::W1; ++ow) {
                float* row = &t.col1[(((size_t)b * CA::H1 + oh) * CA::W1 + ow) * CA::K1];
                for (int k = 0; k < CA::K1; ++k)
                    row[k] = im2col_at(x, b, oh, ow, k, true, CA::C0, CA::H0, CA::W0, 1, CA::IN);
            }
    t.a1.resize((size_t)R1 * CA::C1);
    dense(t.col1.data(), P + ar.oWc1, P + ar.obc1, R1, CA::K1, CA::C1, true, t.a1.data());
    t.col2.resize((size_t)R2 * CA::K2);
    for (int b = 0; b < B; ++b)
        for (int oh = 0; oh < CA::H2; ++oh)
            for (int ow = 0; ow < CA::W2; ++ow) {
                float* row = &t.col2[(((size_t)b * CA::H2 + oh) * CA::W2 + ow) * CA::K2];
                for (int k = 0; k < CA::K2; ++k)
                    row[k] = im2col_at(t.a1.data(), b, oh, ow, k, false, CA::C1, CA::H1, CA::W1, 2,
                                       (size_t)CA::H1 * CA::W1 * CA::C1);
            }
    t.a2.resize((size_t)R2 * CA::C2);
    dense(t.col2.data(), P + ar.oWc2, P + ar.obc2, R2, CA::K2, CA::C2, true, t.a2.data());
    const int FZ = CA::F + CA::NS;
    t.z.resize((size_t)B * FZ);
    for (int b = 0; b < B; ++b) {
        std::memcpy(&t.z[(size_t)b * FZ], &t.a2[(size_t)b * CA::F], sizeof(float) * CA::F);
        std::memcpy(&t.z[(size_t)b * FZ + CA::F], x + (size_t)b * CA::IN + (CA::IN - CA::NS), sizeof(float) * CA::NS);
    }
    t.h.resize((size_t)B * ar.H);
    dense(t.z.data(), P + ar.oW3, P + ar.ob3, B, FZ, ar.H, true, t.h.data());
    t.q.resize((size_t)B * ar.A);
    dense(t.h.data(), P + ar.oW4, P + ar.ob4, B, ar.H, ar.A, false, t.q.data());
}

ConvQ::ConvQ(int A, int H) : arch(A, H), P(arch.n, 0.f) {}

void ConvQ::init_random(unsigned seed) {
    std::mt19937 rng(seed);
    auto he = [&](size_t off, size_t count, int fan_in) {
        std::normal_distribution<float> nd(0.f, std::sqrt(2.f / fan_in));
        for (size_t i = 0; i < count; ++i) P[off + i] = nd(rng);
    };
    std::fill(P.begin(), P.end(), 0.f);
    he(arch.oWc1, (size_t)CA::K1 * CA::C1, CA::K1);
    he(arch.oWc2, (size_t)CA::K2 * CA::C2, CA::K2);
    he(arch.oW3, (size_t)(CA::F + CA::NS) * arch.H, CA::F + CA::NS);
    he(arch.oW4, (size_t)arch.H * arch.A, arch.H);
}

void ConvQ::forward(const float* x, int B, float* q) const {
    ConvActs t;
    conv_forward(arch, P.data(), x, B, t);
    std::copy(t.q.begin(), t.q.end(), q);
}

bool ConvQ::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    auto off = arch.offsets();
    for (size_t i = 0; i + 1 < off.size(); ++i) {
        int n = (int)(off[i + 1] - off[i]);
        std::fwrite(&n, sizeof(int), 1, f);
        std::fwrite(P.data() + off[i], sizeof(float), n, f);
    }
    std::fclose(f);
    return true;
}

bool ConvQ::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    auto off = arch.offsets();
    bool ok = true;
    for (size_t i = 0; i + 1 < off.size() && ok; ++i) {
        int n = 0;
        ok = std::fread(&n, sizeof(int), 1, f) == 1 && n == (int)(off[i + 1] - off[i]) &&
             std::fread(P.data() + off[i], sizeof(float), n, f) == (size_t)n;
    }
    std::fclose(f);
    return ok;
}

void ConvQ::get_params(std::vector<std::vector<float>>& p) const {
    auto off = arch.offsets();
    p.resize(off.size() - 1);
    for (size_t i = 0; i + 1 < off.size(); ++i) p[i].assign(P.begin() + off[i], P.begin() + off[i + 1]);
}

void ConvQ::set_params(const std::vector<std::vector<float>>& p) {
    auto off = arch.offsets();
    for (size_t i = 0; i + 1 < off.size(); ++i) std::copy(p[i].begin(), p[i].end(), P.begin() + off[i]);
}

bool is_conv_file(const std::string& path, int* hidden_out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    int n = 0;
    bool conv = std::fread(&n, sizeof(int), 1, f) == 1 && n == CA::K1 * CA::C1;
    if (conv && hidden_out) {   // skip to W3 to read H
        long skip = (long)sizeof(float) * n;
        int m = 0;
        std::fseek(f, skip, SEEK_CUR);
        for (int i = 0; i < 3 && conv; ++i) {   // bc1, Wc2, bc2
            conv = std::fread(&m, sizeof(int), 1, f) == 1;
            std::fseek(f, (long)sizeof(float) * m, SEEK_CUR);
        }
        if (conv && std::fread(&m, sizeof(int), 1, f) == 1) *hidden_out = m / (CA::F + CA::NS);
    }
    std::fclose(f);
    return conv;
}

namespace {

class ConvCpuLearner : public Learner {
public:
    ConvArch ar;
    std::vector<float> P, T, G, M, V;
    long t = 0;
    ConvActs on_s, on_ns, tg_ns;

    ConvCpuLearner(int a, int h) : Learner(ConvArch::IN, a, h), ar(a, h) {
        P.assign(ar.n, 0.f); T = P; G = P; M = P; V = P;
    }
    const char* name() const override { return "cpu-conv"; }
    void set_params(const std::vector<std::vector<float>>& p) override {
        auto off = ar.offsets();
        for (size_t i = 0; i + 1 < off.size(); ++i) std::copy(p[i].begin(), p[i].end(), P.begin() + off[i]);
        T = P;
    }
    void get_params(std::vector<std::vector<float>>& p) override {
        auto off = ar.offsets();
        p.resize(off.size() - 1);
        for (size_t i = 0; i + 1 < off.size(); ++i) p[i].assign(P.begin() + off[i], P.begin() + off[i + 1]);
    }
    void sync_target() override { T = P; }
    void forward(const float* x, int B, float* q) override {
        ConvActs tt;
        conv_forward(ar, P.data(), x, B, tt);
        std::copy(tt.q.begin(), tt.q.end(), q);
    }

    float train(const Batch& b) override {
        const int B = b.B, A_ = A;
        conv_forward(ar, P.data(), b.ns.data(), B, on_ns);
        conv_forward(ar, T.data(), b.ns.data(), B, tg_ns);
        conv_forward(ar, P.data(), b.s.data(), B, on_s);
        std::vector<float> dq((size_t)B * A_);
        double loss = 0;
        for (int r = 0; r < B; ++r)
            loss += dqfd_row(&on_s.q[(size_t)r * A_], &on_ns.q[(size_t)r * A_], &tg_ns.q[(size_t)r * A_],
                             b.a[r], b.ret[r], b.gn[r], b.demo[r], A_, B, margin, lambda_demo, &dq[(size_t)r * A_]);
        const int FZ = CA::F + CA::NS, R1 = B * CA::H1 * CA::W1, R2 = B * CA::H2 * CA::W2;
        std::vector<float> dh((size_t)B * H), dz((size_t)B * FZ), da2((size_t)R2 * CA::C2),
            dcol2((size_t)R2 * CA::K2), da1((size_t)R1 * CA::C1);
        dense_back(on_s.h.data(), &P[ar.oW4], dq.data(), B, H, A_, &G[ar.oW4], &G[ar.ob4], dh.data());
        for (size_t i = 0; i < dh.size(); ++i) if (on_s.h[i] <= 0.f) dh[i] = 0.f;
        dense_back(on_s.z.data(), &P[ar.oW3], dh.data(), B, FZ, H, &G[ar.oW3], &G[ar.ob3], dz.data());
        for (int r = 0; r < B; ++r)
            for (int k = 0; k < CA::F; ++k) {
                float v = dz[(size_t)r * FZ + k];
                da2[(size_t)r * CA::F + k] = on_s.a2[(size_t)r * CA::F + k] > 0.f ? v : 0.f;
            }
        dense_back(on_s.col2.data(), &P[ar.oWc2], da2.data(), R2, CA::K2, CA::C2, &G[ar.oWc2], &G[ar.obc2], dcol2.data());
        for (int bb = 0; bb < B; ++bb)
            for (int h = 0; h < CA::H1; ++h)
                for (int w = 0; w < CA::W1; ++w)
                    for (int c = 0; c < CA::C1; ++c) {
                        size_t i = (((size_t)bb * CA::H1 + h) * CA::W1 + w) * CA::C1 + c;
                        da1[i] = on_s.a1[i] > 0.f ? col2im_at(dcol2.data(), bb, h, w, c, CA::C1, CA::H2, CA::W2, 2) : 0.f;
                    }
        dense_back(on_s.col1.data(), &P[ar.oWc1], da1.data(), R1, CA::K1, CA::C1, &G[ar.oWc1], &G[ar.obc1], nullptr);

        double sq = 0;
        for (float g : G) sq += (double)g * g;
        float norm = (float)std::sqrt(sq);
        float scale = norm > clip ? clip / (norm + 1e-6f) : 1.f;
        ++t;
        float bc1 = 1.f - std::pow(beta1, (float)t), bc2 = 1.f - std::pow(beta2, (float)t);
        for (size_t j = 0; j < P.size(); ++j) {
            float g = G[j] * scale;
            M[j] = beta1 * M[j] + (1 - beta1) * g;
            V[j] = beta2 * V[j] + (1 - beta2) * g * g;
            P[j] -= lr * (M[j] / bc1) / (std::sqrt(V[j] / bc2) + adam_eps);
        }
        return (float)(loss / B);
    }
};

}  // namespace

std::unique_ptr<Learner> make_conv_cpu_learner(int A, int H) { return std::unique_ptr<Learner>(new ConvCpuLearner(A, H)); }

#ifndef MARIO_CUDA
std::unique_ptr<Learner> make_conv_cuda_learner(int, int) { return nullptr; }
#endif

}  // namespace rl
