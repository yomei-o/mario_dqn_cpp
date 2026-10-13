// CPU backend of the batched DQfD learner. Plain loops, same math and the same
// parameter layout as the CUDA backend -- it is the reference the GPU path is
// checked against (`mario_gpu parity`) and the fallback when there is no GPU.
#include "dqn_learner.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace rl {

namespace {

// Y[B,O] = X[B,I] W[I,O] + b (optionally ReLU)
void dense(const float* X, const float* W, const float* b, int B, int I, int O, bool relu, float* Y) {
    for (int r = 0; r < B; ++r) {
        float* y = Y + (size_t)r * O;
        for (int o = 0; o < O; ++o) y[o] = b[o];
        const float* x = X + (size_t)r * I;
        for (int i = 0; i < I; ++i) {
            float xv = x[i];
            if (xv == 0.f) continue;
            const float* w = W + (size_t)i * O;
            for (int o = 0; o < O; ++o) y[o] += xv * w[o];
        }
        if (relu) for (int o = 0; o < O; ++o) y[o] = y[o] > 0.f ? y[o] : 0.f;
    }
}

// Given dY[B,O]: dW[I,O] = X^T dY, db = colsum dY, dX[B,I] = dY W^T (if dX).
void dense_back(const float* X, const float* W, const float* dY, int B, int I, int O,
                float* dW, float* db, float* dX) {
    std::memset(dW, 0, sizeof(float) * (size_t)I * O);
    std::memset(db, 0, sizeof(float) * O);
    for (int r = 0; r < B; ++r) {
        const float* x = X + (size_t)r * I;
        const float* dy = dY + (size_t)r * O;
        for (int o = 0; o < O; ++o) db[o] += dy[o];
        for (int i = 0; i < I; ++i) {
            float xv = x[i];
            if (xv == 0.f) continue;
            float* dw = dW + (size_t)i * O;
            for (int o = 0; o < O; ++o) dw[o] += xv * dy[o];
        }
        if (dX) {
            float* dx = dX + (size_t)r * I;
            for (int i = 0; i < I; ++i) {
                const float* w = W + (size_t)i * O;
                float acc = 0.f;
                for (int o = 0; o < O; ++o) acc += dy[o] * w[o];
                dx[i] = acc;
            }
        }
    }
}

class CpuLearner : public Learner {
public:
    std::vector<float> P, T, G, M, V;   // online, target, grad, Adam m/v (flat, QNet order)
    size_t oW1, ob1, oW2, ob2, oW3, ob3;
    long t = 0;
    std::vector<float> h1, h2, q, h1n, h2n, qno, qnt, dq, dh2, dh1;

    CpuLearner(int s, int a, int h) : Learner(s, a, h) {
        oW1 = 0; ob1 = oW1 + (size_t)S * H; oW2 = ob1 + H; ob2 = oW2 + (size_t)H * H;
        oW3 = ob2 + H; ob3 = oW3 + (size_t)H * A;
        size_t n = num_params();
        P.assign(n, 0.f); T.assign(n, 0.f); G.assign(n, 0.f); M.assign(n, 0.f); V.assign(n, 0.f);
    }
    const char* name() const override { return "cpu"; }

    void set_params(const std::vector<std::vector<float>>& p) override {
        size_t off[6] = {oW1, ob1, oW2, ob2, oW3, ob3};
        for (int i = 0; i < 6; ++i) std::copy(p[i].begin(), p[i].end(), P.begin() + off[i]);
        T = P;
    }
    void get_params(std::vector<std::vector<float>>& p) override {
        size_t off[7] = {oW1, ob1, oW2, ob2, oW3, ob3, P.size()};
        p.resize(6);
        for (int i = 0; i < 6; ++i) p[i].assign(P.begin() + off[i], P.begin() + off[i + 1]);
    }
    void sync_target() override { T = P; }

    void fwd(const std::vector<float>& W, const float* x, int B,
             std::vector<float>& a1, std::vector<float>& a2, std::vector<float>& out) {
        a1.resize((size_t)B * H); a2.resize((size_t)B * H); out.resize((size_t)B * A);
        dense(x, &W[oW1], &W[ob1], B, S, H, true, a1.data());
        dense(a1.data(), &W[oW2], &W[ob2], B, H, H, true, a2.data());
        dense(a2.data(), &W[oW3], &W[ob3], B, H, A, false, out.data());
    }

    void forward(const float* x, int B, float* qout) override {
        std::vector<float> a1, a2, out;
        fwd(P, x, B, a1, a2, out);
        std::copy(out.begin(), out.end(), qout);
    }

    float train(const Batch& b) override {
        const int B = b.B;
        fwd(P, b.ns.data(), B, h1n, h2n, qno);
        fwd(T, b.ns.data(), B, h1n, h2n, qnt);
        fwd(P, b.s.data(), B, h1, h2, q);
        dq.assign((size_t)B * A, 0.f);
        double loss = 0;
        for (int r = 0; r < B; ++r)
            loss += dqfd_row(&q[(size_t)r * A], &qno[(size_t)r * A], &qnt[(size_t)r * A],
                             b.a[r], b.ret[r], b.gn[r], b.demo[r], A, B, margin, lambda_demo,
                             &dq[(size_t)r * A]);
        dh2.resize((size_t)B * H); dh1.resize((size_t)B * H);
        dense_back(h2.data(), &P[oW3], dq.data(), B, H, A, &G[oW3], &G[ob3], dh2.data());
        for (size_t i = 0; i < dh2.size(); ++i) if (h2[i] <= 0.f) dh2[i] = 0.f;
        dense_back(h1.data(), &P[oW2], dh2.data(), B, H, H, &G[oW2], &G[ob2], dh1.data());
        for (size_t i = 0; i < dh1.size(); ++i) if (h1[i] <= 0.f) dh1[i] = 0.f;
        dense_back(b.s.data(), &P[oW1], dh1.data(), B, S, H, &G[oW1], &G[ob1], nullptr);

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

std::unique_ptr<Learner> make_cpu_learner(int S, int A, int H) {
    return std::unique_ptr<Learner>(new CpuLearner(S, A, H));
}

#ifndef MARIO_CUDA
std::unique_ptr<Learner> make_cuda_learner(int, int, int) { return nullptr; }
#endif

}  // namespace rl
