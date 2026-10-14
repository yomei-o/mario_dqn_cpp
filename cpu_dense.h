#pragma once
// Plain-loop dense layer forward / backward shared by the CPU learners (MLP and
// conv): the reference math the CUDA kernels are checked against.
#include <cstring>
#include <cstddef>

namespace rl {

// Y[B,O] = X[B,I] W[I,O] + b (optionally ReLU)
inline void dense(const float* X, const float* W, const float* b, int B, int I, int O, bool relu, float* Y) {
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
inline void dense_back(const float* X, const float* W, const float* dY, int B, int I, int O,
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

}  // namespace rl
