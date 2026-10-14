#pragma once
// Convolutional Q-network for the egocentric board of smb_env (smbx::Env):
//
//   board 5ch x 13 x 16 --conv3x3 s1--> 32 x 13 x 16 --conv3x3 s2--> 64 x 7 x 8
//   [flatten 3584 | 8 scalars] --fc--> H --fc--> A
//
// The same 3x3 filters slide over the whole board, so "enemy just ahead",
// "ceiling overhead" or "pit below" is one learned pattern wherever it shows
// up, instead of a separate weight for every cell as in the MLP. Each conv is
// im2col + a dense matmul: positions are rows, (kh, kw, channel) are columns,
// and activations are kept NHWC so a conv output is directly the next im2col's
// input and the flattened fc input.
//
// Parameters (flat, in this order): Wc1[45 x 32] bc1[32] Wc2[288 x 64] bc2[64]
// W3[(3584 + 8) x H] b3[H] W4[H x A] b4[A] -- the same file format as QNet
// (int count + floats per tensor), told apart by the first tensor's size.
#include <vector>
#include <string>
#include <cstdio>
#include <cstddef>
#include "dqn_learner.h"

namespace rl {

struct ConvArch {
    static constexpr int C0 = 5, H0 = 13, W0 = 16, NS = 8;     // board channels / rows / cols, scalars
    static constexpr int C1 = 32, H1 = 13, W1 = 16;             // conv1: 3x3 stride 1 pad 1
    static constexpr int C2 = 64, H2 = 7, W2 = 8;               // conv2: 3x3 stride 2 pad 1
    static constexpr int K1 = 9 * C0, K2 = 9 * C1;
    static constexpr int F = H2 * W2 * C2;                      // 3584
    static constexpr int IN = C0 * H0 * W0 + NS;                // obs size (1048)
    int A, H;
    size_t oWc1, obc1, oWc2, obc2, oW3, ob3, oW4, ob4, n;
    ConvArch(int a, int h) : A(a), H(h) {
        oWc1 = 0; obc1 = oWc1 + (size_t)K1 * C1; oWc2 = obc1 + C1; obc2 = oWc2 + (size_t)K2 * C2;
        oW3 = obc2 + C2; ob3 = oW3 + (size_t)(F + NS) * H; oW4 = ob3 + H; ob4 = oW4 + (size_t)H * A; n = ob4 + A;
    }
    std::vector<size_t> offsets() const { return {oWc1, obc1, oWc2, obc2, oW3, ob3, oW4, ob4, n}; }
};

// im2col source indexing. conv1 reads the obs (channel-major [c][row][col]);
// conv2 reads conv1's NHWC output.
RL_HD inline float im2col_at(const float* src, int b, int oh, int ow, int k, bool from_obs,
                             int Cin, int Hin, int Win, int stride, size_t b_stride) {
    int c = k % Cin, kk = k / Cin, kw = kk % 3, kh = kk / 3;
    int h = oh * stride - 1 + kh, w = ow * stride - 1 + kw;
    if (h < 0 || h >= Hin || w < 0 || w >= Win) return 0.f;
    const float* base = src + (size_t)b * b_stride;
    return from_obs ? base[((size_t)c * Hin + h) * Win + w] : base[((size_t)h * Win + w) * Cin + c];
}

// Gradient wrt a conv2 input element (b, h, w, c) (NHWC) gathered from dCol.
RL_HD inline float col2im_at(const float* dcol, int b, int h, int w, int c,
                             int Cin, int Hout, int Wout, int stride) {
    float acc = 0.f;
    for (int kh = 0; kh < 3; ++kh) {
        int t = h + 1 - kh;
        if (t < 0 || t % stride) continue;
        int oh = t / stride;
        if (oh >= Hout) continue;
        for (int kw = 0; kw < 3; ++kw) {
            int u = w + 1 - kw;
            if (u < 0 || u % stride) continue;
            int ow = u / stride;
            if (ow >= Wout) continue;
            size_t row = ((size_t)b * Hout + oh) * Wout + ow;
            acc += dcol[row * (9 * Cin) + (kh * 3 + kw) * Cin + c];
        }
    }
    return acc;
}

std::unique_ptr<Learner> make_conv_cpu_learner(int A, int H);
std::unique_ptr<Learner> make_conv_cuda_learner(int A, int H);   // nullptr without CUDA / GPU

// CPU inference for actors / evaluation (and the reference forward).
class ConvQ {
public:
    ConvArch arch;
    std::vector<float> P;
    ConvQ(int A, int H);
    void init_random(unsigned seed);
    void forward(const float* x, int B, float* q) const;
    bool save(const std::string& path) const;
    bool load(const std::string& path);
    void get_params(std::vector<std::vector<float>>& p) const;
    void set_params(const std::vector<std::vector<float>>& p);
};

// Is this weights file a ConvQ (vs a QNet MLP)? Reads the first tensor's size.
bool is_conv_file(const std::string& path, int* hidden_out = nullptr);

}  // namespace rl
