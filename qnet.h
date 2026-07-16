#pragma once
// Q-network: a small MLP on the self-made autograd, mapping a state to one
// Q-value per action. Includes an Adam optimizer (DQN is much more stable with
// Adam than plain SGD) and a target-network copy.
#include "autograd.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <string>

namespace rl {

class QNet {
public:
    int sdim, hdim, adim;
    ag::Tensor W1, b1, W2, b2, W3, b3;

    QNet(int state_dim, int n_actions, int hidden = 128)
        : sdim(state_dim), hdim(hidden), adim(n_actions) {
        auto he = [](int fan_in) { return std::sqrt(2.0f / fan_in); };
        W1 = ag::Tensor::randn({sdim, hdim}, he(sdim), true);
        b1 = ag::Tensor::zeros({hdim}, true);
        W2 = ag::Tensor::randn({hdim, hdim}, he(hdim), true);
        b2 = ag::Tensor::zeros({hdim}, true);
        W3 = ag::Tensor::randn({hdim, adim}, he(hdim), true);
        b3 = ag::Tensor::zeros({adim}, true);
    }

    // x:(B, sdim) -> Q:(B, adim)
    ag::Tensor forward(const ag::Tensor& x) {
        auto h = ag::relu(ag::add_bias_2d(ag::matmul(x, W1), b1));
        h = ag::relu(ag::add_bias_2d(ag::matmul(h, W2), b2));
        return ag::add_bias_2d(ag::matmul(h, W3), b3);
    }

    std::vector<ag::Tensor> params() { return {W1, b1, W2, b2, W3, b3}; }

    void copy_from(QNet& o) {
        auto p = params(), q = o.params();
        for (size_t i = 0; i < p.size(); ++i) p[i].data() = q[i].data();
    }

    bool save(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        for (auto& t : params()) {
            int n = t.numel(); std::fwrite(&n, sizeof(int), 1, f);
            std::fwrite(t.data().data(), sizeof(float), n, f);
        }
        std::fclose(f);
        return true;
    }
    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        for (auto& t : params()) {
            int n = 0;
            if (std::fread(&n, sizeof(int), 1, f) != 1 || n != t.numel()) { std::fclose(f); return false; }
            std::fread(t.data().data(), sizeof(float), n, f);
        }
        std::fclose(f);
        return true;
    }

    // Warm-start from a checkpoint with a SMALLER input dimension: load the old
    // input-to-hidden weights (W1) into the first rows and ZERO the new feature's
    // row, so the initial policy is identical to the old net while the new input
    // starts contributing from zero. All other layers must match exactly.
    // Lets us append observation features without discarding a trained policy.
    bool load_expand(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        auto ps = params();
        for (size_t i = 0; i < ps.size(); ++i) {
            int n = 0;
            if (std::fread(&n, sizeof(int), 1, f) != 1) { std::fclose(f); return false; }
            auto& d = ps[i].data();
            if (i == 0) {  // W1: [sdim, hdim] row-major; allow fewer input rows
                if (n > ps[i].numel() || n % hdim != 0) { std::fclose(f); return false; }
                if ((int)std::fread(d.data(), sizeof(float), n, f) != n) { std::fclose(f); return false; }
                for (size_t j = n; j < d.size(); ++j) d[j] = 0.f;   // new feature -> no-op initially
            } else {
                if (n != ps[i].numel()) { std::fclose(f); return false; }
                if ((int)std::fread(d.data(), sizeof(float), n, f) != n) { std::fclose(f); return false; }
            }
        }
        std::fclose(f);
        return true;
    }

    // Net2Wider warm-start: load a checkpoint with a SMALLER input dim AND SMALLER
    // hidden width into this (wider) net, PRESERVING the function (so greedy == the
    // old policy at init) while adding capacity that trains up. New hidden units get
    // random INCOMING weights (from the constructor's init, so they activate and get
    // gradient) but ZERO OUTGOING weights (so they contribute nothing to the output
    // initially -> function unchanged). New input rows feeding OLD hidden units are
    // zeroed (the old net didn't see them). Assumes this net was just constructed
    // (non-overwritten weights keep their random init). adim must match.
    bool load_widen(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        auto rd = [&](std::vector<float>& v) -> bool {
            int n = 0; if (std::fread(&n, sizeof(int), 1, f) != 1) return false;
            v.resize(n); return (int)std::fread(v.data(), sizeof(float), n, f) == n;
        };
        std::vector<float> oW1, ob1, oW2, ob2, oW3, ob3;
        bool ok = rd(oW1) && rd(ob1) && rd(oW2) && rd(ob2) && rd(oW3) && rd(ob3);
        std::fclose(f);
        if (!ok) return false;
        int oh = (int)ob1.size();
        if (oh <= 0 || (int)oW1.size() % oh != 0) return false;
        int os = (int)oW1.size() / oh;                 // old sdim
        if (os > sdim || oh > hdim || (int)ob3.size() != adim) return false;
        auto& w1 = W1.data(); auto& B1 = b1.data();
        auto& w2 = W2.data(); auto& B2 = b2.data();
        auto& w3 = W3.data(); auto& B3 = b3.data();
        // W1 [sdim x hdim]: old block; new-input rows over OLD-hidden cols -> 0.
        for (int i = 0; i < os; ++i) for (int j = 0; j < oh; ++j) w1[i * hdim + j] = oW1[i * oh + j];
        for (int i = os; i < sdim; ++i) for (int j = 0; j < oh; ++j) w1[i * hdim + j] = 0.f;
        for (int j = 0; j < oh; ++j) B1[j] = ob1[j];
        // W2 [hdim x hdim]: old block; NEW-hidden1 rows over OLD-hidden2 cols -> 0.
        for (int i = 0; i < oh; ++i) for (int j = 0; j < oh; ++j) w2[i * hdim + j] = oW2[i * oh + j];
        for (int i = oh; i < hdim; ++i) for (int j = 0; j < oh; ++j) w2[i * hdim + j] = 0.f;
        for (int j = 0; j < oh; ++j) B2[j] = ob2[j];
        // W3 [hdim x adim]: old block; NEW-hidden2 rows -> 0 (no output contribution).
        for (int i = 0; i < oh; ++i) for (int a = 0; a < adim; ++a) w3[i * adim + a] = oW3[i * adim + a];
        for (int i = oh; i < hdim; ++i) for (int a = 0; a < adim; ++a) w3[i * adim + a] = 0.f;
        for (int a = 0; a < adim; ++a) B3[a] = ob3[a];
        return true;
    }
};

// Adam optimizer over a fixed parameter list.
class Adam {
public:
    std::vector<ag::Tensor> ps;
    std::vector<std::vector<float>> m, v;
    float lr, b1, b2, eps;
    long t = 0;

    Adam(std::vector<ag::Tensor> params, float lr = 1e-3f,
         float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
        : ps(std::move(params)), lr(lr), b1(beta1), b2(beta2), eps(eps) {
        for (auto& p : ps) { m.emplace_back(p.numel(), 0.f); v.emplace_back(p.numel(), 0.f); }
    }

    void zero_grad() { for (auto& p : ps) p.zero_grad(); }

    void step() {
        ++t;
        float bc1 = 1.f - std::pow(b1, (float)t), bc2 = 1.f - std::pow(b2, (float)t);
        for (size_t i = 0; i < ps.size(); ++i) {
            auto& pd = ps[i].data(); auto& pg = ps[i].grad();
            for (size_t j = 0; j < pd.size(); ++j) {
                m[i][j] = b1 * m[i][j] + (1 - b1) * pg[j];
                v[i][j] = b2 * v[i][j] + (1 - b2) * pg[j] * pg[j];
                float mh = m[i][j] / bc1, vh = v[i][j] / bc2;
                pd[j] -= lr * mh / (std::sqrt(vh) + eps);
            }
        }
    }
};

}  // namespace rl
