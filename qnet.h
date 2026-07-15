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
