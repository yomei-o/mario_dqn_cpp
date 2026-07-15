#pragma once
// Q-network: a small MLP on the self-made autograd, mapping a state to one
// Q-value per action. Includes an Adam optimizer (DQN is much more stable with
// Adam than plain SGD) and a target-network copy.
#include "autograd.h"
#include <vector>
#include <cmath>

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
