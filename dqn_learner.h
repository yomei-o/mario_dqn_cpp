#pragma once
// Batched DQfD learner for the Q-net MLP (S -> H -> H -> A, ReLU), with two
// backends that run the SAME math: a plain-C++ CPU one (dqn_learner_cpu.cpp,
// always built) and a CUDA one (dqn_learner_cuda.cu, cuBLAS GEMMs + small
// kernels, built when CMake finds nvcc). The weights use QNet's layout and file
// format, so a net trained here loads in mario_dqn / record / the WASM demo.
//
// One train() call = one Adam step on a minibatch of n-step transitions:
//   TD:     huber( Q(s,a) - [ret + gn * Q_target(ns, argmax_a' Q(ns,a'))] )   (Double DQN)
//   demo:   lambda * ( max_a [Q(s,a) + margin*(a != a_E)] - Q(s,a_E) )        (DQfD large margin)
// `gn` carries gamma^n, or 0 when the episode terminated within the n steps.
// The demo term only applies to rows with demo=1. Gradients are clipped by
// global norm before the Adam step.
#include <vector>
#include <memory>

namespace rl {

struct Batch {
    int B = 0;
    std::vector<float> s, ns;     // B*S each
    std::vector<int> a;           // B
    std::vector<float> ret, gn;   // B
    std::vector<float> demo;      // B (1 = demonstration row)
    void resize(int b, int S) {
        B = b; s.assign((size_t)b * S, 0.f); ns.assign((size_t)b * S, 0.f);
        a.assign(b, 0); ret.assign(b, 0.f); gn.assign(b, 0.f); demo.assign(b, 0.f);
    }
};

class Learner {
public:
    int S, A, H;
    float lr = 1e-4f, margin = 0.8f, lambda_demo = 1.0f, clip = 10.0f;
    float beta1 = 0.9f, beta2 = 0.999f, adam_eps = 1e-8f;

    Learner(int s, int a, int h) : S(s), A(a), H(h) {}
    virtual ~Learner() {}
    virtual const char* name() const = 0;
    // Parameters in QNet order: W1[S*H], b1[H], W2[H*H], b2[H], W3[H*A], b3[A] (row-major).
    virtual void set_params(const std::vector<std::vector<float>>& p) = 0;
    virtual void get_params(std::vector<std::vector<float>>& p) = 0;
    virtual void sync_target() = 0;                       // target <- online
    virtual float train(const Batch& b) = 0;              // one Adam step, returns mean loss
    virtual void forward(const float* x, int B, float* q) = 0;   // online net, q[B*A]

    size_t num_params() const { return (size_t)S * H + H + (size_t)H * H + H + (size_t)H * A + A; }
};

std::unique_ptr<Learner> make_cpu_learner(int S, int A, int H);
// nullptr when built without CUDA or when no GPU is visible.
std::unique_ptr<Learner> make_cuda_learner(int S, int A, int H);

#ifdef __CUDACC__
#define RL_HD __host__ __device__
#else
#define RL_HD
#endif

// Per-row loss + dL/dQ (already divided by B) -- shared by both backends (the
// CUDA kernel calls it per thread) so the loss lives in exactly one place.
// q/qn_on/qn_tg/dq point at row i.
RL_HD inline float dqfd_row(const float* q, const float* qn_on, const float* qn_tg,
                      int a, float ret, float gn, float demo, int A, int B,
                      float margin, float lambda_demo, float* dq) {
    int astar = 0;
    for (int k = 1; k < A; ++k) if (qn_on[k] > qn_on[astar]) astar = k;
    float y = ret + gn * qn_tg[astar];
    float td = q[a] - y;
    float ad = td < 0 ? -td : td;
    float loss = ad <= 1.f ? 0.5f * td * td : ad - 0.5f;
    for (int k = 0; k < A; ++k) dq[k] = 0.f;
    dq[a] += (td > 1.f ? 1.f : td < -1.f ? -1.f : td) / B;
    if (demo > 0.f) {
        int am = 0; float best = -1e30f;
        for (int k = 0; k < A; ++k) {
            float v = q[k] + (k == a ? 0.f : margin);
            if (v > best) { best = v; am = k; }
        }
        float lm = best - q[a];
        if (lm > 0.f) {
            loss += lambda_demo * lm;
            dq[am] += lambda_demo / B;
            dq[a]  -= lambda_demo / B;
        }
    }
    return loss;
}

}  // namespace rl
