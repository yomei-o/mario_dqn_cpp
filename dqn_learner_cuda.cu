// CUDA backend of the batched DQfD learner: cuBLAS SGEMM for the three dense
// layers (forward and backward), small kernels for bias/ReLU, the per-row loss
// (dqfd_row from dqn_learner.h, the same code the CPU backend runs), the global
// grad-norm and Adam. Parameters live on the GPU as one flat buffer in QNet
// order, so get/set_params is a plain memcpy.
//
// Row-major matrices are handed to column-major cuBLAS as their transposes:
// a row-major X[B,I] is a column-major I x B matrix with ld=I. For a layer with
// row-major W[I,O]:  Y = X W      -> gemm(N,N, O,B,I, W,O, X,I, Y,O)
//                   dW = X^T dY   -> gemm(N,T, O,I,B, dY,O, X,I, dW,O)
//                   dX = dY W^T   -> gemm(T,N, I,B,O, W,O, dY,O, dX,I)
#include "dqn_learner.h"
#include "conv_net.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

namespace rl {

namespace {

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::abort(); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuBLAS %s:%d status %d\n", __FILE__, __LINE__, (int)s_); std::abort(); } } while (0)

__global__ void k_bias_act(float* Y, const float* b, int B, int O, int relu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * O) return;
    float v = Y[i] + b[i % O];
    Y[i] = (relu && v < 0.f) ? 0.f : v;
}

__global__ void k_relu_mask(float* dY, const float* Y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && Y[i] <= 0.f) dY[i] = 0.f;
}

// db[o] = sum_r dY[r,o]
__global__ void k_colsum(const float* dY, int B, int O, float* db) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= O) return;
    float acc = 0.f;
    for (int r = 0; r < B; ++r) acc += dY[(size_t)r * O + o];
    db[o] = acc;
}

__global__ void k_loss(const float* q, const float* qno, const float* qnt, const int* a,
                       const float* ret, const float* gn, const float* demo, int B, int A,
                       float margin, float lambda_demo, float* dq, float* loss) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= B) return;
    loss[r] = dqfd_row(q + (size_t)r * A, qno + (size_t)r * A, qnt + (size_t)r * A,
                       a[r], ret[r], gn[r], demo[r], A, B, margin, lambda_demo, dq + (size_t)r * A);
}

__global__ void k_sumsq(const float* g, size_t n, float* out) {
    __shared__ float sh[256];
    float acc = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        acc += g[i] * g[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}

__global__ void k_adam(float* P, const float* G, float* M, float* V, size_t n, const float* sumsq,
                       float clip, float lr, float b1, float b2, float bc1, float bc2, float eps) {
    size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (j >= n) return;
    float norm = sqrtf(*sumsq);
    float scale = norm > clip ? clip / (norm + 1e-6f) : 1.f;
    float g = G[j] * scale;
    float m = b1 * M[j] + (1 - b1) * g;
    float v = b2 * V[j] + (1 - b2) * g * g;
    M[j] = m; V[j] = v;
    P[j] -= lr * (m / bc1) / (sqrtf(v / bc2) + eps);
}

inline int nblk(size_t n, int t = 256) { return (int)((n + t - 1) / t); }

class CudaLearner : public Learner {
public:
    cublasHandle_t hb = nullptr;
    size_t n, oW1, ob1, oW2, ob2, oW3, ob3;
    float *P = nullptr, *T = nullptr, *G = nullptr, *M = nullptr, *V = nullptr, *sumsq = nullptr;
    int cap = 0;   // batch capacity of the scratch buffers
    float *X = nullptr, *XN = nullptr, *h1 = nullptr, *h2 = nullptr, *q = nullptr;
    float *h1n = nullptr, *h2n = nullptr, *qno = nullptr, *qnt = nullptr;
    float *dq = nullptr, *dh2 = nullptr, *dh1 = nullptr;
    float *ret = nullptr, *gn = nullptr, *demo = nullptr, *loss = nullptr;
    int* act = nullptr;
    long t = 0;
    std::vector<float> loss_h;

    CudaLearner(int s, int a, int h) : Learner(s, a, h) {
        oW1 = 0; ob1 = oW1 + (size_t)S * H; oW2 = ob1 + H; ob2 = oW2 + (size_t)H * H;
        oW3 = ob2 + H; ob3 = oW3 + (size_t)H * A;
        n = num_params();
        CB(cublasCreate(&hb));
        for (float** p : {&P, &T, &G, &M, &V}) { CK(cudaMalloc(p, n * sizeof(float))); CK(cudaMemset(*p, 0, n * sizeof(float))); }
        CK(cudaMalloc(&sumsq, sizeof(float)));
    }
    ~CudaLearner() override {
        for (float* p : {P, T, G, M, V, sumsq}) cudaFree(p);
        free_scratch();
        if (hb) cublasDestroy(hb);
    }
    const char* name() const override { return "cuda"; }

    void free_scratch() {
        for (float* p : {X, XN, h1, h2, q, h1n, h2n, qno, qnt, dq, dh2, dh1, ret, gn, demo, loss}) cudaFree(p);
        cudaFree(act);
    }
    void ensure(int B) {
        if (B <= cap) return;
        free_scratch();
        cap = B;
        auto al = [&](float** p, size_t cnt) { CK(cudaMalloc(p, cnt * sizeof(float))); };
        al(&X, (size_t)B * S); al(&XN, (size_t)B * S);
        al(&h1, (size_t)B * H); al(&h2, (size_t)B * H); al(&q, (size_t)B * A);
        al(&h1n, (size_t)B * H); al(&h2n, (size_t)B * H); al(&qno, (size_t)B * A); al(&qnt, (size_t)B * A);
        al(&dq, (size_t)B * A); al(&dh2, (size_t)B * H); al(&dh1, (size_t)B * H);
        al(&ret, B); al(&gn, B); al(&demo, B); al(&loss, B);
        CK(cudaMalloc(&act, B * sizeof(int)));
        loss_h.resize(B);
    }

    void set_params(const std::vector<std::vector<float>>& p) override {
        size_t off[6] = {oW1, ob1, oW2, ob2, oW3, ob3};
        for (int i = 0; i < 6; ++i)
            CK(cudaMemcpy(P + off[i], p[i].data(), p[i].size() * sizeof(float), cudaMemcpyHostToDevice));
        sync_target();
    }
    void get_params(std::vector<std::vector<float>>& p) override {
        size_t off[7] = {oW1, ob1, oW2, ob2, oW3, ob3, n};
        p.resize(6);
        for (int i = 0; i < 6; ++i) {
            p[i].resize(off[i + 1] - off[i]);
            CK(cudaMemcpy(p[i].data(), P + off[i], p[i].size() * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
    void sync_target() override { CK(cudaMemcpy(T, P, n * sizeof(float), cudaMemcpyDeviceToDevice)); }

    void gemm_fwd(const float* W, const float* Xin, int B, int I, int O, float* Y) {
        const float one = 1.f, zero = 0.f;
        CB(cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, O, B, I, &one, W, O, Xin, I, &zero, Y, O));
    }
    void layer(const float* W, const float* b, const float* Xin, int B, int I, int O, bool relu, float* Y) {
        gemm_fwd(W, Xin, B, I, O, Y);
        k_bias_act<<<nblk((size_t)B * O), 256>>>(Y, b, B, O, relu ? 1 : 0);
    }
    void fwd(const float* W, const float* x, int B, float* a1, float* a2, float* out) {
        layer(W + oW1, W + ob1, x, B, S, H, true, a1);
        layer(W + oW2, W + ob2, a1, B, H, H, true, a2);
        layer(W + oW3, W + ob3, a2, B, H, A, false, out);
    }
    void back(const float* W, const float* Xin, const float* dY, int B, int I, int O,
              float* dW, float* db, float* dX) {
        const float one = 1.f, zero = 0.f;
        CB(cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_T, O, I, B, &one, dY, O, Xin, I, &zero, dW, O));
        k_colsum<<<nblk(O), 256>>>(dY, B, O, db);
        if (dX) CB(cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, I, B, O, &one, W, O, dY, O, &zero, dX, I));
    }

    void forward(const float* x, int B, float* qout) override {
        ensure(B);
        CK(cudaMemcpy(X, x, (size_t)B * S * sizeof(float), cudaMemcpyHostToDevice));
        fwd(P, X, B, h1, h2, q);
        CK(cudaMemcpy(qout, q, (size_t)B * A * sizeof(float), cudaMemcpyDeviceToHost));
    }

    float train(const Batch& b) override {
        const int B = b.B;
        ensure(B);
        CK(cudaMemcpy(X, b.s.data(), (size_t)B * S * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(XN, b.ns.data(), (size_t)B * S * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(act, b.a.data(), B * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ret, b.ret.data(), B * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(gn, b.gn.data(), B * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(demo, b.demo.data(), B * sizeof(float), cudaMemcpyHostToDevice));

        fwd(P, XN, B, h1n, h2n, qno);
        fwd(T, XN, B, h1n, h2n, qnt);
        fwd(P, X, B, h1, h2, q);
        k_loss<<<nblk(B, 128), 128>>>(q, qno, qnt, act, ret, gn, demo, B, A, margin, lambda_demo, dq, loss);

        back(P + oW3, h2, dq, B, H, A, G + oW3, G + ob3, dh2);
        k_relu_mask<<<nblk((size_t)B * H), 256>>>(dh2, h2, B * H);
        back(P + oW2, h1, dh2, B, H, H, G + oW2, G + ob2, dh1);
        k_relu_mask<<<nblk((size_t)B * H), 256>>>(dh1, h1, B * H);
        back(P + oW1, X, dh1, B, S, H, G + oW1, G + ob1, nullptr);

        CK(cudaMemset(sumsq, 0, sizeof(float)));
        k_sumsq<<<128, 256>>>(G, n, sumsq);
        ++t;
        float bc1 = 1.f - std::pow(beta1, (float)t), bc2 = 1.f - std::pow(beta2, (float)t);
        k_adam<<<nblk(n), 256>>>(P, G, M, V, n, sumsq, clip, lr, beta1, beta2, bc1, bc2, adam_eps);

        CK(cudaGetLastError());
        CK(cudaMemcpy(loss_h.data(), loss, B * sizeof(float), cudaMemcpyDeviceToHost));
        double acc = 0;
        for (int r = 0; r < B; ++r) acc += loss_h[r];
        return (float)(acc / B);
    }
};

// ---------------------------------------------------------------- conv net (conv_net.h)
using CA = ConvArch;

__global__ void k_im2col(float* col, const float* src, int B, int Ho, int Wo, int K, int from_obs,
                         int Cin, int Hin, int Win, int stride, size_t b_stride) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t total = (size_t)B * Ho * Wo * K;
    if (i >= total) return;
    int k = (int)(i % K);
    size_t p = i / K;
    int ow = (int)(p % Wo), oh = (int)((p / Wo) % Ho), b = (int)(p / ((size_t)Wo * Ho));
    col[i] = im2col_at(src, b, oh, ow, k, from_obs != 0, Cin, Hin, Win, stride, b_stride);
}

__global__ void k_concat(float* z, const float* a2, const float* x, int B, int F, int NS, int IN) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    int FZ = F + NS;
    if (i >= (size_t)B * FZ) return;
    int k = (int)(i % FZ), b = (int)(i / FZ);
    z[i] = k < F ? a2[(size_t)b * F + k] : x[(size_t)b * IN + (IN - NS) + (k - F)];
}

__global__ void k_split_mask(float* da2, const float* dz, const float* a2, int B, int F, int FZ) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)B * F) return;
    int k = (int)(i % F), b = (int)(i / F);
    da2[i] = a2[i] > 0.f ? dz[(size_t)b * FZ + k] : 0.f;
}

__global__ void k_col2im_mask(float* da1, const float* dcol, const float* a1, int B, int H1, int W1, int C1,
                              int H2, int W2, int stride) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)B * H1 * W1 * C1) return;
    int c = (int)(i % C1);
    size_t p = i / C1;
    int w = (int)(p % W1), h = (int)((p / W1) % H1), b = (int)(p / ((size_t)W1 * H1));
    da1[i] = a1[i] > 0.f ? col2im_at(dcol, b, h, w, c, C1, H2, W2, stride) : 0.f;
}

class ConvCudaLearner : public Learner {
public:
    cublasHandle_t hb = nullptr;
    ConvArch ar;
    float *P = nullptr, *T = nullptr, *G = nullptr, *M = nullptr, *V = nullptr, *sumsq = nullptr;
    int cap = 0;
    // per-pass activations: [0] online(s) [1] online(ns) [2] target(ns)
    float *X = nullptr, *XN = nullptr;
    float *col1[3] = {}, *a1[3] = {}, *col2[3] = {}, *a2[3] = {}, *z[3] = {}, *hh[3] = {}, *q[3] = {};
    float *dq = nullptr, *dh = nullptr, *dz = nullptr, *da2 = nullptr, *dcol2 = nullptr, *da1 = nullptr;
    float *ret = nullptr, *gn = nullptr, *demo = nullptr, *loss = nullptr;
    int* act = nullptr;
    long t = 0;
    std::vector<float> loss_h;

    ConvCudaLearner(int a, int h) : Learner(CA::IN, a, h), ar(a, h) {
        CB(cublasCreate(&hb));
        for (float** p : {&P, &T, &G, &M, &V}) { CK(cudaMalloc(p, ar.n * sizeof(float))); CK(cudaMemset(*p, 0, ar.n * sizeof(float))); }
        CK(cudaMalloc(&sumsq, sizeof(float)));
    }
    ~ConvCudaLearner() override {
        for (float* p : {P, T, G, M, V, sumsq}) cudaFree(p);
        free_scratch();
        if (hb) cublasDestroy(hb);
    }
    const char* name() const override { return "cuda-conv"; }

    void free_scratch() {
        for (int i = 0; i < 3; ++i) for (float* p : {col1[i], a1[i], col2[i], a2[i], z[i], hh[i], q[i]}) cudaFree(p);
        for (float* p : {X, XN, dq, dh, dz, da2, dcol2, da1, ret, gn, demo, loss}) cudaFree(p);
        cudaFree(act);
    }
    void ensure(int B) {
        if (B <= cap) return;
        free_scratch();
        cap = B;
        auto al = [&](float** p, size_t cnt) { CK(cudaMalloc(p, cnt * sizeof(float))); };
        const size_t R1 = (size_t)B * CA::H1 * CA::W1, R2 = (size_t)B * CA::H2 * CA::W2, FZ = CA::F + CA::NS;
        al(&X, (size_t)B * CA::IN); al(&XN, (size_t)B * CA::IN);
        for (int i = 0; i < 3; ++i) {
            al(&col1[i], R1 * CA::K1); al(&a1[i], R1 * CA::C1); al(&col2[i], R2 * CA::K2); al(&a2[i], R2 * CA::C2);
            al(&z[i], (size_t)B * FZ); al(&hh[i], (size_t)B * H); al(&q[i], (size_t)B * A);
        }
        al(&dq, (size_t)B * A); al(&dh, (size_t)B * H); al(&dz, (size_t)B * FZ); al(&da2, R2 * CA::C2);
        al(&dcol2, R2 * CA::K2); al(&da1, R1 * CA::C1);
        al(&ret, B); al(&gn, B); al(&demo, B); al(&loss, B);
        CK(cudaMalloc(&act, B * sizeof(int)));
        loss_h.resize(B);
    }

    void set_params(const std::vector<std::vector<float>>& p) override {
        auto off = ar.offsets();
        for (size_t i = 0; i + 1 < off.size(); ++i)
            CK(cudaMemcpy(P + off[i], p[i].data(), p[i].size() * sizeof(float), cudaMemcpyHostToDevice));
        sync_target();
    }
    void get_params(std::vector<std::vector<float>>& p) override {
        auto off = ar.offsets();
        p.resize(off.size() - 1);
        for (size_t i = 0; i + 1 < off.size(); ++i) {
            p[i].resize(off[i + 1] - off[i]);
            CK(cudaMemcpy(p[i].data(), P + off[i], p[i].size() * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
    void sync_target() override { CK(cudaMemcpy(T, P, ar.n * sizeof(float), cudaMemcpyDeviceToDevice)); }

    void layer(const float* W, const float* b, const float* Xin, int B, int I, int O, bool relu, float* Y) {
        const float one = 1.f, zero = 0.f;
        CB(cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, O, B, I, &one, W, O, Xin, I, &zero, Y, O));
        k_bias_act<<<nblk((size_t)B * O), 256>>>(Y, b, B, O, relu ? 1 : 0);
    }
    void back(const float* W, const float* Xin, const float* dY, int B, int I, int O, float* dW, float* db, float* dX) {
        const float one = 1.f, zero = 0.f;
        CB(cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_T, O, I, B, &one, dY, O, Xin, I, &zero, dW, O));
        k_colsum<<<nblk(O), 256>>>(dY, B, O, db);
        if (dX) CB(cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, I, B, O, &one, W, O, dY, O, &zero, dX, I));
    }
    void fwd(const float* W, const float* x, int B, int s) {
        const int R1 = B * CA::H1 * CA::W1, R2 = B * CA::H2 * CA::W2, FZ = CA::F + CA::NS;
        k_im2col<<<nblk((size_t)R1 * CA::K1), 256>>>(col1[s], x, B, CA::H1, CA::W1, CA::K1, 1, CA::C0, CA::H0, CA::W0, 1, CA::IN);
        layer(W + ar.oWc1, W + ar.obc1, col1[s], R1, CA::K1, CA::C1, true, a1[s]);
        k_im2col<<<nblk((size_t)R2 * CA::K2), 256>>>(col2[s], a1[s], B, CA::H2, CA::W2, CA::K2, 0, CA::C1, CA::H1, CA::W1, 2,
                                                     (size_t)CA::H1 * CA::W1 * CA::C1);
        layer(W + ar.oWc2, W + ar.obc2, col2[s], R2, CA::K2, CA::C2, true, a2[s]);
        k_concat<<<nblk((size_t)B * FZ), 256>>>(z[s], a2[s], x, B, CA::F, CA::NS, CA::IN);
        layer(W + ar.oW3, W + ar.ob3, z[s], B, FZ, H, true, hh[s]);
        layer(W + ar.oW4, W + ar.ob4, hh[s], B, H, A, false, q[s]);
    }

    void forward(const float* x, int B, float* qout) override {
        ensure(B);
        CK(cudaMemcpy(X, x, (size_t)B * CA::IN * sizeof(float), cudaMemcpyHostToDevice));
        fwd(P, X, B, 0);
        CK(cudaMemcpy(qout, q[0], (size_t)B * A * sizeof(float), cudaMemcpyDeviceToHost));
    }

    float train(const Batch& b) override {
        const int B = b.B;
        ensure(B);
        CK(cudaMemcpy(X, b.s.data(), (size_t)B * CA::IN * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(XN, b.ns.data(), (size_t)B * CA::IN * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(act, b.a.data(), B * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ret, b.ret.data(), B * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(gn, b.gn.data(), B * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(demo, b.demo.data(), B * sizeof(float), cudaMemcpyHostToDevice));
        fwd(P, XN, B, 1);
        fwd(T, XN, B, 2);
        fwd(P, X, B, 0);
        k_loss<<<nblk(B, 128), 128>>>(q[0], q[1], q[2], act, ret, gn, demo, B, A, margin, lambda_demo, dq, loss);

        const int R1 = B * CA::H1 * CA::W1, R2 = B * CA::H2 * CA::W2, FZ = CA::F + CA::NS;
        back(P + ar.oW4, hh[0], dq, B, H, A, G + ar.oW4, G + ar.ob4, dh);
        k_relu_mask<<<nblk((size_t)B * H), 256>>>(dh, hh[0], B * H);
        back(P + ar.oW3, z[0], dh, B, FZ, H, G + ar.oW3, G + ar.ob3, dz);
        k_split_mask<<<nblk((size_t)B * CA::F), 256>>>(da2, dz, a2[0], B, CA::F, FZ);
        back(P + ar.oWc2, col2[0], da2, R2, CA::K2, CA::C2, G + ar.oWc2, G + ar.obc2, dcol2);
        k_col2im_mask<<<nblk((size_t)R1 * CA::C1), 256>>>(da1, dcol2, a1[0], B, CA::H1, CA::W1, CA::C1, CA::H2, CA::W2, 2);
        back(P + ar.oWc1, col1[0], da1, R1, CA::K1, CA::C1, G + ar.oWc1, G + ar.obc1, nullptr);

        CK(cudaMemset(sumsq, 0, sizeof(float)));
        k_sumsq<<<128, 256>>>(G, ar.n, sumsq);
        ++t;
        float bc1 = 1.f - std::pow(beta1, (float)t), bc2 = 1.f - std::pow(beta2, (float)t);
        k_adam<<<nblk(ar.n), 256>>>(P, G, M, V, ar.n, sumsq, clip, lr, beta1, beta2, bc1, bc2, adam_eps);
        CK(cudaGetLastError());
        CK(cudaMemcpy(loss_h.data(), loss, B * sizeof(float), cudaMemcpyDeviceToHost));
        double acc = 0;
        for (int r = 0; r < B; ++r) acc += loss_h[r];
        return (float)(acc / B);
    }
};

}  // namespace

std::unique_ptr<Learner> make_conv_cuda_learner(int A, int H) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return nullptr;
    return std::unique_ptr<Learner>(new ConvCudaLearner(A, H));
}

std::unique_ptr<Learner> make_cuda_learner(int S, int A, int H) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return nullptr;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
        std::printf("   cuda device 0: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    return std::unique_ptr<Learner>(new CudaLearner(S, A, H));
}

}  // namespace rl
