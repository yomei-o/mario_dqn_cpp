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

}  // namespace

std::unique_ptr<Learner> make_cuda_learner(int S, int A, int H) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return nullptr;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
        std::printf("   cuda device 0: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    return std::unique_ptr<Learner>(new CudaLearner(S, A, H));
}

}  // namespace rl
