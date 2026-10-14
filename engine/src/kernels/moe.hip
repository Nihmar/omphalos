// The mixture-of-experts kernels (#493): see moe.hh for the semantics.
#include "kernels/moe.hh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>

namespace omph::kernels {
namespace {

// One block per token, one thread per expert: the logits are the block's
// threads, the softmax a block reduction, the top-k a small scan in thread 0
// (a serial scan of the shared probabilities, ties keeping the lower id).
//
// The router row of expert e starts at `router[e * k]` (F32, ne[0] = k); x is
// f16. expf (not __expf): the reference's soft_max is the f32 one.
constexpr int kRouteThreads = 256;

__global__ void moe_route_kernel(const float * __restrict__ router, const __half * __restrict__ x,
                                 int * __restrict__ ids, float * __restrict__ weights,
                                 const long long k, const int n_expert, const int topk) {
    extern __shared__ float sh[];  // n_expert probabilities
    const int t = blockIdx.x;
    const int e = threadIdx.x;
    const __half * xr = x + (size_t) t * (size_t) k;

    float logit = 0.0f;
    if (e < n_expert) {
        const float * r = router + (size_t) e * (size_t) k;
        for (long long j = 0; j < k; ++j) {
            logit += __half2float(xr[j]) * r[j];
        }
        sh[e] = logit;
    }
    __syncthreads();

    // softmax over all the experts
    float m = -INFINITY;
    for (int i = e; i < n_expert; i += kRouteThreads) {
        m = fmaxf(m, sh[i]);
    }
    for (int o = kRouteThreads / 2; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor(m, o));
    }
    __shared__ float s_max;
    if (e == 0) {
        s_max = m;
    }
    __syncthreads();
    float sum = 0.0f;
    for (int i = e; i < n_expert; i += kRouteThreads) {
        const float p = expf(sh[i] - s_max);
        sh[i] = p;
        sum += p;
    }
    for (int o = kRouteThreads / 2; o > 0; o >>= 1) {
        sum += __shfl_xor(sum, o);
    }
    __shared__ float s_sum;
    if (e == 0) {
        s_sum = sum;
    }
    __syncthreads();

    if (e == 0) {
        int * id_row = ids + (size_t) t * (size_t) topk;
        float * w_row = weights + (size_t) t * (size_t) topk;
        float picked_sum = 0.0f;
        for (int s = 0; s < topk; ++s) {
            int best = -1;
            float best_p = -1.0f;
            for (int i = 0; i < n_expert; ++i) {
                const float p = sh[i] / s_sum;
                // strict: a tie keeps the lower id, as the reference's argsort
                if (p > best_p) {
                    best_p = p;
                    best = i;
                }
            }
            id_row[s] = best;
            w_row[s] = best_p;
            picked_sum += best_p;
            sh[best] = 0.0f;  // do not pick it again
        }
        // renormalize the selected probabilities to sum 1 (norm_w)
        for (int s = 0; s < topk; ++s) {
            w_row[s] /= picked_sum;
        }
    }
}

// out[t][j] += w[t][slot] * y[t][j], one thread per output element.
__global__ void moe_acc_kernel(float * __restrict__ out, const float * __restrict__ y,
                               const float * __restrict__ weights, const long long n, const int stride,
                               const int slot) {
    const long long t = blockIdx.x;
    const float w = weights[(size_t) t * (size_t) stride + (size_t) slot];
    float * orow = out + (size_t) t * (size_t) n;
    const float * yrow = y + (size_t) t * (size_t) n;
    for (long long j = threadIdx.x; j < n; j += blockDim.x) {
        orow[j] += w * yrow[j];
    }
}

// out[t] = sigmoid(dot(x_t, w)), one thread per token, w a F32 row of k values.
__global__ void moe_shexp_gate_kernel(const float * __restrict__ w, const __half * __restrict__ x,
                                      float * __restrict__ out, const long long tokens,
                                      const long long k) {
    const long long t = blockIdx.x * (long long) blockDim.x + threadIdx.x;
    if (t >= tokens) {
        return;
    }
    const __half * xr = x + (size_t) t * (size_t) k;
    float acc = 0.0f;
    for (long long j = 0; j < k; ++j) {
        acc += __half2float(xr[j]) * w[j];
    }
    out[t] = 1.0f / (1.0f + expf(-acc));
}

// one block per row: copy k f16 values
__global__ void moe_gather_kernel(const __half * __restrict__ x, const long long k,
                                  const int * __restrict__ ids, __half * __restrict__ dst) {
    const long long i = blockIdx.x;
    const __half * s = x + (size_t) ids[i] * (size_t) k;
    __half * d = dst + (size_t) i * (size_t) k;
    for (long long j = threadIdx.x; j < k; j += blockDim.x) {
        d[j] = s[j];
    }
}

// one block per gathered row: store it at its (token, slot) in the per-slot
// buffer (a plain store: every (token, slot) pair appears once)
__global__ void moe_scatter_store_kernel(float * __restrict__ xo, const float * __restrict__ y,
                                         const int * __restrict__ slot, const int * __restrict__ dst,
                                         const long long ne, const int topk) {
    const long long i = blockIdx.x;
    const long long at = ((size_t) dst[i] * (size_t) topk + (size_t) slot[i]) * (size_t) ne;
    const float * yrow = y + (size_t) i * (size_t) ne;
    float * orow = xo + at;
    for (long long j = threadIdx.x; j < ne; j += blockDim.x) {
        orow[j] = yrow[j];
    }
}

// one block per token: the slot-ordered weighted sum
__global__ void moe_combine_kernel(float * __restrict__ out, const float * __restrict__ xo,
                                   const float * __restrict__ weights, const long long ne,
                                   const int topk) {
    const long long t = blockIdx.x;
    const float * wrow = weights + (size_t) t * (size_t) topk;
    const float * xrow = xo + (size_t) t * (size_t) topk * (size_t) ne;
    float * orow = out + (size_t) t * (size_t) ne;
    for (long long j = threadIdx.x; j < ne; j += blockDim.x) {
        float acc = 0.0f;
        for (int e = 0; e < topk; ++e) {
            acc += wrow[e] * xrow[(size_t) e * (size_t) ne + j];
        }
        orow[j] = acc;
    }
}

} // namespace

bool moe_route(const void * router, const void * x_f16, int * ids, float * weights, const int64_t tokens,
               const int64_t k, const int64_t n_expert, const int topk, hipStream_t stream) {
    if (router == nullptr || x_f16 == nullptr || ids == nullptr || weights == nullptr || tokens <= 0 ||
        k <= 0 || n_expert <= 0 || n_expert > kRouteThreads || topk <= 0 || topk > n_expert) {
        return false;
    }
    const size_t sh_bytes = (size_t) n_expert * sizeof(float);
    moe_route_kernel<<<(unsigned) tokens, kRouteThreads, sh_bytes, stream>>>(
        static_cast<const float *>(router), static_cast<const __half *>(x_f16), ids, weights,
        (long long) k, (int) n_expert, topk);
    return true;
}

bool moe_acc(float * out, const float * y, const float * weights, const int64_t tokens, const int64_t stride,
             const int slot, const int64_t n, hipStream_t stream) {
    if (out == nullptr || y == nullptr || weights == nullptr || tokens <= 0 || n <= 0 || stride <= 0 ||
        slot < 0 || slot >= stride) {
        return false;
    }
    const unsigned threads = n < 256 ? (unsigned) n : 256;
    moe_acc_kernel<<<(unsigned) tokens, threads, 0, stream>>>(out, y, weights, (long long) n, (int) stride,
                                                              slot);
    return true;
}

bool moe_gather(const void * x_f16, const int64_t k, const int * ids, const int64_t n, void * dst,
                hipStream_t stream) {
    if (x_f16 == nullptr || ids == nullptr || dst == nullptr || k <= 0 || n <= 0) {
        return false;
    }
    moe_gather_kernel<<<(unsigned) n, 256, 0, stream>>>(static_cast<const __half *>(x_f16),
                                                        (long long) k, ids,
                                                        static_cast<__half *>(dst));
    return hipGetLastError() == hipSuccess;
}

bool moe_scatter_store(float * xo, const float * y, const int * slot, const int * dst, const int64_t n,
                       const int64_t ne, const int topk, hipStream_t stream) {
    if (xo == nullptr || y == nullptr || slot == nullptr || dst == nullptr || n <= 0 || ne <= 0 ||
        topk <= 0) {
        return false;
    }
    const unsigned threads = ne < 256 ? (unsigned) ne : 256;
    moe_scatter_store_kernel<<<(unsigned) n, threads, 0, stream>>>(xo, y, slot, dst, (long long) ne,
                                                                   topk);
    return hipGetLastError() == hipSuccess;
}

bool moe_combine(float * out, const float * xo, const float * weights, const int64_t tokens,
                 const int topk, const int64_t ne, hipStream_t stream) {
    if (out == nullptr || xo == nullptr || weights == nullptr || tokens <= 0 || ne <= 0 || topk <= 0) {
        return false;
    }
    const unsigned threads = ne < 256 ? (unsigned) ne : 256;
    moe_combine_kernel<<<(unsigned) tokens, threads, 0, stream>>>(out, xo, weights, (long long) ne,
                                                                  topk);
    return hipGetLastError() == hipSuccess;
}

bool moe_shexp_gate(const void * w, const void * x_f16, float * out, const int64_t tokens, const int64_t k,
                    hipStream_t stream) {
    if (w == nullptr || x_f16 == nullptr || out == nullptr || tokens <= 0 || k <= 0) {
        return false;
    }
    moe_shexp_gate_kernel<<<(unsigned) ((tokens + 127) / 128), 128, 0, stream>>>(
        static_cast<const float *>(w), static_cast<const __half *>(x_f16), out, (long long) tokens,
        (long long) k);
    return true;
}

} // namespace omph::kernels
