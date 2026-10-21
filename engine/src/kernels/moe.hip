// The mixture-of-experts kernels (#493): see moe.hh for the semantics.
#include "kernels/moe.hh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>

namespace omph::kernels {
namespace {

// wave32 reductions (gfx1200): offsets 16..1 stay inside the wave, every lane
// ends with the result. A wider offset returns the lane's own value, so the
// block-level step goes through shared memory (#509).
constexpr int kWave = 32;

__device__ inline float wave_max(float v) {
    for (int o = kWave / 2; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor(v, o));
    }
    return v;
}

__device__ inline float wave_sum(float v) {
    for (int o = kWave / 2; o > 0; o >>= 1) {
        v += __shfl_xor(v, o);
    }
    return v;
}

// the larger value, a tie keeping the lower id (the reference's argsort)
__device__ inline void wave_argmax(float & v, int & i) {
    for (int o = kWave / 2; o > 0; o >>= 1) {
        const float ov = __shfl_xor(v, o);
        const int oi = __shfl_xor(i, o);
        if (ov > v || (ov == v && oi < i)) {
            v = ov;
            i = oi;
        }
    }
}

// The router logits of a few tokens (#509): one wave per (expert, token), a
// fixed reduction order, so a token's logits do not depend on how many tokens
// the call has. The GEMM is ~30 us for one token; this reads the 1 MiB once.
constexpr int kRouterWaves = 8;

__global__ void __launch_bounds__(kRouterWaves * kWave)
moe_router_kernel(const __half * __restrict__ w, const __half * __restrict__ x, const __half * __restrict__ x_lo,
                  float * __restrict__ logits, const int n_expert, const int k) {
    const int e = blockIdx.x * kRouterWaves + (int) (threadIdx.x / kWave);
    const int lane = (int) (threadIdx.x % kWave);
    const int t = blockIdx.y;
    if (e >= n_expert) {
        return;
    }
    const __half2 * wr = reinterpret_cast<const __half2 *>(w + (size_t) e * (size_t) k);
    const __half2 * xr = reinterpret_cast<const __half2 *>(x + (size_t) t * (size_t) k);
    const __half2 * lr = reinterpret_cast<const __half2 *>(x_lo + (size_t) t * (size_t) k);
    float acc = 0.0f;
    for (int j = lane; j < k / 2; j += kWave) {
        const float2 a = __half22float2(wr[j]);
        const float2 b = __half22float2(xr[j]);
        const float2 c = __half22float2(lr[j]);
        acc += a.x * (b.x + c.x) + a.y * (b.y + c.y);
    }
    acc = wave_sum(acc);
    if (lane == 0) {
        logits[(size_t) t * (size_t) n_expert + e] = acc;
    }
}

// One block per token, one thread per expert: the softmax over the token's
// router logits, then topk rounds of a block argmax (ties keeping the lower
// id). The serial scan in thread 0 this replaced was 69 us a layer (#509).
constexpr int kTopkThreads = 256;
constexpr int kTopkWaves = kTopkThreads / kWave;

__global__ void __launch_bounds__(kTopkThreads)
moe_topk_kernel(const float * __restrict__ logits, const float * __restrict__ logits_lo, int * __restrict__ ids,
                float * __restrict__ weights, const int n_expert, const int topk) {
    __shared__ float s_v[kTopkWaves];
    __shared__ int s_i[kTopkWaves];
    const int t = blockIdx.x;
    const int e = threadIdx.x;
    const int lane = e % kWave;
    const int wv = e / kWave;
    const size_t at = (size_t) t * (size_t) n_expert + (size_t) e;
    const float l = e < n_expert ? logits[at] + (logits_lo != nullptr ? logits_lo[at] : 0.0f) : -INFINITY;

    // softmax over all the experts (expf: the reference's soft_max is the f32 one)
    float m = wave_max(l);
    if (lane == 0) {
        s_v[wv] = m;
    }
    __syncthreads();
    m = s_v[0];
    for (int i = 1; i < kTopkWaves; ++i) {
        m = fmaxf(m, s_v[i]);
    }
    __syncthreads();
    const float ex = e < n_expert ? expf(l - m) : 0.0f;
    float sum = wave_sum(ex);
    if (lane == 0) {
        s_v[wv] = sum;
    }
    __syncthreads();
    sum = 0.0f;
    for (int i = 0; i < kTopkWaves; ++i) {
        sum += s_v[i];
    }
    __syncthreads();

    // a picked (or absent) expert competes with -1: below every probability
    float cand = e < n_expert ? ex / sum : -1.0f;
    float picked_sum = 0.0f;
    int * id_row = ids + (size_t) t * (size_t) topk;
    float * w_row = weights + (size_t) t * (size_t) topk;
    for (int s = 0; s < topk; ++s) {
        float v = cand;
        int i = e;
        wave_argmax(v, i);
        if (lane == 0) {
            s_v[wv] = v;
            s_i[wv] = i;
        }
        __syncthreads();
        v = s_v[0];
        i = s_i[0];
        for (int r = 1; r < kTopkWaves; ++r) {
            if (s_v[r] > v || (s_v[r] == v && s_i[r] < i)) {
                v = s_v[r];
                i = s_i[r];
            }
        }
        __syncthreads();
        if (e == i) {
            cand = -1.0f;
        }
        if (e == 0) {
            id_row[s] = i;
            w_row[s] = v;
            picked_sum += v;
        }
    }
    // renormalize the selected probabilities to sum 1 (norm_w)
    if (e == 0) {
        for (int s = 0; s < topk; ++s) {
            w_row[s] /= picked_sum;
        }
    }
}

// one block per gathered row: store it at its (token, slot) in the per-slot
// buffer (a plain store: every (token, slot) pair appears once)
// out[t][j] += w[t][slot] * y[t][j], one thread per output element (the decode
// path's per-expert accumulate and the shared expert's weighted sum).
__global__ void moe_acc_kernel(float * __restrict__ out, const float * __restrict__ y,
                               const float * __restrict__ weights, const long long n, const int stride,
                               const int slot) {
    const long long t = blockIdx.y;
    const long long j = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) {
        return;
    }
    const float w = weights[(size_t) t * (size_t) stride + (size_t) slot];
    out[(size_t) t * (size_t) n + j] += w * y[(size_t) t * (size_t) n + j];
}

// The routing table on the device (#493): per-expert counts, their exclusive
// prefix sum, and the gather (xgi = the token) / slot arrays in expert-major
// order - so a batched step needs no host round trip. The order *inside* an
// expert's rows does not affect any result: the outputs are summed back per
// (token, slot), and the combine runs in slot order.
// one block, one thread per expert (its counter)
constexpr int kOrderThreads = 256;

__global__ void moe_order_kernel(const int * __restrict__ ids, const int n_pairs, const int n_expert,
                                 const int topk, int * __restrict__ counts, int * __restrict__ offs,
                                 int * __restrict__ xgi, int * __restrict__ slot,
                                 int * __restrict__ max_count) {
    extern __shared__ int cnt[];  // n_expert + 1 counters
    const int e = threadIdx.x;
    if (e < n_expert) {
        // every thread counts all the pairs: the thread is the *bin*, the loop
        // is over the pairs (a strided read here would compare ids[i] with i)
        int c = 0;
        for (int i = 0; i < n_pairs; ++i) {
            c += ids[i] == e;
        }
        cnt[e] = c;
        counts[e] = c;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int at = 0;
        int mx = 0;
        for (int i = 0; i < n_expert; ++i) {
            offs[i] = at;
            const int c = cnt[i];
            cnt[i] = at;
            at += c;
            mx = c > mx ? c : mx;
        }
        offs[n_expert] = at;
        *max_count = mx;
    }
    __syncthreads();
    for (int i = (int) threadIdx.x; i < n_pairs; i += (int) blockDim.x) {
        const int at = atomicAdd(&cnt[ids[i]], 1);
        xgi[at] = i / topk;
        slot[at] = i % topk;
    }
}

// rows of 32-bit words, one thread per word (blockIdx.y the row)
__global__ void moe_copy_words_kernel(uint32_t * __restrict__ dst, const long long dpitch,
                                      const uint32_t * __restrict__ src, const long long spitch,
                                      const long long n) {
    const long long j = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        dst[(long long) blockIdx.y * dpitch + j] = src[(long long) blockIdx.y * spitch + j];
    }
}

// one block per row: copy k f16 values (the gather of an expert's rows)
__global__ void moe_gather_kernel(const __half * __restrict__ x, const long long k,
                                  const int * __restrict__ ids, __half * __restrict__ dst) {
    const long long i = blockIdx.x;
    const __half * s = x + (size_t) ids[i] * (size_t) k;
    __half * d = dst + (size_t) i * (size_t) k;
    for (long long j = threadIdx.x; j < k; j += blockDim.x) {
        d[j] = s[j];
    }
}

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

// The end of a MoE layer (#522): out[t][j] = sum_e w[t][e] * xo[t][e][j] (slot
// order; or out[t][j] as it is when xo is null: the per-expert fallback
// accumulated it) + sigmoid(x_t . wg) * ysh[t][j]. Each block computes the
// shared gate's dot itself (a fixed order: 8 waves, then their sums in order),
// one launch instead of three.
constexpr int kFinishThreads = 256;

__global__ void __launch_bounds__(kFinishThreads)
moe_finish_kernel(float * __restrict__ out, const float * __restrict__ xo, const float * __restrict__ weights,
                  const int topk, const float * __restrict__ ysh, const float * __restrict__ wg,
                  const __half * __restrict__ x, const __half * __restrict__ x_lo, const long long ne) {
    __shared__ float s_sum[kFinishThreads / kWave];
    const long long t = blockIdx.y;
    const __half * xr = x + (size_t) t * (size_t) ne;
    const __half * lr = x_lo + (size_t) t * (size_t) ne;
    float d = 0.0f;
    for (long long j = threadIdx.x; j < ne; j += kFinishThreads) {
        d += (__half2float(xr[j]) + __half2float(lr[j])) * wg[j];
    }
    d = wave_sum(d);
    if (threadIdx.x % kWave == 0) {
        s_sum[threadIdx.x / kWave] = d;
    }
    __syncthreads();
    d = 0.0f;
    for (int i = 0; i < kFinishThreads / kWave; ++i) {
        d += s_sum[i];
    }
    const float g = 1.0f / (1.0f + expf(-d));
    const long long j = (long long) blockIdx.x * kFinishThreads + threadIdx.x;
    if (j >= ne) {
        return;
    }
    float acc;
    if (xo != nullptr) {
        const float * wrow = weights + (size_t) t * (size_t) topk;
        const float * xrow = xo + (size_t) t * (size_t) topk * (size_t) ne;
        acc = 0.0f;
        for (int e = 0; e < topk; ++e) {
            acc += wrow[e] * xrow[(size_t) e * (size_t) ne + j];
        }
    } else {
        acc = out[(size_t) t * (size_t) ne + j];
    }
    out[(size_t) t * (size_t) ne + j] = acc + g * ysh[(size_t) t * (size_t) ne + j];
}

} // namespace

bool moe_topk(const void * logits, const void * logits_lo, int * ids, float * weights, const int64_t tokens,
              const int64_t n_expert, const int topk, hipStream_t stream) {
    if (logits == nullptr || ids == nullptr || weights == nullptr || tokens <= 0 || n_expert <= 0 ||
        n_expert > kTopkThreads || topk <= 0 || topk > n_expert) {
        return false;
    }
    moe_topk_kernel<<<(unsigned) tokens, kTopkThreads, 0, stream>>>(
        static_cast<const float *>(logits), static_cast<const float *>(logits_lo), ids, weights, (int) n_expert,
        topk);
    return hipGetLastError() == hipSuccess;
}

bool moe_router(const void * w_f16, const void * x_f16, const void * x_lo_f16, float * logits, const int64_t tokens,
                const int64_t n_expert, const int64_t k, hipStream_t stream) {
    if (w_f16 == nullptr || x_f16 == nullptr || x_lo_f16 == nullptr || logits == nullptr || tokens <= 0 ||
        n_expert <= 0 ||
        k <= 0 || k % 2 != 0) {
        return false;
    }
    const dim3 grid((unsigned) ((n_expert + kRouterWaves - 1) / kRouterWaves), (unsigned) tokens);
    moe_router_kernel<<<grid, kRouterWaves * kWave, 0, stream>>>(
        static_cast<const __half *>(w_f16), static_cast<const __half *>(x_f16),
        static_cast<const __half *>(x_lo_f16), logits, (int) n_expert, (int) k);
    return hipGetLastError() == hipSuccess;
}

bool moe_copy_words(void * dst, const int64_t dpitch, const void * src, const int64_t spitch, const int64_t n,
                    const int64_t rows, hipStream_t stream) {
    if (dst == nullptr || src == nullptr || n <= 0 || rows <= 0 || (rows > 1 && (dpitch < n || spitch < n))) {
        return false;
    }
    const dim3 grid((unsigned) ((n + 255) / 256), (unsigned) rows);
    moe_copy_words_kernel<<<grid, 256, 0, stream>>>(static_cast<uint32_t *>(dst), (long long) dpitch,
                                                    static_cast<const uint32_t *>(src), (long long) spitch,
                                                    (long long) n);
    return hipGetLastError() == hipSuccess;
}

bool moe_order(const void * ids, const int64_t n_pairs, const int64_t n_expert, const int64_t topk,
               void * counts, void * offs, void * xgi, void * slot, void * max_count,
               hipStream_t stream) {
    if (ids == nullptr || counts == nullptr || offs == nullptr || xgi == nullptr || slot == nullptr ||
        max_count == nullptr || n_pairs <= 0 || n_expert <= 0 || n_expert > kOrderThreads || topk <= 0) {
        return false;
    }
    const size_t sh_bytes = (size_t) (n_expert + 1) * sizeof(int);
    moe_order_kernel<<<1, kOrderThreads, sh_bytes, stream>>>(static_cast<const int *>(ids), (int) n_pairs,
                                                   (int) n_expert, (int) topk,
                                                   static_cast<int *>(counts), static_cast<int *>(offs),
                                                   static_cast<int *>(xgi), static_cast<int *>(slot),
                                                   static_cast<int *>(max_count));
    return hipGetLastError() == hipSuccess;
}

bool moe_acc(float * out, const float * y, const float * weights, const int64_t tokens, const int64_t stride,
             const int slot, const int64_t n, hipStream_t stream) {
    if (out == nullptr || y == nullptr || weights == nullptr || tokens <= 0 || n <= 0 || stride <= 0 ||
        slot < 0 || slot >= stride) {
        return false;
    }
    const dim3 grid((unsigned) ((n + 255) / 256), (unsigned) tokens);
    moe_acc_kernel<<<grid, 256, 0, stream>>>(out, y, weights, (long long) n, (int) stride, slot);
    return hipGetLastError() == hipSuccess;
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

bool moe_finish(float * out, const float * xo, const float * weights, const int topk, const float * ysh,
                const void * wg, const void * x_f16, const void * x_lo_f16, const int64_t tokens, const int64_t ne,
                hipStream_t stream) {
    if (out == nullptr || ysh == nullptr || wg == nullptr || x_f16 == nullptr || x_lo_f16 == nullptr || tokens <= 0 ||
        ne <= 0 ||
        (xo != nullptr && (weights == nullptr || topk <= 0))) {
        return false;
    }
    const dim3 grid((unsigned) ((ne + kFinishThreads - 1) / kFinishThreads), (unsigned) tokens);
    moe_finish_kernel<<<grid, kFinishThreads, 0, stream>>>(out, xo, weights, topk, ysh,
                                                           static_cast<const float *>(wg),
                                                           static_cast<const __half *>(x_f16),
                                                           static_cast<const __half *>(x_lo_f16), (long long) ne);
    return hipGetLastError() == hipSuccess;
}


} // namespace omph::kernels
