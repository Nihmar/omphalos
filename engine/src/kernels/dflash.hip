// DFlash2 drafter kernels (#245); see dflash.hh.
#include "kernels/dflash.hh"

#include "kernels/quant_decode.hh"

#include <cfloat>

namespace omph::kernels {
namespace {

constexpr int kHd = 128;      // head dim
constexpr int kGqa = 4;       // query heads per KV head (32 / 8)
constexpr int kKvHeads = 8;
constexpr int kTopK = 16;
constexpr int kRank = 256;    // selector rank
constexpr int kMaxKeys = 2048 + 64;

__global__ void capture_kernel(const float * __restrict__ x, const long long n, const long long ld_x,
                               __half * __restrict__ feat, const long long ld_f, const long long col) {
    const long long t = blockIdx.y;
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        feat[t * ld_f + col + i] = __float2half(x[t * ld_x + i]);
    }
}

// One block per (token, head), 128 threads: RMSNorm, then NeoX RoPE on (i, i + 64).
__global__ __launch_bounds__(kHd) void norm_rope_kernel(float * __restrict__ x, const long long heads,
                                                        const float * __restrict__ w, const float eps,
                                                        const long long pos0, const float theta) {
    __shared__ float red[kHd / 32];
    __shared__ float v[kHd];
    const long long t = blockIdx.x / heads;
    float * row = x + (long long) blockIdx.x * kHd;
    const int i = (int) threadIdx.x;
    float a = row[i];
    if (w != nullptr) {
        float sq = a * a;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sq += __shfl_xor(sq, off);
        }
        if ((i & 31) == 0) {
            red[i >> 5] = sq;
        }
        __syncthreads();
        const float tot = red[0] + red[1] + red[2] + red[3];
        a = a * (1.0f / sqrtf(tot / (float) kHd + eps)) * w[i];
    }
    v[i] = a;
    __syncthreads();
    if (i < kHd / 2) {
        const float ang = (float) (pos0 + t) * powf(theta, -2.0f * (float) i / (float) kHd);
        const float c = cosf(ang);
        const float s = sinf(ang);
        const float x0 = v[i];
        const float x1 = v[i + kHd / 2];
        row[i] = x0 * c - x1 * s;
        row[i + kHd / 2] = x1 * c + x0 * s;
    }
}

__global__ void ring_put_kernel(const float * __restrict__ src, const long long row, const long long pos0,
                                __half * __restrict__ ring, int32_t * __restrict__ tags, const long long slots) {
    const long long t = blockIdx.y;
    const long long slot = (pos0 + t) % slots;
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row) {
        ring[slot * row + i] = __float2half(src[t * row + i]);
    }
    if (i == 0 && tags != nullptr) {
        tags[slot] = (int32_t) (pos0 + t);
    }
}

// One block per (query head, token), 256 threads = 8 waves. Scores: one wave per
// key, lane l on dims 4 l .. 4 l + 3; softmax in LDS; output: thread = (dim,
// half of the keys).
__global__ __launch_bounds__(256) void attention_kernel(const float * __restrict__ q,
                                                        const __half * __restrict__ ring_k,
                                                        const __half * __restrict__ ring_v,
                                                        const int32_t * __restrict__ tags, const long long slots,
                                                        const float * __restrict__ bk, const float * __restrict__ bv,
                                                        const long long T, const long long pos0,
                                                        const long long n_swa, const float scale,
                                                        float * __restrict__ out) {
    __shared__ float qs[kHd];
    __shared__ float sc[kMaxKeys];
    __shared__ float red[8];
    __shared__ float part[2][kHd];
    const int h = (int) blockIdx.x;
    const long long t = blockIdx.y;
    const int kvh = h / kGqa;
    const int tid = (int) threadIdx.x;
    const int lane = tid & 31;
    const int wv = tid >> 5;
    const long long pq = pos0 + t;
    long long lo = pq - n_swa + 1;
    lo = lo < 0 ? 0 : lo;
    lo = lo < pos0 - slots ? pos0 - slots : lo;  // only the ring's last `slots` positions
    const long long nr = pos0 - lo;              // ring keys
    const long long nk = nr + T;
    if (tid < kHd) {
        qs[tid] = q[(t * 32 + h) * kHd + tid];
    }
    __syncthreads();
    const int rs = kKvHeads * kHd;  // ring / block row: 8 KV heads
    for (long long j = wv; j < nk; j += 8) {
        float s;
        if (j < nr) {
            const long long p = lo + j;
            const long long slot = p % slots;
            const __half * kr = ring_k + slot * rs + kvh * kHd + lane * 4;
            s = __half2float(kr[0]) * qs[lane * 4] + __half2float(kr[1]) * qs[lane * 4 + 1] +
                __half2float(kr[2]) * qs[lane * 4 + 2] + __half2float(kr[3]) * qs[lane * 4 + 3];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                s += __shfl_xor(s, off);
            }
            s = tags[slot] == (int32_t) p ? s * scale : -FLT_MAX;
        } else {
            // the block's own keys, rounded to f16 as llama.cpp's cache holds them
            const float * kr = bk + (j - nr) * rs + kvh * kHd + lane * 4;
            s = 0.0f;
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                s += __half2float(__float2half(kr[d])) * qs[lane * 4 + d];
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                s += __shfl_xor(s, off);
            }
            s *= scale;
        }
        if (lane == 0) {
            sc[j] = s;
        }
    }
    __syncthreads();
    float m = -FLT_MAX;
    for (long long j = tid; j < nk; j += 256) {
        m = fmaxf(m, sc[j]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor(m, off));
    }
    if (lane == 0) {
        red[wv] = m;
    }
    __syncthreads();
    m = red[0];
    for (int i = 1; i < 8; ++i) {
        m = fmaxf(m, red[i]);
    }
    __syncthreads();
    float sum = 0.0f;
    for (long long j = tid; j < nk; j += 256) {
        const float e = sc[j] == -FLT_MAX ? 0.0f : expf(sc[j] - m);
        sc[j] = e;
        sum += e;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor(sum, off);
    }
    if (lane == 0) {
        red[wv] = sum;
    }
    __syncthreads();
    sum = 0.0f;
    for (int i = 0; i < 8; ++i) {
        sum += red[i];
    }
    const int d = tid & (kHd - 1);
    const int half = tid >> 7;
    float acc = 0.0f;
    for (long long j = half; j < nk; j += 2) {
        const float p = sc[j];
        if (p == 0.0f) {
            continue;
        }
        float vv;
        if (j < nr) {
            const long long slot = (lo + j) % slots;
            vv = __half2float(ring_v[slot * rs + kvh * kHd + d]);
        } else {
            vv = __half2float(__float2half(bv[(j - nr) * rs + kvh * kHd + d]));
        }
        acc = fmaf(p, vv, acc);
    }
    part[half][d] = acc;
    __syncthreads();
    if (tid < kHd) {
        out[(t * 32 + h) * kHd + tid] = (part[0][tid] + part[1][tid]) / sum;
    }
}

__global__ void conv_kernel(const float * __restrict__ x, const float * __restrict__ dyn,
                            const float * __restrict__ base, const long long n, const long long group,
                            const int side, float * __restrict__ out) {
    const long long t = blockIdx.y;
    const long long c = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) {
        return;
    }
    const long long groups = n / group;
    const long long g = c / group;
    const float * dr = dyn + t * groups * 4;  // (groups, 2 taps, 2 sides) per token
    float acc = 0.0f;
    for (int tap = 0; tap < 2; ++tap) {
        const float w = dr[g + groups * (tap + 2 * side)] + base[c + n * (tap + 2 * side)];
        const float xv = t - tap >= 0 ? x[(t - tap) * n + c] : 0.0f;
        const float term = w * xv;
        acc = tap == 0 ? term : acc + term;
    }
    out[t * n + c] = acc;
}

// One block per row, 256 threads: each keeps its top 16 (sorted), then 16
// rounds pick the best list head.
__global__ __launch_bounds__(256) void topk_kernel(const float * __restrict__ x, const long long n,
                                                   int32_t * __restrict__ ids, float * __restrict__ vals) {
    __shared__ float lv[256][kTopK];
    __shared__ int32_t li[256][kTopK];
    __shared__ float rv[256];
    __shared__ int rt[256];
    const float * row = x + (long long) blockIdx.x * n;
    const int tid = (int) threadIdx.x;
    float bv[kTopK];
    int32_t bi[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        bv[k] = -FLT_MAX;
        bi[k] = -1;
    }
    for (long long i = tid; i < n; i += 256) {
        const float v = row[i];
        if (v > bv[kTopK - 1]) {
            int k = kTopK - 1;
            while (k > 0 && bv[k - 1] < v) {
                bv[k] = bv[k - 1];
                bi[k] = bi[k - 1];
                --k;
            }
            bv[k] = v;
            bi[k] = (int32_t) i;
        }
    }
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        lv[tid][k] = bv[k];
        li[tid][k] = bi[k];
    }
    int head = 0;
    __syncthreads();
    for (int r = 0; r < kTopK; ++r) {
        rv[tid] = head < kTopK ? lv[tid][head] : -FLT_MAX;
        rt[tid] = tid;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s) {
                const float a = rv[tid];
                const float b = rv[tid + s];
                const bool take = b > a || (b == a && rt[tid + s] < rt[tid]);
                if (take) {
                    rv[tid] = b;
                    rt[tid] = rt[tid + s];
                }
            }
            __syncthreads();
        }
        const int win = rt[0];
        __syncthreads();
        if (tid == win) {
            ids[(long long) blockIdx.x * kTopK + r] = li[tid][head];
            vals[(long long) blockIdx.x * kTopK + r] = lv[tid][head];
            ++head;
        }
        __syncthreads();
    }
}

// One block per position i = 1 .. P, 256 threads = (pred a, succ b).
__global__ __launch_bounds__(256) void selector_kernel(const int32_t * __restrict__ cand,
                                                       const float * __restrict__ vals,
                                                       const float * __restrict__ gate, const int32_t anchor,
                                                       const qd::DecQ4k prev, const qd::DecQ4k next,
                                                       float * __restrict__ scores) {
    __shared__ float pr[kTopK][kRank + 1];
    __shared__ float nx[kTopK][kRank + 1];
    __shared__ float gs[kRank];
    const int i = (int) blockIdx.x + 1;
    const int tid = (int) threadIdx.x;
    gs[tid] = gate[(long long) i * kRank + tid];
    {
        const int task = tid & 127;
        const int r = task >> 3;
        const int s = task & 7;
        const bool is_next = tid >= 128;
        const int32_t id = is_next ? cand[(i - 1) * kTopK + r] : (i == 1 ? anchor : cand[(i - 2) * kTopK + r]);
        const qd::DecQ4k & dec = is_next ? next : prev;
        float v[32];
        dec.decode(dec.load(id, s), s, v);
        float (*dst)[kRank + 1] = is_next ? nx : pr;
#pragma unroll
        for (int e = 0; e < 32; ++e) {
            dst[r][s * 32 + e] = v[e];
        }
    }
    __syncthreads();
    const int a = tid >> 4;
    const int b = tid & 15;
    float acc = 0.0f;
    for (int r = 0; r < kRank; ++r) {
        acc = fmaf(pr[a][r] * gs[r], nx[b][r], acc);
    }
    scores[((long long) (i - 1) * kTopK + a) * kTopK + b] = acc + vals[(i - 1) * kTopK + b];
}

}  // namespace

bool dflash_capture(const float * x, const int64_t T, const int64_t n, const int64_t ld_x, __half * feat,
                    const int64_t ld_f, const int64_t col, hipStream_t stream) {
    if (x == nullptr || feat == nullptr || T <= 0 || n <= 0) {
        return false;
    }
    capture_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, stream>>>(x, n, ld_x, feat, ld_f, col);
    return hipGetLastError() == hipSuccess;
}

bool dflash_norm_rope(float * x, const int64_t T, const int64_t heads, const float * w, const float eps,
                      const int64_t pos0, const float theta, hipStream_t stream) {
    if (x == nullptr || T <= 0 || heads <= 0) {
        return false;
    }
    norm_rope_kernel<<<(unsigned) (T * heads), kHd, 0, stream>>>(x, heads, w, eps, pos0, theta);
    return hipGetLastError() == hipSuccess;
}

bool dflash_ring_put(const float * src, const int64_t T, const int64_t row, const int64_t pos0, __half * ring,
                     int32_t * tags, const int64_t slots, hipStream_t stream) {
    if (src == nullptr || ring == nullptr || T <= 0 || row <= 0 || slots <= 0 || pos0 < 0) {
        return false;
    }
    ring_put_kernel<<<dim3((unsigned) ((row + 255) / 256), (unsigned) T), 256, 0, stream>>>(src, row, pos0, ring,
                                                                                          tags, slots);
    return hipGetLastError() == hipSuccess;
}

bool dflash_attention(const float * q, const __half * ring_k, const __half * ring_v, const int32_t * tags,
                      const int64_t slots, const float * bk, const float * bv, const int64_t T, const int64_t pos0,
                      const int64_t n_swa, const float scale, float * out, hipStream_t stream) {
    if (q == nullptr || ring_k == nullptr || ring_v == nullptr || tags == nullptr || bk == nullptr ||
        bv == nullptr || out == nullptr || T <= 0 || T > 64 || slots > 2048 || n_swa <= 0) {
        return false;
    }
    attention_kernel<<<dim3(32, (unsigned) T), 256, 0, stream>>>(q, ring_k, ring_v, tags, slots, bk, bv, T, pos0,
                                                                 n_swa, scale, out);
    return hipGetLastError() == hipSuccess;
}

bool dflash_conv(const float * x, const float * dyn, const float * base, const int64_t T, const int64_t n,
                 const int64_t group, const int side, float * out, hipStream_t stream) {
    if (x == nullptr || dyn == nullptr || base == nullptr || out == nullptr || x == out || T <= 0 ||
        group <= 0 || n % group != 0 || side < 0 || side > 1) {
        return false;
    }
    conv_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, stream>>>(x, dyn, base, n, group, side,
                                                                                    out);
    return hipGetLastError() == hipSuccess;
}

bool dflash_topk(const float * x, const int64_t rows, const int64_t n, int32_t * ids, float * vals,
                 hipStream_t stream) {
    if (x == nullptr || ids == nullptr || vals == nullptr || rows <= 0 || n < kTopK) {
        return false;
    }
    topk_kernel<<<(unsigned) rows, 256, 0, stream>>>(x, n, ids, vals);
    return hipGetLastError() == hipSuccess;
}

bool dflash_selector(const int32_t * cand, const float * vals, const float * gate, const int32_t anchor,
                     const void * prev_tab, const void * next_tab, const int64_t vocab, const int64_t P,
                     float * scores, hipStream_t stream) {
    if (cand == nullptr || vals == nullptr || gate == nullptr || prev_tab == nullptr || next_tab == nullptr ||
        scores == nullptr || P <= 0) {
        return false;
    }
    const auto prev = qd::DecQ4k::make(static_cast<const uint8_t *>(prev_tab), vocab, 1);
    const auto next = qd::DecQ4k::make(static_cast<const uint8_t *>(next_tab), vocab, 1);
    selector_kernel<<<(unsigned) P, 256, 0, stream>>>(cand, vals, gate, anchor, prev, next, scores);
    return hipGetLastError() == hipSuccess;
}

} // namespace omph::kernels
