// Gated delta net kernels (PLAN.md §10.2, autoregressive path).
#include "kernels/gdn.hh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace omph::kernels {
namespace {

// Fused one-token delta rule: one workgroup per head, 256 threads (two per
// column, each covering half of the rows). Each thread keeps its 64 state
// elements in registers, so the state is read once and written once (the
// earlier form walked it five times: decay + S^T k, update, S^T q, #66).
//
//   tid = 0..255; col = tid & (s-1); half = tid >> 7 -> rows [half*s/2, ...)
//   decay, then the S^T k partials (combined across the halves in LDS), d,
//   the rank-1 update, and the S^T q partials.
constexpr int kDeltaS = 128;              // state size; the wrapper checks it
constexpr int kDeltaRows = kDeltaS / 2;   // rows per thread

__global__ __launch_bounds__(256) void delta_step_kernel(const GdnStep a) {
    __shared__ float part[2 * kDeltaS];
    __shared__ float dd[kDeltaS];
    __shared__ float kq[2 * kDeltaS];  // k then q of this head's key group
    __shared__ float red[8];
    const long long h = blockIdx.x;
    float * S = a.state + h * kDeltaS * kDeltaS;
    const int tid = (int) threadIdx.x;
    const int col = tid & (kDeltaS - 1);
    const int half = tid >> 7;
    const int lo = half * kDeltaRows;
    const long long kh = h % a.n_kh;

    // conv1d + silu of this head's inputs, straight from the qkv projection
    // (#83; formerly conv_silu_split_l2): threads 0..127 make k, 128..255 q of
    // key head kh, and threads 0..127 also v of value head h. Input row r of
    // the conv is conv_cur row r for r < K - 1, else qkv row r - (K - 1).
    const long long rows = a.conv_k - 1;
    const auto conv = [&](const long long c) {
        float acc = 0.0f;
        for (long long j = 0; j < a.conv_k; ++j) {
            const long long r = a.t + j;
            const float xv = r < rows ? a.conv_cur[r * a.channels + c]
                                      : a.qkv[(r - rows) * a.channels + c];
            acc = fmaf(a.conv_w[c * a.conv_k + j], xv, acc);
        }
        return acc / (1.0f + expf(-acc));
    };
    const long long c_qk = half == 0 ? a.q_dims + kh * kDeltaS + col : kh * kDeltaS + col;
    const long long c_v = a.q_dims + a.kv_dims + h * kDeltaS + col;
    float y = conv(c_qk);
    const float vval = half == 0 ? conv(c_v) : 0.0f;
    if (a.t == 0) {  // shift the conv state once per launch sequence, by its owners
        const auto shift = [&](const long long c) {
            for (long long r = 0; r < rows; ++r) {
                const long long src = a.tokens + r;
                a.conv_new[r * a.channels + c] = src < rows ? a.conv_cur[src * a.channels + c]
                                                            : a.qkv[(src - rows) * a.channels + c];
            }
            if (a.conv_hist != nullptr) {  // the whole input history, for a rollback
                for (long long r = 0; r < rows + a.tokens; ++r) {
                    a.conv_hist[r * a.channels + c] = r < rows ? a.conv_cur[r * a.channels + c]
                                                               : a.qkv[(r - rows) * a.channels + c];
                }
            }
        };
        if (h < a.n_kh) {
            shift(c_qk);
        }
        if (half == 0) {
            shift(c_v);
        }
    }
    // L2 normalization of k (waves 0-3) and q (waves 4-7): rms_norm(eps) * scale
    float sq = y * y;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sq += __shfl_xor(sq, off);
    }
    if ((tid & 31) == 0) {
        red[tid >> 5] = sq;
    }
    __syncthreads();
    const float tot = red[half * 4] + red[half * 4 + 1] + red[half * 4 + 2] + red[half * 4 + 3];
    y = y * (1.0f / sqrtf(tot / (float) kDeltaS + a.eps_l2)) * a.l2_scale;
    kq[tid] = y;  // k then q, as the rows of the update read them
    // The raw gates: this head's rows of the BF16 beta / alpha projections dotted
    // with the normed input (#88; formerly two GEMV launches on the side stream).
    // A bf16 is the top half of an f32.
    __shared__ float gred[2][8];
    float pb = 0.0f;
    float pa = 0.0f;
    for (long long i = tid; i < a.k_in; i += 256) {
        const float xi = __half2float(a.x16[i]);
        pb = fmaf(__uint_as_float((uint32_t) a.w_beta[h * a.k_in + i] << 16), xi, pb);
        pa = fmaf(__uint_as_float((uint32_t) a.w_alpha[h * a.k_in + i] << 16), xi, pa);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        pb += __shfl_xor(pb, off);
        pa += __shfl_xor(pa, off);
    }
    if ((tid & 31) == 0) {
        gred[0][tid >> 5] = pb;
        gred[1][tid >> 5] = pa;
    }
    __syncthreads();
    float beta_raw = 0.0f;
    float alpha_raw = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        beta_raw += gred[0][w];
        alpha_raw += gred[1][w];
    }
    // b = sigmoid(beta), decay = exp(softplus(alpha + dt_bias) * A) (#78)
    const float b = 1.0f / (1.0f + expf(-beta_raw));
    const float ar = alpha_raw + a.dt_bias[h];
    const float decay = expf((log1pf(expf(-fabsf(ar))) + fmaxf(ar, 0.0f)) * a.ssm_a[h]);
    float * rec = a.replay != nullptr
                      ? a.replay + a.t * (gridDim.x + a.n_kh * kDeltaS + gridDim.x * kDeltaS)
                      : nullptr;
    if (rec != nullptr) {
        if (tid == 0) {
            rec[h] = decay;
        }
        if (h < a.n_kh && half == 0) {  // block h owns key head h's k (kh == h here)
            rec[gridDim.x + h * kDeltaS + col] = y;
        }
    }

    float st[kDeltaRows];
#pragma unroll
    for (int r = 0; r < kDeltaRows; ++r) {
        st[r] = S[(lo + r) * kDeltaS + col];
    }
    __syncthreads();  // kq
    float acc = 0.0f;
#pragma unroll
    for (int r = 0; r < kDeltaRows; ++r) {
        st[r] *= decay;
        acc = fmaf(st[r], kq[lo + r], acc);
    }
    part[half * kDeltaS + col] = acc;
    __syncthreads();
    if (half == 0) {
        dd[col] = (vval - (part[col] + part[kDeltaS + col])) * b;
        if (rec != nullptr) {
            rec[gridDim.x + a.n_kh * kDeltaS + h * kDeltaS + col] = dd[col];
        }
    }
    __syncthreads();
    const float dc = dd[col];
    float acc2 = 0.0f;
    float * S_out = a.state_out != nullptr ? a.state_out + h * kDeltaS * kDeltaS : S;
#pragma unroll
    for (int r = 0; r < kDeltaRows; ++r) {
        st[r] = fmaf(kq[lo + r], dc, st[r]);
        acc2 = fmaf(st[r], kq[kDeltaS + lo + r], acc2);
        S_out[(lo + r) * kDeltaS + col] = st[r];
    }
    __syncthreads();  // part is reused
    part[half * kDeltaS + col] = acc2;
    __syncthreads();
    const float ov = half == 0 ? a.l2_scale * (part[col] + part[kDeltaS + col]) : 0.0f;
    // Gated RMSNorm of this head's output, the input of the ssm_out GEMV (#83;
    // formerly gated_norm_f16): y = o / sqrt(mean(o^2) + eps) * w * silu(z).
    __shared__ float wsq[8];
    float osq = ov * ov;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        osq += __shfl_xor(osq, off);
    }
    if ((tid & 31) == 0) {
        wsq[tid >> 5] = osq;
    }
    __syncthreads();
    if (half == 0) {
        const float inv =
            1.0f / sqrtf((wsq[0] + wsq[1] + wsq[2] + wsq[3]) / (float) kDeltaS + a.eps_norm);
        const float zn = a.z[h * kDeltaS + col];
        a.out16[h * kDeltaS + col] =
            __float2half(ov * inv * a.norm_w[col] * (zn / (1.0f + expf(-zn))));
    }
}


// gdn_step over a whole chunk in one launch (#96): the single-step kernel's
// body per token, in order, with the state kept in registers between tokens.
// Same float operations in the same order: the result equals T launches of
// the single-step kernel, bit for bit.
__global__ __launch_bounds__(256) void delta_chunk_kernel(GdnStep a) {
    __shared__ float part[2 * kDeltaS];
    __shared__ float dd[kDeltaS];
    __shared__ float kq[2 * kDeltaS];  // k then q of this head's key group
    __shared__ float red[8];
    const long long h = blockIdx.x;
    float * S = a.state + h * kDeltaS * kDeltaS;
    const int tid = (int) threadIdx.x;
    const int col = tid & (kDeltaS - 1);
    const int half = tid >> 7;
    const int lo = half * kDeltaRows;
    const long long kh = h % a.n_kh;

    // conv1d + silu of this head's inputs, straight from the qkv projection
    // (#83; formerly conv_silu_split_l2): threads 0..127 make k, 128..255 q of
    // key head kh, and threads 0..127 also v of value head h. Input row r of
    // the conv is conv_cur row r for r < K - 1, else qkv row r - (K - 1).
    const long long rows = a.conv_k - 1;
    const auto conv = [&](const long long c) {
        float acc = 0.0f;
        for (long long j = 0; j < a.conv_k; ++j) {
            const long long r = a.t + j;
            const float xv = r < rows ? a.conv_cur[r * a.channels + c]
                                      : a.qkv[(r - rows) * a.channels + c];
            acc = fmaf(a.conv_w[c * a.conv_k + j], xv, acc);
        }
        return acc / (1.0f + expf(-acc));
    };
    const long long c_qk = half == 0 ? a.q_dims + kh * kDeltaS + col : kh * kDeltaS + col;
    const long long c_v = a.q_dims + a.kv_dims + h * kDeltaS + col;
    // The chunk's tokens in order, the state in registers throughout (#96):
    // read once here, written once at the end, instead of a launch per token.
    float st[kDeltaRows];
#pragma unroll
    for (int r = 0; r < kDeltaRows; ++r) {
        st[r] = S[(lo + r) * kDeltaS + col];
    }
    const __half * x16_0 = a.x16;
    const float * z_0 = a.z;
    __half * out16_0 = a.out16;
    const long long v_dims = (long long) gridDim.x * kDeltaS;
    for (long long t = 0; t < a.tokens; ++t) {
        a.t = t;
        a.x16 = x16_0 + t * a.k_in;
        a.z = z_0 + t * v_dims;
        a.out16 = out16_0 + t * v_dims;
        __syncthreads();  // the shared buffers of the previous token are free
        float y = conv(c_qk);
        const float vval = half == 0 ? conv(c_v) : 0.0f;
        if (a.t == 0) {  // shift the conv state once per launch sequence, by its owners
            const auto shift = [&](const long long c) {
                for (long long r = 0; r < rows; ++r) {
                    const long long src = a.tokens + r;
                    a.conv_new[r * a.channels + c] = src < rows ? a.conv_cur[src * a.channels + c]
                                                                : a.qkv[(src - rows) * a.channels + c];
                }
                if (a.conv_hist != nullptr) {  // the whole input history, for a rollback
                    for (long long r = 0; r < rows + a.tokens; ++r) {
                        a.conv_hist[r * a.channels + c] = r < rows ? a.conv_cur[r * a.channels + c]
                                                                   : a.qkv[(r - rows) * a.channels + c];
                    }
                }
            };
            if (h < a.n_kh) {
                shift(c_qk);
            }
            if (half == 0) {
                shift(c_v);
            }
        }
        // L2 normalization of k (waves 0-3) and q (waves 4-7): rms_norm(eps) * scale
        float sq = y * y;
    #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sq += __shfl_xor(sq, off);
        }
        if ((tid & 31) == 0) {
            red[tid >> 5] = sq;
        }
        __syncthreads();
        const float tot = red[half * 4] + red[half * 4 + 1] + red[half * 4 + 2] + red[half * 4 + 3];
        y = y * (1.0f / sqrtf(tot / (float) kDeltaS + a.eps_l2)) * a.l2_scale;
        kq[tid] = y;  // k then q, as the rows of the update read them
        // The raw gates: this head's rows of the BF16 beta / alpha projections dotted
        // with the normed input (#88; formerly two GEMV launches on the side stream).
        // A bf16 is the top half of an f32.
        __shared__ float gred[2][8];
        float pb = 0.0f;
        float pa = 0.0f;
        for (long long i = tid; i < a.k_in; i += 256) {
            const float xi = __half2float(a.x16[i]);
            pb = fmaf(__uint_as_float((uint32_t) a.w_beta[h * a.k_in + i] << 16), xi, pb);
            pa = fmaf(__uint_as_float((uint32_t) a.w_alpha[h * a.k_in + i] << 16), xi, pa);
        }
    #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            pb += __shfl_xor(pb, off);
            pa += __shfl_xor(pa, off);
        }
        if ((tid & 31) == 0) {
            gred[0][tid >> 5] = pb;
            gred[1][tid >> 5] = pa;
        }
        __syncthreads();
        float beta_raw = 0.0f;
        float alpha_raw = 0.0f;
    #pragma unroll
        for (int w = 0; w < 8; ++w) {
            beta_raw += gred[0][w];
            alpha_raw += gred[1][w];
        }
        // b = sigmoid(beta), decay = exp(softplus(alpha + dt_bias) * A) (#78)
        const float b = 1.0f / (1.0f + expf(-beta_raw));
        const float ar = alpha_raw + a.dt_bias[h];
        const float decay = expf((log1pf(expf(-fabsf(ar))) + fmaxf(ar, 0.0f)) * a.ssm_a[h]);
        float * rec = a.replay != nullptr
                          ? a.replay + a.t * (gridDim.x + a.n_kh * kDeltaS + gridDim.x * kDeltaS)
                          : nullptr;
        if (rec != nullptr) {
            if (tid == 0) {
                rec[h] = decay;
            }
            if (h < a.n_kh && half == 0) {  // block h owns key head h's k (kh == h here)
                rec[gridDim.x + h * kDeltaS + col] = y;
            }
        }

        __syncthreads();  // kq
        float acc = 0.0f;
    #pragma unroll
        for (int r = 0; r < kDeltaRows; ++r) {
            st[r] *= decay;
            acc = fmaf(st[r], kq[lo + r], acc);
        }
        part[half * kDeltaS + col] = acc;
        __syncthreads();
        if (half == 0) {
            dd[col] = (vval - (part[col] + part[kDeltaS + col])) * b;
            if (rec != nullptr) {
                rec[gridDim.x + a.n_kh * kDeltaS + h * kDeltaS + col] = dd[col];
            }
        }
        __syncthreads();
        const float dc = dd[col];
        float acc2 = 0.0f;
    #pragma unroll
        for (int r = 0; r < kDeltaRows; ++r) {
            st[r] = fmaf(kq[lo + r], dc, st[r]);
            acc2 = fmaf(st[r], kq[kDeltaS + lo + r], acc2);
        }
        __syncthreads();  // part is reused
        part[half * kDeltaS + col] = acc2;
        __syncthreads();
        const float ov = half == 0 ? a.l2_scale * (part[col] + part[kDeltaS + col]) : 0.0f;
        // Gated RMSNorm of this head's output, the input of the ssm_out GEMV (#83;
        // formerly gated_norm_f16): y = o / sqrt(mean(o^2) + eps) * w * silu(z).
        __shared__ float wsq[8];
        float osq = ov * ov;
    #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            osq += __shfl_xor(osq, off);
        }
        if ((tid & 31) == 0) {
            wsq[tid >> 5] = osq;
        }
        __syncthreads();
        if (half == 0) {
            const float inv =
                1.0f / sqrtf((wsq[0] + wsq[1] + wsq[2] + wsq[3]) / (float) kDeltaS + a.eps_norm);
            const float zn = a.z[h * kDeltaS + col];
            a.out16[h * kDeltaS + col] =
                __float2half(ov * inv * a.norm_w[col] * (zn / (1.0f + expf(-zn))));
        }
    }
    float * S_out = a.state_out != nullptr ? a.state_out + h * kDeltaS * kDeltaS : S;
#pragma unroll
    for (int r = 0; r < kDeltaRows; ++r) {
        S_out[(lo + r) * kDeltaS + col] = st[r];
    }
}

// The rank-1 updates of recorded tokens, on the state of the step before them:
// gdn_step's thread layout and float operations (decay, then fma with k and d).
__global__ __launch_bounds__(256) void gdn_replay_kernel(float * __restrict__ state,
                                                         const float * __restrict__ replay,
                                                         const long long tokens,
                                                         const long long n_kh) {
    const long long h = blockIdx.x;
    const long long heads = gridDim.x;
    const int tid = (int) threadIdx.x;
    const int col = tid & (kDeltaS - 1);
    const int lo = (tid >> 7) * kDeltaRows;
    const long long kh = h % n_kh;
    const long long stride = heads + n_kh * kDeltaS + heads * kDeltaS;
    float * S = state + h * kDeltaS * kDeltaS;
    float st[kDeltaRows];
#pragma unroll
    for (int r = 0; r < kDeltaRows; ++r) {
        st[r] = S[(lo + r) * kDeltaS + col];
    }
    for (long long t = 0; t < tokens; ++t) {
        const float * rec = replay + t * stride;
        const float decay = rec[h];
        const float * k = rec + heads + kh * kDeltaS;
        const float dc = rec[heads + n_kh * kDeltaS + h * kDeltaS + col];
#pragma unroll
        for (int r = 0; r < kDeltaRows; ++r) {
            st[r] *= decay;
        }
#pragma unroll
        for (int r = 0; r < kDeltaRows; ++r) {
            st[r] = fmaf(k[lo + r], dc, st[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < kDeltaRows; ++r) {
        S[(lo + r) * kDeltaS + col] = st[r];
    }
}

__global__ void gdn_conv_select_kernel(const float * __restrict__ hist, float * __restrict__ dst,
                                       const long long accepted, const long long rows,
                                       const long long channels) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * channels) {
        dst[i] = hist[accepted * channels + i];
    }
}

} // namespace

bool gdn_replay(float * state, const float * replay, const int64_t tokens, const int64_t heads,
                const int64_t n_kh, hipStream_t stream) {
    if (state == nullptr || replay == nullptr || tokens < 0 || heads <= 0 || n_kh <= 0) {
        return false;
    }
    if (tokens == 0) {
        return true;
    }
    gdn_replay_kernel<<<(unsigned) heads, 256, 0, stream>>>(state, replay, (long long) tokens,
                                                           (long long) n_kh);
    return hipGetLastError() == hipSuccess;
}

bool gdn_conv_select(const float * conv_hist, float * conv_dst, const int64_t accepted,
                     const int64_t conv_k, const int64_t channels, hipStream_t stream) {
    if (conv_hist == nullptr || conv_dst == nullptr || accepted < 0 || conv_k <= 1 ||
        channels <= 0) {
        return false;
    }
    const long long n = (long long) (conv_k - 1) * channels;
    gdn_conv_select_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(
        conv_hist, conv_dst, (long long) accepted, (long long) (conv_k - 1), (long long) channels);
    return hipGetLastError() == hipSuccess;
}

bool gdn_chunk(const GdnStep & a, const int64_t heads, hipStream_t stream) {
    if (a.state == nullptr || a.qkv == nullptr || a.conv_w == nullptr || a.conv_cur == nullptr ||
        a.conv_new == nullptr || a.x16 == nullptr || a.w_beta == nullptr ||
        a.w_alpha == nullptr || a.k_in <= 0 || a.dt_bias == nullptr ||
        a.ssm_a == nullptr || a.z == nullptr || a.norm_w == nullptr || a.out16 == nullptr ||
        heads <= 0 || a.n_kh <= 0 || a.conv_k <= 1 || a.tokens <= 0 ||
        a.q_dims != a.n_kh * kDeltaS || a.kv_dims != a.q_dims ||
        a.channels != a.q_dims + a.kv_dims + heads * kDeltaS) {
        return false;
    }
    delta_chunk_kernel<<<(unsigned) heads, 256, 0, stream>>>(a);
    return hipGetLastError() == hipSuccess;
}

bool gdn_step(const GdnStep & a, const int64_t heads, hipStream_t stream) {
    if (a.state == nullptr || a.qkv == nullptr || a.conv_w == nullptr || a.conv_cur == nullptr ||
        a.conv_new == nullptr || a.x16 == nullptr || a.w_beta == nullptr ||
        a.w_alpha == nullptr || a.k_in <= 0 || a.dt_bias == nullptr ||
        a.ssm_a == nullptr || a.z == nullptr || a.norm_w == nullptr || a.out16 == nullptr ||
        heads <= 0 || a.n_kh <= 0 || a.conv_k <= 1 || a.t < 0 || a.t >= a.tokens ||
        a.q_dims != a.n_kh * kDeltaS || a.kv_dims != a.q_dims ||
        a.channels != a.q_dims + a.kv_dims + heads * kDeltaS) {
        return false;  // written for s = 128 and the q | k | v channel layout
    }
    delta_step_kernel<<<(unsigned) heads, 256, 0, stream>>>(a);
    return hipGetLastError() == hipSuccess;
}

} // namespace omph::kernels
