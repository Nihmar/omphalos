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

// The token-parallel chunk (#96): delta_chunk_kernel cut in four. Every float
// operation and reduction order is the single-step kernel's, so the result is
// still bit-identical to gdn_step per token. Work layout (gdn_work_floats):
// cv (tokens, channels) the conv + silu of q | k | v, q and k L2-normed in
// place; vo (tokens, heads, 128) the delta rule's output; g (tokens, heads, 2)
// beta | decay.
struct ChunkWork {
    float * cv;
    float * vo;
    float * g;
};

__device__ inline ChunkWork chunk_work(const GdnStep & a, const long long heads) {
    ChunkWork w;
    w.cv = a.work;
    w.vo = w.cv + a.tokens * a.channels;
    w.g = w.vo + a.tokens * heads * kDeltaS;
    return w;
}

// 1a. Conv + silu of every channel and token; grid (channels / 128, tokens),
// 128 threads: one head's 128 channels of one token, so the q and k blocks
// also L2-normalize it (four wave partials summed in gdn_step's order). The
// first token's blocks shift the conv state (and record its history).
__global__ __launch_bounds__(kDeltaS) void delta_conv_kernel(const GdnStep a) {
    __shared__ float red[4];
    const long long t = blockIdx.y;
    const int tid = (int) threadIdx.x;
    const long long c = (long long) blockIdx.x * kDeltaS + tid;
    const long long rows = a.conv_k - 1;
    const auto in = [&](const long long r) {
        return r < rows ? a.conv_cur[r * a.channels + c] : a.qkv[(r - rows) * a.channels + c];
    };
    float acc = 0.0f;
    for (long long j = 0; j < a.conv_k; ++j) {
        acc = fmaf(a.conv_w[c * a.conv_k + j], in(t + j), acc);
    }
    float y = acc / (1.0f + expf(-acc));
    if (t == 0) {
        for (long long r = 0; r < rows; ++r) {
            a.conv_new[r * a.channels + c] = in(a.tokens + r);
        }
        if (a.conv_hist != nullptr) {
            for (long long r = 0; r < rows + a.tokens; ++r) {
                a.conv_hist[r * a.channels + c] = in(r);
            }
        }
    }
    const long long qk = a.q_dims + a.kv_dims;
    if (c < qk) {  // uniform per block: q or k of one key head
        float sq = y * y;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sq += __shfl_xor(sq, off);
        }
        if ((tid & 31) == 0) {
            red[tid >> 5] = sq;
        }
        __syncthreads();
        const float tot = red[0] + red[1] + red[2] + red[3];
        y = y * (1.0f / sqrtf(tot / (float) kDeltaS + a.eps_l2)) * a.l2_scale;
        if (a.replay != nullptr && c >= a.q_dims) {  // the token's k, for a rollback
            const long long heads = (a.channels - qk) / kDeltaS;
            a.replay[t * (heads + a.n_kh * kDeltaS + heads * kDeltaS) + heads + c - a.q_dims] = y;
        }
    }
    a.work[t * a.channels + c] = y;
}

// 1b. The gates: grid (heads, ceil(tokens / kGateTokens)), the BF16 beta /
// alpha rows (2 x 10 KB per head) read once for the block's tokens, each
// token's dot in gdn_step's order.
constexpr int kGateTokens = 16;

__global__ __launch_bounds__(256) void delta_gate_kernel(const GdnStep a) {
    __shared__ float gred[2][kGateTokens][8];
    const long long h = blockIdx.x;
    const long long heads = gridDim.x;
    const long long t0 = (long long) blockIdx.y * kGateTokens;
    const int nt = (int) (a.tokens - t0 < kGateTokens ? a.tokens - t0 : kGateTokens);
    const int tid = (int) threadIdx.x;
    float pb[kGateTokens];
    float pa[kGateTokens];
#pragma unroll
    for (int j = 0; j < kGateTokens; ++j) {
        pb[j] = 0.0f;
        pa[j] = 0.0f;
    }
    // a partial group reads the last token again for its missing rows (never stored)
    const int k_in = (int) a.k_in;
    const uint16_t * __restrict__ wbr = a.w_beta + h * k_in;
    const uint16_t * __restrict__ war = a.w_alpha + h * k_in;
    const __half * __restrict__ xr[kGateTokens];
#pragma unroll
    for (int j = 0; j < kGateTokens; ++j) {
        xr[j] = a.x16 + (t0 + (j < nt ? j : nt - 1)) * a.k_in;
    }
    for (int i = tid; i < k_in; i += 256) {
        const float wb = __uint_as_float((uint32_t) wbr[i] << 16);
        const float wa = __uint_as_float((uint32_t) war[i] << 16);
#pragma unroll
        for (int j = 0; j < kGateTokens; ++j) {
            const float xi = __half2float(xr[j][i]);
            pb[j] = fmaf(wb, xi, pb[j]);
            pa[j] = fmaf(wa, xi, pa[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < kGateTokens; ++j) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            pb[j] += __shfl_xor(pb[j], off);
            pa[j] += __shfl_xor(pa[j], off);
        }
        if ((tid & 31) == 0) {
            gred[0][j][tid >> 5] = pb[j];
            gred[1][j][tid >> 5] = pa[j];
        }
    }
    __syncthreads();
    if (tid < nt) {
        const long long t = t0 + tid;
        float beta_raw = 0.0f;
        float alpha_raw = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            beta_raw += gred[0][tid][i];
            alpha_raw += gred[1][tid][i];
        }
        const float b = 1.0f / (1.0f + expf(-beta_raw));
        const float ar = alpha_raw + a.dt_bias[h];
        const float decay = expf((log1pf(expf(-fabsf(ar))) + fmaxf(ar, 0.0f)) * a.ssm_a[h]);
        const ChunkWork w = chunk_work(a, heads);
        w.g[(t * heads + h) * 2] = b;
        w.g[(t * heads + h) * 2 + 1] = decay;
        if (a.replay != nullptr) {
            a.replay[t * (heads + a.n_kh * kDeltaS + heads * kDeltaS) + h] = decay;
        }
    }
}

// 2. The recurrence: grid (heads, 128 / kRecCols), one lane per value column
// holding all 128 rows of it. A column's update needs only its own S^T k and
// S^T q, so no lane talks to another; the two accumulators (rows 0..63 and
// 64..127) are summed at the end, exactly gdn_step's part[col] + part[s + col].
// k | q are the same for the whole wave: the restrict pointers let them come
// through scalar loads (LDS staging was 2.7x slower, #96).
constexpr int kRecCols = 64;

__global__ __launch_bounds__(kRecCols) void delta_rec_kernel(
    const float * __restrict__ cv, const float * __restrict__ gates, float * __restrict__ vo,
    float * __restrict__ replay, const float * state, float * state_out,  // may alias
    const long long tokens, const long long n_kh, const long long channels, const float l2_scale) {
    const long long h = blockIdx.x;
    const long long heads = gridDim.x;
    const int col = (int) (blockIdx.y * kRecCols + threadIdx.x);
    const long long kh = h % n_kh;
    float st[kDeltaS];
#pragma unroll
    for (int r = 0; r < kDeltaS; ++r) {
        st[r] = state[(h * kDeltaS + r) * kDeltaS + col];
    }
    const long long rec_stride = heads + n_kh * kDeltaS + heads * kDeltaS;
    for (long long t = 0; t < tokens; ++t) {
        const float b = gates[(t * heads + h) * 2];
        const float decay = gates[(t * heads + h) * 2 + 1];
        const float * qq = cv + t * channels + kh * kDeltaS;
        const float * kk = qq + n_kh * kDeltaS;
        float a0 = 0.0f;
        float a1 = 0.0f;
#pragma unroll
        for (int r = 0; r < kDeltaRows; ++r) {
            st[r] *= decay;
            a0 = fmaf(st[r], kk[r], a0);
            st[kDeltaRows + r] *= decay;
            a1 = fmaf(st[kDeltaRows + r], kk[kDeltaRows + r], a1);
        }
        const float dc = (cv[t * channels + 2 * n_kh * kDeltaS + h * kDeltaS + col] - (a0 + a1)) * b;
        float c0 = 0.0f;
        float c1 = 0.0f;
#pragma unroll
        for (int r = 0; r < kDeltaRows; ++r) {
            st[r] = fmaf(kk[r], dc, st[r]);
            c0 = fmaf(st[r], qq[r], c0);
            st[kDeltaRows + r] = fmaf(kk[kDeltaRows + r], dc, st[kDeltaRows + r]);
            c1 = fmaf(st[kDeltaRows + r], qq[kDeltaRows + r], c1);
        }
        vo[(t * heads + h) * kDeltaS + col] = l2_scale * (c0 + c1);
        if (replay != nullptr) {
            replay[t * rec_stride + heads + n_kh * kDeltaS + h * kDeltaS + col] = dc;
        }
    }
#pragma unroll
    for (int r = 0; r < kDeltaS; ++r) {
        state_out[(h * kDeltaS + r) * kDeltaS + col] = st[r];
    }
}

// 3. The gated RMSNorm per (head, token), as f16: grid (heads, tokens), 128 threads.
__global__ __launch_bounds__(kDeltaS) void delta_post_kernel(const GdnStep a) {
    __shared__ float wsq[4];
    const long long h = blockIdx.x;
    const long long t = blockIdx.y;
    const long long heads = gridDim.x;
    const int col = (int) threadIdx.x;
    const ChunkWork w = chunk_work(a, heads);
    const float ov = w.vo[(t * heads + h) * kDeltaS + col];
    float osq = ov * ov;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        osq += __shfl_xor(osq, off);
    }
    if ((col & 31) == 0) {
        wsq[col >> 5] = osq;
    }
    __syncthreads();
    const float inv =
        1.0f / sqrtf((wsq[0] + wsq[1] + wsq[2] + wsq[3]) / (float) kDeltaS + a.eps_norm);
    const float zn = a.z[(t * heads + h) * kDeltaS + col];
    a.out16[(t * heads + h) * kDeltaS + col] =
        __float2half(ov * inv * a.norm_w[col] * (zn / (1.0f + expf(-zn))));
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
    if (a.work == nullptr) {
        delta_chunk_kernel<<<(unsigned) heads, 256, 0, stream>>>(a);
        return hipGetLastError() == hipSuccess;
    }
    const dim3 per_token((unsigned) heads, (unsigned) a.tokens);
    delta_conv_kernel<<<dim3((unsigned) (a.channels / kDeltaS), (unsigned) a.tokens), kDeltaS, 0,
                        stream>>>(a);
    delta_gate_kernel<<<dim3((unsigned) heads,
                             (unsigned) ((a.tokens + kGateTokens - 1) / kGateTokens)),
                        256, 0, stream>>>(a);
    float * vo = a.work + a.tokens * a.channels;
    const float * gates = vo + a.tokens * heads * kDeltaS;
    delta_rec_kernel<<<dim3((unsigned) heads, kDeltaS / kRecCols), kRecCols, 0, stream>>>(
        a.work, gates, vo, a.replay, a.state, a.state_out != nullptr ? a.state_out : a.state,
        a.tokens, a.n_kh, a.channels, a.l2_scale);
    delta_post_kernel<<<per_token, kDeltaS, 0, stream>>>(a);
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
