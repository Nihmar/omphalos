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

// 1a. Conv + silu of every channel and token; grid (channels / 128, ceil(tokens /
// kConvTokens)), 128 threads: one head's 128 channels of kConvTokens tokens (the
// conv weights and the input rows loaded once for them, #240), so the q and k
// blocks also L2-normalize each token (four wave partials summed in gdn_step's
// order). The first token's block shifts the conv state (and records its history).
constexpr int kConvTokens = 8;

__global__ __launch_bounds__(kDeltaS) void delta_conv_kernel(const GdnStep a) {
    constexpr int kMaxConv = 4;  // conv_k (checked at launch)
    __shared__ float red[kConvTokens][4];
    const long long t0 = (long long) blockIdx.y * kConvTokens;
    const int nt = (int) (a.tokens - t0 < kConvTokens ? a.tokens - t0 : kConvTokens);
    const int tid = (int) threadIdx.x;
    const long long c = (long long) blockIdx.x * kDeltaS + tid;
    const long long rows = a.conv_k - 1;
    const auto in = [&](const long long r) {
        return r < rows ? a.conv_cur[r * a.channels + c] : a.qkv[(r - rows) * a.channels + c];
    };
    float w[kMaxConv];
#pragma unroll
    for (int j = 0; j < kMaxConv; ++j) {
        w[j] = a.conv_w[c * kMaxConv + j];
    }
    float x[kConvTokens + kMaxConv - 1];
#pragma unroll
    for (int r = 0; r < kConvTokens + kMaxConv - 1; ++r) {
        x[r] = r < nt + kMaxConv - 1 ? in(t0 + r) : 0.0f;
    }
    float y[kConvTokens];
#pragma unroll
    for (int i = 0; i < kConvTokens; ++i) {
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < kMaxConv; ++j) {
            acc = fmaf(w[j], x[i + j], acc);
        }
        y[i] = acc / (1.0f + expf(-acc));
    }
    if (t0 == 0) {
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
#pragma unroll
        for (int i = 0; i < kConvTokens; ++i) {
            float sq = y[i] * y[i];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                sq += __shfl_xor(sq, off);
            }
            if ((tid & 31) == 0) {
                red[i][tid >> 5] = sq;
            }
        }
        __syncthreads();
        const long long heads = (a.channels - qk) / kDeltaS;
#pragma unroll
        for (int i = 0; i < kConvTokens; ++i) {
            const float tot = red[i][0] + red[i][1] + red[i][2] + red[i][3];
            y[i] = y[i] * (1.0f / sqrtf(tot / (float) kDeltaS + a.eps_l2)) * a.l2_scale;
            if (a.replay != nullptr && c >= a.q_dims && i < nt) {  // the token's k, for a rollback
                a.replay[(t0 + i) * (heads + a.n_kh * kDeltaS + heads * kDeltaS) + heads + c - a.q_dims] = y[i];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kConvTokens; ++i) {
        if (i < nt) {
            a.work[(t0 + i) * a.channels + c] = y[i];
        }
    }
}

// 1b. The gates: grid (heads / HB, ceil(tokens / kGateTokens)), HB heads' BF16
// beta / alpha rows (2 x 10 KB each) read once for the block's tokens and each
// token's x read once for the HB heads (HB = 1 re-read x for every head: ~250 MB
// of L2 traffic per 512-token chunk, #240); each dot in gdn_step's order.
constexpr int kGateTokens = 16;

template <int HB>
__global__ __launch_bounds__(256) void delta_gate_kernel(const GdnStep a) {
    __shared__ float gred[2][HB][kGateTokens][8];
    const long long h0 = (long long) blockIdx.x * HB;
    const long long heads = (long long) gridDim.x * HB;
    const long long t0 = (long long) blockIdx.y * kGateTokens;
    const int nt = (int) (a.tokens - t0 < kGateTokens ? a.tokens - t0 : kGateTokens);
    const int tid = (int) threadIdx.x;
    float pb[HB][kGateTokens];
    float pa[HB][kGateTokens];
#pragma unroll
    for (int hb = 0; hb < HB; ++hb) {
#pragma unroll
        for (int j = 0; j < kGateTokens; ++j) {
            pb[hb][j] = 0.0f;
            pa[hb][j] = 0.0f;
        }
    }
    // a partial group reads the last token again for its missing rows (never stored)
    const int k_in = (int) a.k_in;
    const __half * __restrict__ xr[kGateTokens];
#pragma unroll
    for (int j = 0; j < kGateTokens; ++j) {
        xr[j] = a.x16 + (t0 + (j < nt ? j : nt - 1)) * a.k_in;
    }
    for (int i = tid; i < k_in; i += 256) {
        float wb[HB];
        float wa[HB];
#pragma unroll
        for (int hb = 0; hb < HB; ++hb) {
            wb[hb] = __uint_as_float((uint32_t) a.w_beta[(h0 + hb) * k_in + i] << 16);
            wa[hb] = __uint_as_float((uint32_t) a.w_alpha[(h0 + hb) * k_in + i] << 16);
        }
#pragma unroll
        for (int j = 0; j < kGateTokens; ++j) {
            const float xi = __half2float(xr[j][i]);
#pragma unroll
            for (int hb = 0; hb < HB; ++hb) {
                pb[hb][j] = fmaf(wb[hb], xi, pb[hb][j]);
                pa[hb][j] = fmaf(wa[hb], xi, pa[hb][j]);
            }
        }
    }
#pragma unroll
    for (int hb = 0; hb < HB; ++hb) {
#pragma unroll
        for (int j = 0; j < kGateTokens; ++j) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                pb[hb][j] += __shfl_xor(pb[hb][j], off);
                pa[hb][j] += __shfl_xor(pa[hb][j], off);
            }
            if ((tid & 31) == 0) {
                gred[0][hb][j][tid >> 5] = pb[hb][j];
                gred[1][hb][j][tid >> 5] = pa[hb][j];
            }
        }
    }
    __syncthreads();
    if (tid < nt * HB) {
        const int hb = tid / nt;
        const int jt = tid % nt;
        const long long h = h0 + hb;
        const long long t = t0 + jt;
        float beta_raw = 0.0f;
        float alpha_raw = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            beta_raw += gred[0][hb][jt][i];
            alpha_raw += gred[1][hb][jt][i];
        }
        const float b = 1.0f / (1.0f + expf(-beta_raw));
        const float ar = alpha_raw + a.dt_bias[h];
        const float lgd = (log1pf(expf(-fabsf(ar))) + fmaxf(ar, 0.0f)) * a.ssm_a[h];
        const float decay = expf(lgd);
        const ChunkWork w = chunk_work(a, heads);
        if (a.wy != nullptr) {
            a.wy[t * heads + h] = lgd;  // the WY form's log decay (#240)
        }
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
__global__ __launch_bounds__(kDeltaS) void delta_post_kernel(const GdnStep a, const float * __restrict__ vo) {
    __shared__ float wsq[4];
    const long long h = blockIdx.x;
    const long long t = blockIdx.y;
    const long long heads = gridDim.x;
    const int col = (int) threadIdx.x;
    const float ov = vo[(t * heads + h) * kDeltaS + col];
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


// --- The chunked WY form of the recurrence (#240, PLAN.md §10.3) ---------------
//
// Per 64-token chunk of one head, with h the state at the chunk start (S[i][j]:
// i the key dim, j the value dim), g_t the cumulative log decay inside the chunk
// (gamma_t = exp(g_t), inclusive) and Gam_ts = exp(g_t - g_s):
//   A = strict_lower(beta_t (k_t . k_s) Gam_ts), T = (I + A)^-1
//   D = T beta (V - gamma K h)                       (the per-token d_t)
//   o_t = l2_scale (gamma_t q_t h + sum_{s<=t} (q_t . k_s) Gam_ts D_s)
//   h' = gamma_C h + (K gamma_C / gamma)^T D
// the per-token delta rule's values (a NumPy check gives 1e-15). Phase A builds
// T (as two f16: high and low part) and P = (Q K^T) o Gam for every (head,
// chunk) in parallel, plus K^T per key head; phase B walks the chunks of a head
// with h in f32 WMMA accumulators, each wave owning 16 value columns, so the
// waves never talk. Q, K and V come straight from the conv output. The products
// are f16 WMMAs with f32 accumulation: not bit-identical to gdn_step, prefill
// only (verifications keep the exact path). KL vs the exact path on 4096-token
// prompts: 0.0002-0.0005 (the default KV cache's own: 0.0012-0.0015).
typedef _Float16 wy_h8 __attribute__((ext_vector_type(8)));
typedef float wy_f8 __attribute__((ext_vector_type(8)));
constexpr int kWyC = (int) kWyChunk;
constexpr int kWyAld = kWyC + 4;  // f32 LDS row of A (16-byte aligned rows)

struct WyHead {  // float offsets in a (value head, chunk) block
    static constexpr int kTh = 0;      // f16 [64][64], T's high part
    static constexpr int kTl = 2048;   // f16 [64][64], T - Th
    static constexpr int kP = 4096;    // f16 [64][64], (q_t . k_s) Gam_ts for s <= t
    static constexpr int kG = 6144;    // g_t
    static constexpr int kB = 6208;    // beta_t
};
static_assert(WyHead::kB + 64 == kWyHeadFloats, "gdn.hh's kWyHeadFloats");

__device__ inline float * wy_head(float * wy, const long long tokens, const long long heads, const long long n_kh,
                                  const long long h, const long long c) {
    const long long lg = (tokens * heads + 15) / 16 * 16;
    const long long per = n_kh * kWyKeyFloats + heads * kWyHeadFloats;
    return wy + lg + c * per + n_kh * kWyKeyFloats + h * kWyHeadFloats;
}

// a key head's chunk: f16 K^T [128][64], then K [64][128] and Q [64][128]
__device__ inline _Float16 * wy_kt(float * wy, const long long tokens, const long long heads, const long long n_kh,
                                   const long long kh, const long long c) {
    const long long lg = (tokens * heads + 15) / 16 * 16;
    const long long per = n_kh * kWyKeyFloats + heads * kWyHeadFloats;
    return reinterpret_cast<_Float16 *>(wy + lg + c * per + kh * kWyKeyFloats);
}

// phase B's output: (chunks x 64) rows x heads x 128, the rows past the last token
// written too (no guard in the loop), read by delta_post_kernel
__host__ __device__ inline float * wy_out(float * wy, const long long tokens, const long long heads,
                                          const long long n_kh) {
    const long long lg = (tokens * heads + 15) / 16 * 16;
    const long long chunks = (tokens + kWyC - 1) / kWyC;
    return wy + lg + chunks * (n_kh * kWyKeyFloats + heads * kWyHeadFloats);
}

__device__ inline wy_f8 wy_mma(const wy_h8 a, const wy_h8 b, const wy_f8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

// 8 consecutive floats of a conv-output row as f16 (rows past the chunk: 0)
__device__ inline wy_h8 wy_row8(const float * p, const bool in) {
    wy_h8 r;
    if (in) {
        const float4 a = reinterpret_cast<const float4 *>(p)[0];
        const float4 b = reinterpret_cast<const float4 *>(p)[1];
        r = wy_h8{(_Float16) a.x, (_Float16) a.y, (_Float16) a.z, (_Float16) a.w,
                  (_Float16) b.x, (_Float16) b.y, (_Float16) b.z, (_Float16) b.w};
    } else {
        r = wy_h8{};
    }
    return r;
}

// Phase A: grid (heads, chunks), 256 threads.
__global__ __launch_bounds__(256) void delta_wy_a_kernel(const float * __restrict__ cv,
                                                        const float * __restrict__ gates, float * __restrict__ wy,
                                                        const long long tokens, const long long n_kh,
                                                        const long long channels) {
    __shared__ __attribute__((aligned(16))) float am[kWyC * kWyAld];  // K^T staging (f16), then A
    __shared__ float gs[kWyC];
    __shared__ float bs[kWyC];
    const long long h = blockIdx.x;
    const long long heads = gridDim.x;
    const long long c = blockIdx.y;
    const long long t0 = c * kWyC;
    const int n = (int) (tokens - t0 < kWyC ? tokens - t0 : kWyC);
    const int tid = (int) threadIdx.x;
    const int lane = tid & 31;
    const int wv = tid >> 5;
    const int r16 = lane & 15;
    const int hk = (lane >> 4) * 8;
    const long long kh = h % n_kh;
    const float * qrow = cv + t0 * channels + kh * kDeltaS;
    const float * krow = qrow + n_kh * kDeltaS;
    float * blk = wy_head(wy, tokens, heads, n_kh, h, c);
    if (h < n_kh) {  // this key head's K^T, K and Q in f16, once (K^T staged through LDS)
        _Float16 * st = reinterpret_cast<_Float16 *>(am);
        _Float16 * kt = wy_kt(wy, tokens, heads, n_kh, kh, c);
        for (int e = tid; e < kWyC * kDeltaS; e += 256) {
            const int t = e / kDeltaS;
            const int i = e % kDeltaS;
            const _Float16 k = (_Float16) (t < n ? krow[t * channels + i] : 0.0f);
            st[t * kDeltaS + i] = k;
            kt[kWyC * kDeltaS + e] = k;
            kt[2 * kWyC * kDeltaS + e] = (_Float16) (t < n ? qrow[t * channels + i] : 0.0f);
        }
        __syncthreads();
        for (int e = tid; e < kWyC * kDeltaS; e += 256) {
            const int i = e / kWyC;
            const int t = e % kWyC;
            kt[i * kWyC + t] = st[t * kDeltaS + i];
        }
        __syncthreads();
    }
    if (tid < kWyC) {  // padded rows: beta 0, no decay, so they add nothing
        bs[tid] = tid < n ? gates[((t0 + tid) * heads + h) * 2] : 0.0f;
        gs[tid] = tid < n ? wy[(t0 + tid) * heads + h] : 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
        float acc = 0.0f;
        for (int t = 0; t < kWyC; ++t) {
            acc += gs[t];
            gs[t] = acc;
        }
    }
    __syncthreads();
    if (tid < kWyC) {
        blk[WyHead::kG + tid] = gs[tid];
        blk[WyHead::kB + tid] = bs[tid];
    }
    // K K^T -> A (LDS) and Q K^T -> P: the 10 lower 16 x 16 tiles each, 20 jobs over 8 waves,
    // the operands straight from the conv output
    _Float16 * pm = reinterpret_cast<_Float16 *>(blk + WyHead::kP);
    for (int job = wv; job < 20; job += 8) {
        const bool is_p = job >= 10;
        int st = job % 10;
        int rt = 0;
        while (st > rt) {
            st -= rt + 1;
            ++rt;
        }
        const int ta = rt * 16 + r16;
        const int tb = st * 16 + r16;
        const float * arow = (is_p ? qrow : krow) + (long long) ta * channels + hk;
        const float * brow = krow + (long long) tb * channels + hk;
        wy_f8 acc = {};
#pragma unroll
        for (int kk = 0; kk < kDeltaS; kk += 16) {
            acc = wy_mma(wy_row8(arow + kk, ta < n), wy_row8(brow + kk, tb < n), acc);
        }
        const int s = tb;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int t = rt * 16 + hk + e;
            const float gam = expf(gs[t] - gs[s]);
            if (is_p) {
                pm[t * kWyC + s] = (_Float16) (s <= t ? acc[e] * gam : 0.0f);
            } else {
                am[t * kWyAld + s] = s < t ? bs[t] * acc[e] * gam : 0.0f;
            }
        }
    }
    __syncthreads();
    // T = (I + A)^-1 (unit lower) by forward substitution in f32, one column per thread;
    // A's upper tiles are never read. T can have large entries that cancel, so it goes
    // out as two f16 (one lost a token's whole distribution at 4k).
    if (tid < kWyC) {
        const int col = tid;
        _Float16 * th = reinterpret_cast<_Float16 *>(blk + WyHead::kTh);
        _Float16 * tl = reinterpret_cast<_Float16 *>(blk + WyHead::kTl);
        float tc[kWyC];
#pragma unroll
        for (int t = 0; t < kWyC; ++t) {
            float p0 = t == col ? 1.0f : 0.0f;
            float p1 = 0.0f;
            float p2 = 0.0f;
            float p3 = 0.0f;
#pragma unroll
            for (int s = 0; s + 3 < t; s += 4) {
                const float4 a4 = *reinterpret_cast<const float4 *>(am + t * kWyAld + s);
                p0 = fmaf(-a4.x, tc[s], p0);
                p1 = fmaf(-a4.y, tc[s + 1], p1);
                p2 = fmaf(-a4.z, tc[s + 2], p2);
                p3 = fmaf(-a4.w, tc[s + 3], p3);
            }
#pragma unroll
            for (int s = t & ~3; s < t; ++s) {
                p0 = fmaf(-am[t * kWyAld + s], tc[s], p0);
            }
            const float v = t < col ? 0.0f : (p0 + p1) + (p2 + p3);
            tc[t] = v;
            const _Float16 hi = (_Float16) v;
            th[t * kWyC + col] = hi;
            tl[t * kWyC + col] = (_Float16) (v - (float) hi);
        }
    }
}

// One chunk of phase B. Rows past the last token (n < 64, the last chunk) read the
// last real row of V, which beta = 0 cancels, and store nothing; a second, guarded
// instance of this body in the loop pushed the kernel past 256 VGPRs (#240).
__device__ __forceinline__ void wy_b_chunk(wy_f8 (&hs)[8], const float * __restrict__ cv, float * wy,
                                       float * __restrict__ vo, const long long tokens, const long long heads,
                                       const long long n_kh, const long long channels, const float l2_scale,
                                       const long long h, const long long kh, const long long c, const int n,
                                       const int r16, const int hk, const int j) {
    const long long t0 = c * kWyC;
    const float * blk = wy_head(wy, tokens, heads, n_kh, h, c);
    const _Float16 * th = reinterpret_cast<const _Float16 *>(blk + WyHead::kTh);
    const _Float16 * tl = reinterpret_cast<const _Float16 *>(blk + WyHead::kTl);
    const _Float16 * pm = reinterpret_cast<const _Float16 *>(blk + WyHead::kP);
    const float * gg = blk + WyHead::kG;
    const float * bb = blk + WyHead::kB;
    const _Float16 * kt = wy_kt(wy, tokens, heads, n_kh, kh, c);
    const _Float16 * kr = kt + kWyC * kDeltaS;
    const _Float16 * qr = kt + 2 * kWyC * kDeltaS;
    const float * vrow = cv + t0 * channels + 2 * n_kh * kDeltaS + h * kDeltaS;
    // K h, gamma q h and R = beta (V - gamma K h) per row tile; h as f16 B operands once
    wy_h8 hb[8];
#pragma unroll
    for (int it = 0; it < 8; ++it) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            hb[it][e] = (_Float16) hs[it][e];
        }
    }
    wy_h8 rb[4];
#pragma unroll
    for (int rt = 0; rt < 4; ++rt) {
        wy_f8 kx = {};
#pragma unroll
        for (int it = 0; it < 8; ++it) {
            kx = wy_mma(*reinterpret_cast<const wy_h8 *>(kr + (rt * 16 + r16) * kDeltaS + it * 16 + hk), hb[it], kx);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int t = rt * 16 + hk + e;
            const float v = vrow[(t < n ? t : n - 1) * (int) channels + j];
            rb[rt][e] = (_Float16) (bb[t] * (v - expf(gg[t]) * kx[e]));
        }
        __builtin_amdgcn_sched_barrier(0);  // one row tile's loads at a time (VGPRs)
    }
    __builtin_amdgcn_sched_barrier(0);  // keep each phase's loads in it (VGPRs)
    // D = T R
    wy_h8 db[4];
#pragma unroll
    for (int rt = 0; rt < 4; ++rt) {
        wy_f8 d = {};
#pragma unroll
        for (int st = 0; st <= rt; ++st) {
            const wy_h8 a = *reinterpret_cast<const wy_h8 *>(th + (rt * 16 + r16) * kWyC + st * 16 + hk);
            const wy_h8 al = *reinterpret_cast<const wy_h8 *>(tl + (rt * 16 + r16) * kWyC + st * 16 + hk);
            d = wy_mma(al, rb[st], wy_mma(a, rb[st], d));
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            db[rt][e] = (_Float16) d[e];
        }
    }
    __builtin_amdgcn_sched_barrier(0);  // keep each phase's loads in it (VGPRs)
    // o = gamma q h + P D, one row tile at a time
#pragma unroll
    for (int rt = 0; rt < 4; ++rt) {
        wy_f8 o = {};
#pragma unroll
        for (int it = 0; it < 8; ++it) {
            o = wy_mma(*reinterpret_cast<const wy_h8 *>(qr + (rt * 16 + r16) * kDeltaS + it * 16 + hk), hb[it], o);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            o[e] *= expf(gg[rt * 16 + hk + e]);
        }
#pragma unroll
        for (int st = 0; st <= rt; ++st) {
            const wy_h8 a = *reinterpret_cast<const wy_h8 *>(pm + (rt * 16 + r16) * kWyC + st * 16 + hk);
            o = wy_mma(a, db[st], o);
        }
        float * vp = vo + ((t0 + rt * 16 + hk) * heads + h) * kDeltaS + j;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            vp[e * heads * kDeltaS] = l2_scale * o[e];
        }
        __builtin_amdgcn_sched_barrier(0);
    }
    __builtin_amdgcn_sched_barrier(0);  // keep each phase's loads in it (VGPRs)
    // h' = gamma_C h + (K gamma_C / gamma)^T D
    const float gl = gg[kWyC - 1];
    const float gc = expf(gl);
#pragma unroll
    for (int it = 0; it < 8; ++it) {
        hs[it] *= gc;
    }
#pragma unroll
    for (int st = 0; st < 4; ++st) {
        float fs[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            fs[e] = expf(gl - gg[st * 16 + hk + e]);
        }
#pragma unroll
        for (int it = 0; it < 8; ++it) {
            const wy_h8 k8 = *reinterpret_cast<const wy_h8 *>(kt + (it * 16 + r16) * kWyC + st * 16 + hk);
            wy_h8 a;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                a[e] = (_Float16) ((float) k8[e] * fs[e]);
            }
            hs[it] = wy_mma(a, db[st], hs[it]);
        }
    }
    }

// Phase B: grid (heads), 256 threads; wave w owns value columns 16 w .. 16 w + 15.
__global__ __launch_bounds__(256) void delta_wy_b_kernel(const float * __restrict__ cv, float * __restrict__ wy,
                                                        const float * state,
                                                        float * state_out,  // may alias
                                                        const long long tokens, const long long n_kh,
                                                        const long long channels, const float l2_scale) {
    const long long h = blockIdx.x;
    const long long heads = gridDim.x;
    const long long kh = h % n_kh;
    const int lane = threadIdx.x & 31;
    const int wv = threadIdx.x >> 5;
    const int r16 = lane & 15;
    const int hk = (lane >> 4) * 8;
    const int j = wv * 16 + r16;
    wy_f8 hs[8];  // h rows it * 16 + hk + e, column j
    const long long sbase = (h * kDeltaS + hk) * kDeltaS + j;  // + constant offsets (VGPRs)
#pragma unroll
    for (int it = 0; it < 8; ++it) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            hs[it][e] = state[sbase + (it * 16 + e) * kDeltaS];
        }
    }
    const long long n_chunks = (tokens + kWyC - 1) / kWyC;
    for (long long c = 0; c < n_chunks; ++c) {
        const int n = (int) (tokens - c * kWyC < kWyC ? tokens - c * kWyC : kWyC);
        wy_b_chunk(hs, cv, wy, wy_out(wy, tokens, heads, n_kh), tokens, heads, n_kh, channels, l2_scale, h, kh, c, n,
                   r16, hk, j);
    }
#pragma unroll
    for (int it = 0; it < 8; ++it) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            state_out[sbase + (it * 16 + e) * kDeltaS] = hs[it][e];
        }
    }
}

// The rank-1 updates of recorded tokens, on the state of the step before them:
// gdn_step's thread layout and float operations (decay, then fma with k and d).
__device__ __forceinline__ void replay_body(float * __restrict__ state, const float * __restrict__ replay,
                                            const long long tokens, const long long n_kh, const long long h,
                                            const long long heads) {
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

__global__ __launch_bounds__(256) void gdn_replay_kernel(float * __restrict__ state,
                                                         const float * __restrict__ replay,
                                                         const long long tokens,
                                                         const long long n_kh) {
    replay_body(state, replay, tokens, n_kh, blockIdx.x, gridDim.x);
}

// blockIdx.y = the layer; x < heads: its replay, the rest: its conv tail
__global__ __launch_bounds__(256) void gdn_rollback_kernel(const GdnRollback r, const long long tokens,
                                                           const long long heads, const long long n_kh,
                                                           const long long rows, const long long channels) {
    const int l = (int) blockIdx.y;
    if ((long long) blockIdx.x < heads) {
        replay_body(r.state[l], r.replay[l], tokens, n_kh, blockIdx.x, heads);
        return;
    }
    const long long i = ((long long) blockIdx.x - heads) * blockDim.x + threadIdx.x;
    if (i < rows * channels) {
        r.conv_dst[l][i] = r.conv_hist[l][tokens * channels + i];
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

bool gdn_rollback(const GdnRollback & r, const int64_t accepted, const int64_t heads, const int64_t n_kh,
                  const int64_t conv_k, const int64_t channels, hipStream_t stream) {
    if (r.layers <= 0 || r.layers > kGdnRollbackMax || accepted < 0 || heads <= 0 || n_kh <= 0 || conv_k <= 1 ||
        channels <= 0) {
        return false;
    }
    const long long n = (long long) (conv_k - 1) * channels;
    const dim3 grid((unsigned) (heads + (n + 255) / 256), (unsigned) r.layers);
    gdn_rollback_kernel<<<grid, 256, 0, stream>>>(r, (long long) accepted, (long long) heads, (long long) n_kh,
                                                  (long long) (conv_k - 1), (long long) channels);
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
    if (a.conv_k != 4) {
        return false;  // delta_conv_kernel's kMaxConv
    }
    const unsigned conv_groups = (unsigned) ((a.tokens + kConvTokens - 1) / kConvTokens);
    delta_conv_kernel<<<dim3((unsigned) (a.channels / kDeltaS), conv_groups), kDeltaS, 0, stream>>>(a);
    const unsigned gate_groups = (unsigned) ((a.tokens + kGateTokens - 1) / kGateTokens);
    // 4 heads per block read a prefill chunk's x once for them; a short call
    // (a verification) needs the workgroups more (#271: 12 of them at 8 tokens)
    if (heads % 4 == 0 && a.tokens > kGateTokens) {
        delta_gate_kernel<4><<<dim3((unsigned) (heads / 4), gate_groups), 256, 0, stream>>>(a);
    } else {
        delta_gate_kernel<1><<<dim3((unsigned) heads, gate_groups), 256, 0, stream>>>(a);
    }
    float * vo = a.work + a.tokens * a.channels;
    const float * gates = vo + a.tokens * heads * kDeltaS;
    if (a.wy != nullptr && a.replay == nullptr && a.state_out == nullptr) {
        const unsigned chunks = (unsigned) ((a.tokens + kWyC - 1) / kWyC);
        delta_wy_a_kernel<<<dim3((unsigned) heads, chunks), 256, 0, stream>>>(a.work, gates, a.wy, a.tokens, a.n_kh,
                                                                            a.channels);
        delta_wy_b_kernel<<<(unsigned) heads, 256, 0, stream>>>(a.work, a.wy, a.state, a.state, a.tokens, a.n_kh,
                                                                a.channels, a.l2_scale);
        delta_post_kernel<<<per_token, kDeltaS, 0, stream>>>(a, wy_out(a.wy, a.tokens, heads, a.n_kh));
        return hipGetLastError() == hipSuccess;
    }
    delta_rec_kernel<<<dim3((unsigned) heads, kDeltaS / kRecCols), kRecCols, 0, stream>>>(
        a.work, gates, vo, a.replay, a.state, a.state_out != nullptr ? a.state_out : a.state,
        a.tokens, a.n_kh, a.channels, a.l2_scale);
    delta_post_kernel<<<per_token, kDeltaS, 0, stream>>>(a, vo);
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
