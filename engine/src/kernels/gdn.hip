// Gated delta net kernels (PLAN.md §10.2, autoregressive path).
#include "kernels/gdn.hh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace omph::kernels {
namespace {

constexpr int kBlock = 256;

__global__ void sigmoid_kernel(float * __restrict__ x, const long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

__global__ void silu_kernel(float * __restrict__ x, const long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

__global__ void softplus_bias_kernel(float * __restrict__ alpha, const float * __restrict__ dt,
                                     const long long tokens, const long long heads) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * heads) {
        return;
    }
    const float a = alpha[i] + dt[i % heads];
    alpha[i] = log1pf(expf(-fabsf(a))) + fmaxf(a, 0.0f);
}

__global__ void mul_row_kernel(float * __restrict__ x, const float * __restrict__ w,
                               const long long tokens, const long long heads) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < tokens * heads) {
        x[i] *= w[i % heads];
    }
}

// Depthwise causal conv1d: out[t, c] = sum_j w[c, j] * input[t + j, c] with
// input = [state rows (kernel-1, channels); qkv rows (tokens, channels)].
__global__ void conv1d_kernel(const float * __restrict__ qkv, const float * __restrict__ w,
                              const float * __restrict__ state, float * __restrict__ out,
                              const long long tokens, const long long channels,
                              const long long kernel) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * channels) {
        return;
    }
    const long long c = i % channels;
    const long long t = i / channels;
    float acc = 0.0f;
    for (long long j = 0; j < kernel; ++j) {
        const long long r = t + j;
        const float xv = (r < kernel - 1) ? state[r * channels + c]
                                          : qkv[(r - (kernel - 1)) * channels + c];
        acc += w[c * kernel + j] * xv;
    }
    out[i] = acc;
}

__global__ void conv_state_update_kernel(const float * __restrict__ qkv,
                                         const float * __restrict__ state,
                                         float * __restrict__ new_state, const long long tokens,
                                         const long long channels, const long long kernel) {
    const long long rows = kernel - 1;
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * channels) {
        return;
    }
    const long long r = i / channels;
    const long long c = i % channels;
    const long long src = tokens + r;
    new_state[r * channels + c] = (src < rows) ? state[src * channels + c]
                                               : qkv[(src - rows) * channels + c];
}

__global__ void split_qkv_kernel(const float * __restrict__ fused, float * __restrict__ q,
                                 float * __restrict__ k, float * __restrict__ v,
                                 const long long tokens, const long long q_dims,
                                 const long long kv_dims, const long long v_dims) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const long long channels = q_dims + kv_dims + v_dims;
    if (i >= tokens * channels) {
        return;
    }
    const long long c = i % channels;
    const long long t = i / channels;
    const float value = fused[i];
    if (c < q_dims) {
        q[t * q_dims + c] = value;
    } else if (c < q_dims + kv_dims) {
        k[t * kv_dims + (c - q_dims)] = value;
    } else {
        v[t * v_dims + (c - q_dims - kv_dims)] = value;
    }
}

__global__ void delta_decay_kernel(float * __restrict__ state, const float * __restrict__ gate,
                                   const long long heads, const long long s) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * s * s) {
        return;
    }
    const long long h = i / (s * s);
    state[i] *= expf(gate[h]);
}

__global__ void delta_sk_kernel(const float * __restrict__ state, const float * __restrict__ k,
                                float * __restrict__ sk, const long long heads, const long long n_kh,
                                const long long s) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * s) {
        return;
    }
    const long long h = i / s;
    const long long j = i % s;
    const float * S = state + h * s * s;
    const float * kr = k + (h % n_kh) * s;
    float acc = 0.0f;
    for (long long ii = 0; ii < s; ++ii) {
        acc += S[ii * s + j] * kr[ii];
    }
    sk[i] = acc;
}

__global__ void delta_d_kernel(const float * __restrict__ v, const float * __restrict__ sk,
                               const float * __restrict__ beta, float * __restrict__ d,
                               const long long heads, const long long s) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * s) {
        return;
    }
    const long long h = i / s;
    d[i] = (v[i] - sk[i]) * beta[h];
}

__global__ void delta_update_kernel(float * __restrict__ state, const float * __restrict__ k,
                                    const float * __restrict__ d, const long long heads,
                                    const long long n_kh, const long long s) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * s * s) {
        return;
    }
    const long long h = i / (s * s);
    const long long rem = i % (s * s);
    const long long ii = rem / s;
    const long long j = rem % s;
    state[i] += k[(h % n_kh) * s + ii] * d[h * s + j];
}

__global__ void delta_o_kernel(const float * __restrict__ state, const float * __restrict__ q,
                               float * __restrict__ o, const long long heads, const long long n_kh,
                               const long long s, const float scale) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * s) {
        return;
    }
    const long long h = i / s;
    const long long j = i % s;
    const float * S = state + h * s * s;
    const float * qr = q + (h % n_kh) * s;
    float acc = 0.0f;
    for (long long ii = 0; ii < s; ++ii) {
        acc += S[ii * s + j] * qr[ii];
    }
    o[i] = acc * scale;
}

template <typename OutT>
__global__ void gated_norm_kernel(const float * __restrict__ o, const float * __restrict__ w,
                                  const float * __restrict__ z, OutT * __restrict__ out,
                                  const long long n, const float eps) {
    extern __shared__ float sdata[];
    const long long row = blockIdx.x;
    const float * orow = o + row * n;
    const float * zrow = z + row * n;
    OutT * yrow = out + row * n;

    float acc = 0.0f;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        acc += orow[i] * orow[i];
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if ((int) threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf(sdata[0] / (float) n + eps);
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        const float zn = zrow[i];
        yrow[i] = (OutT) (orow[i] * scale * (w != nullptr ? w[i] : 1.0f) * (zn / (1.0f + expf(-zn))));
    }
}

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

__global__ __launch_bounds__(256) void delta_step_kernel(
    float * __restrict__ state, const float * __restrict__ q, const float * __restrict__ k,
    const float * __restrict__ v, const float * __restrict__ beta,
    const float * __restrict__ alpha, const float * __restrict__ dt_bias,
    const float * __restrict__ ssm_a, float * __restrict__ o, const long long n_kh,
    const long long s, const float scale) {
    __shared__ float part[2 * kDeltaS];
    __shared__ float dd[kDeltaS];
    __shared__ float kq[2 * kDeltaS];  // k then q of this head's key group
    const long long h = blockIdx.x;
    float * S = state + h * kDeltaS * kDeltaS;
    const int tid = (int) threadIdx.x;
    const int col = tid & (kDeltaS - 1);
    const int half = tid >> 7;
    const int lo = half * kDeltaRows;
    kq[tid] = tid < kDeltaS ? k[(h % n_kh) * kDeltaS + tid] : q[(h % n_kh) * kDeltaS + tid - kDeltaS];
    // The gates from the raw projections (formerly three kernels, #78):
    // b = sigmoid(beta), decay = exp(softplus(alpha + dt_bias) * A).
    const float b = 1.0f / (1.0f + expf(-beta[h]));
    const float ar = alpha[h] + dt_bias[h];
    const float decay = expf((log1pf(expf(-fabsf(ar))) + fmaxf(ar, 0.0f)) * ssm_a[h]);

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
        dd[col] = (v[h * kDeltaS + col] - (part[col] + part[kDeltaS + col])) * b;
    }
    __syncthreads();
    const float dc = dd[col];
    float acc2 = 0.0f;
#pragma unroll
    for (int r = 0; r < kDeltaRows; ++r) {
        st[r] = fmaf(kq[lo + r], dc, st[r]);
        acc2 = fmaf(st[r], kq[kDeltaS + lo + r], acc2);
        S[(lo + r) * kDeltaS + col] = st[r];
    }
    __syncthreads();  // part is reused
    part[half * kDeltaS + col] = acc2;
    __syncthreads();
    if (half == 0) {
        o[h * kDeltaS + col] = scale * (part[col] + part[kDeltaS + col]);
    }
    (void) s;
}


// conv1d + silu + q/k/v split + the L2 normalization of q and k per head, in
// one launch (#78; formerly conv_silu_split + two rms_norm launches). grid:
// (tokens, q heads + k heads + v channels / 128); 128 threads, one channel
// each. A q or k workgroup is exactly one head, so its norm is a block
// reduction: y = x / sqrt(mean(x^2) + eps) * scale, as rms_norm computed it.
// The workgroups of token 0 also shift the conv state for their channels.
constexpr int kConvHead = 128;

__global__ __launch_bounds__(kConvHead) void conv_silu_split_l2_kernel(
    const float * __restrict__ qkv, const float * __restrict__ w,
    const float * __restrict__ state, float * __restrict__ new_state, float * __restrict__ q,
    float * __restrict__ k, float * __restrict__ v, const long long tokens,
    const long long channels, const long long kernel, const long long q_dims,
    const long long kv_dims, const long long v_dims, const float eps, const float scale) {
    __shared__ float wsum[kConvHead / 32];
    const long long t = blockIdx.x;
    const long long c = (long long) blockIdx.y * kConvHead + threadIdx.x;
    const long long rows = kernel - 1;
    float acc = 0.0f;
    for (long long j = 0; j < kernel; ++j) {
        const long long r = t + j;
        const float xv = (r < rows) ? state[r * channels + c] : qkv[(r - rows) * channels + c];
        acc = fmaf(w[c * kernel + j], xv, acc);
    }
    float y = acc / (1.0f + expf(-acc));  // silu
    if (t == 0) {
        for (long long r = 0; r < rows; ++r) {
            const long long src = tokens + r;
            new_state[r * channels + c] =
                (src < rows) ? state[src * channels + c] : qkv[(src - rows) * channels + c];
        }
    }
    if (c < q_dims + kv_dims) {  // a whole q or k head: normalize it
        float ss = y * y;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            ss += __shfl_xor(ss, off);
        }
        if ((threadIdx.x & 31) == 0) {
            wsum[threadIdx.x >> 5] = ss;
        }
        __syncthreads();
        const float tot = wsum[0] + wsum[1] + wsum[2] + wsum[3];
        y = y * (1.0f / sqrtf(tot / (float) kConvHead + eps)) * scale;
    }
    if (c < q_dims) {
        q[t * q_dims + c] = y;
    } else if (c < q_dims + kv_dims) {
        k[t * kv_dims + (c - q_dims)] = y;
    } else {
        v[t * v_dims + (c - q_dims - kv_dims)] = y;
    }
}

unsigned grid_for(const long long n) {
    return (unsigned) ((n + kBlock - 1) / kBlock);
}

} // namespace

bool sigmoid_inplace(float * x, const int64_t n, hipStream_t stream) {
    if (x == nullptr || n <= 0) {
        return false;
    }
    sigmoid_kernel<<<grid_for(n), kBlock, 0, stream>>>(x, (long long) n);
    return hipGetLastError() == hipSuccess;
}

bool silu_inplace(float * x, const int64_t n, hipStream_t stream) {
    if (x == nullptr || n <= 0) {
        return false;
    }
    silu_kernel<<<grid_for(n), kBlock, 0, stream>>>(x, (long long) n);
    return hipGetLastError() == hipSuccess;
}

bool softplus_bias_inplace(float * alpha, const float * dt_bias, const int64_t tokens,
                           const int64_t heads, hipStream_t stream) {
    if (alpha == nullptr || dt_bias == nullptr || tokens <= 0 || heads <= 0) {
        return false;
    }
    softplus_bias_kernel<<<grid_for((long long) tokens * heads), kBlock, 0, stream>>>(
        alpha, dt_bias, (long long) tokens, (long long) heads);
    return hipGetLastError() == hipSuccess;
}

bool mul_row_inplace(float * x, const float * w, const int64_t tokens, const int64_t heads,
                     hipStream_t stream) {
    if (x == nullptr || w == nullptr || tokens <= 0 || heads <= 0) {
        return false;
    }
    mul_row_kernel<<<grid_for((long long) tokens * heads), kBlock, 0, stream>>>(
        x, w, (long long) tokens, (long long) heads);
    return hipGetLastError() == hipSuccess;
}

bool conv1d_state(const float * qkv, const float * w, const float * state, float * out,
                  const int64_t tokens, const int64_t channels, const int64_t kernel,
                  hipStream_t stream) {
    if (qkv == nullptr || w == nullptr || state == nullptr || out == nullptr || tokens <= 0 ||
        channels <= 0 || kernel <= 1) {
        return false;
    }
    conv1d_kernel<<<grid_for((long long) tokens * channels), kBlock, 0, stream>>>(
        qkv, w, state, out, (long long) tokens, (long long) channels, (long long) kernel);
    return hipGetLastError() == hipSuccess;
}

bool conv_state_update(const float * qkv, const float * state, float * new_state,
                       const int64_t tokens, const int64_t channels, const int64_t kernel,
                       hipStream_t stream) {
    if (qkv == nullptr || state == nullptr || new_state == nullptr || tokens <= 0 ||
        channels <= 0 || kernel <= 1) {
        return false;
    }
    conv_state_update_kernel<<<grid_for((long long) (kernel - 1) * channels), kBlock, 0, stream>>>(
        qkv, state, new_state, (long long) tokens, (long long) channels, (long long) kernel);
    return hipGetLastError() == hipSuccess;
}

bool split_qkv(const float * fused, float * q, float * k, float * v, const int64_t tokens,
               const int64_t q_dims, const int64_t kv_dims, const int64_t v_dims,
               hipStream_t stream) {
    if (fused == nullptr || q == nullptr || k == nullptr || v == nullptr || tokens <= 0) {
        return false;
    }
    split_qkv_kernel<<<grid_for((long long) tokens * (q_dims + kv_dims + v_dims)), kBlock, 0,
                       stream>>>(fused, q, k, v, (long long) tokens, (long long) q_dims,
                                 (long long) kv_dims, (long long) v_dims);
    return hipGetLastError() == hipSuccess;
}

bool delta_decay(float * state, const float * gate, const int64_t heads, const int64_t s,
                 hipStream_t stream) {
    if (state == nullptr || gate == nullptr || heads <= 0 || s <= 0) {
        return false;
    }
    delta_decay_kernel<<<grid_for((long long) heads * s * s), kBlock, 0, stream>>>(
        state, gate, (long long) heads, (long long) s);
    return hipGetLastError() == hipSuccess;
}

bool delta_sk(const float * state, const float * k, float * sk, const int64_t heads,
              const int64_t n_kh, const int64_t s, hipStream_t stream) {
    if (state == nullptr || k == nullptr || sk == nullptr || heads <= 0 || n_kh <= 0 || s <= 0) {
        return false;
    }
    delta_sk_kernel<<<grid_for((long long) heads * s), kBlock, 0, stream>>>(
        state, k, sk, (long long) heads, (long long) n_kh, (long long) s);
    return hipGetLastError() == hipSuccess;
}

bool delta_d(const float * v, const float * sk, const float * beta, float * d, const int64_t heads,
             const int64_t s, hipStream_t stream) {
    if (v == nullptr || sk == nullptr || beta == nullptr || d == nullptr || heads <= 0 || s <= 0) {
        return false;
    }
    delta_d_kernel<<<grid_for((long long) heads * s), kBlock, 0, stream>>>(
        v, sk, beta, d, (long long) heads, (long long) s);
    return hipGetLastError() == hipSuccess;
}

bool delta_update(float * state, const float * k, const float * d, const int64_t heads,
                  const int64_t n_kh, const int64_t s, hipStream_t stream) {
    if (state == nullptr || k == nullptr || d == nullptr || heads <= 0 || n_kh <= 0 || s <= 0) {
        return false;
    }
    delta_update_kernel<<<grid_for((long long) heads * s * s), kBlock, 0, stream>>>(
        state, k, d, (long long) heads, (long long) n_kh, (long long) s);
    return hipGetLastError() == hipSuccess;
}

bool delta_o(const float * state, const float * q, float * o, const int64_t heads,
             const int64_t n_kh, const int64_t s, const float scale, hipStream_t stream) {
    if (state == nullptr || q == nullptr || o == nullptr || heads <= 0 || n_kh <= 0 || s <= 0) {
        return false;
    }
    delta_o_kernel<<<grid_for((long long) heads * s), kBlock, 0, stream>>>(
        state, q, o, (long long) heads, (long long) n_kh, (long long) s, scale);
    return hipGetLastError() == hipSuccess;
}

bool gated_norm(const float * o, const float * w, const float * z, float * out, const int64_t rows,
                const int64_t n, const float eps, hipStream_t stream) {
    if (o == nullptr || z == nullptr || out == nullptr || rows <= 0 || n <= 0) {
        return false;
    }
    gated_norm_kernel<float><<<(unsigned) rows, kBlock, kBlock * sizeof(float), stream>>>(o, w, z, out,
                                                                                   (long long) n,
                                                                                   eps);
    return hipGetLastError() == hipSuccess;
}

bool gated_norm_f16(const float * o, const float * w, const float * z, void * out_f16,
                    const int64_t rows, const int64_t n, const float eps, hipStream_t stream) {
    if (o == nullptr || z == nullptr || out_f16 == nullptr || rows <= 0 || n <= 0) {
        return false;
    }
    gated_norm_kernel<__half><<<(unsigned) rows, kBlock, kBlock * sizeof(float), stream>>>(
        o, w, z, static_cast<__half *>(out_f16), (long long) n, eps);
    return hipGetLastError() == hipSuccess;
}

bool delta_step_fused(float * state, const float * q, const float * k, const float * v,
                      const float * beta, const float * alpha, const float * dt_bias,
                      const float * ssm_a, float * o, const int64_t heads, const int64_t n_kh,
                      const int64_t s, const float scale, hipStream_t stream) {
    if (state == nullptr || q == nullptr || k == nullptr || v == nullptr || beta == nullptr ||
        alpha == nullptr || dt_bias == nullptr || ssm_a == nullptr || o == nullptr ||
        heads <= 0 || n_kh <= 0 || s != 128) {
        return false;  // the kernel's thread mapping is written for s == 128
    }
    delta_step_kernel<<<(unsigned) heads, 256, 0, stream>>>(state, q, k, v, beta, alpha, dt_bias,
                                                            ssm_a, o, n_kh, s, scale);
    return hipGetLastError() == hipSuccess;
}

bool conv_silu_split_l2(const float * qkv, const float * w, const float * state, float * new_state,
                        float * q, float * k, float * v, const int64_t tokens,
                        const int64_t channels, const int64_t kernel, const int64_t q_dims,
                        const int64_t kv_dims, const int64_t v_dims, const int64_t head_dim,
                        const float eps, const float scale, hipStream_t stream) {
    if (qkv == nullptr || w == nullptr || state == nullptr || new_state == nullptr ||
        q == nullptr || k == nullptr || v == nullptr || tokens <= 0 || kernel <= 1 ||
        head_dim != kConvHead || q_dims % kConvHead != 0 || kv_dims % kConvHead != 0 ||
        v_dims % kConvHead != 0 || channels != q_dims + kv_dims + v_dims) {
        return false;
    }
    const dim3 grid((unsigned) tokens, (unsigned) (channels / kConvHead));
    conv_silu_split_l2_kernel<<<grid, kConvHead, 0, stream>>>(
        qkv, w, state, new_state, q, k, v, (long long) tokens, (long long) channels,
        (long long) kernel, (long long) q_dims, (long long) kv_dims, (long long) v_dims, eps,
        scale);
    return hipGetLastError() == hipSuccess;
}

} // namespace omph::kernels
