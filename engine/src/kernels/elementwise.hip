// Elementwise / reduction kernels for the naive path.
#include "kernels/elementwise.hh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace omph::kernels {
namespace {

constexpr int kBlock = 256;

__global__ void cast_f32_to_f16_kernel(const float * __restrict__ src, __half * __restrict__ dst,
                                       const long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

// One block per row; blockDim.x must be a power of two.
__global__ void rms_norm_kernel(const float * __restrict__ x, const float * __restrict__ w,
                                float * __restrict__ y, const long long n, const float eps,
                                const float scale) {
    extern __shared__ float sdata[];
    const long long row = blockIdx.x;
    const float * xr = x + row * n;
    float * yr = y + row * n;

    float acc = 0.0f;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        acc += v * v;
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if ((int) threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(sdata[0] / (float) n + eps);
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        yr[i] = xr[i] * inv * (w != nullptr ? w[i] : 1.0f) * scale;
    }
}

__global__ void add_inplace_kernel(float * __restrict__ x, const float * __restrict__ y,
                                   const long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += y[i];
    }
}

__global__ void add_out_kernel(const float * __restrict__ x, const float * __restrict__ y,
                               float * __restrict__ out, const long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] + y[i];
    }
}


// out = rms_norm(a + b) * w in f16, optionally also writing the f32 sum a + b
// (the residual stream). The activation only ever feeds a GEMV, so the f16
// output saves the f32 round trip; the fused add saves a launch per layer.
// The decode calls this with one row = one workgroup, so latency is the whole
// cost (#66): 1024 threads with float4 loads keep at most two vectors each in
// registers across the reduction, instead of 256 threads re-reading the row.
constexpr int kNormThreads = 1024;
constexpr int kNormVecs = 2;  // float4s per thread: rows up to 8192

__global__ __launch_bounds__(kNormThreads) void add_rms_norm_f16_kernel(
    const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ sum_out,
    const float * __restrict__ w, __half * __restrict__ y, const long long n, const float eps) {
    __shared__ float warp_sums[kNormThreads / 32];
    const long long row = blockIdx.x;
    const long long n4 = n / 4;
    const float4 * a4 = reinterpret_cast<const float4 *>(a + row * n);
    const float4 * b4 = b != nullptr ? reinterpret_cast<const float4 *>(b + row * n) : nullptr;
    float4 v[kNormVecs];
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < kNormVecs; ++k) {
        const long long i = threadIdx.x + (long long) k * kNormThreads;
        if (i < n4) {
            float4 t = a4[i];
            if (b4 != nullptr) {
                const float4 u = b4[i];
                t.x += u.x;
                t.y += u.y;
                t.z += u.z;
                t.w += u.w;
            }
            v[k] = t;
            acc = fmaf(t.x, t.x, fmaf(t.y, t.y, fmaf(t.z, t.z, fmaf(t.w, t.w, acc))));
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor(acc, off);
    }
    const int lane = (int) threadIdx.x & 31;
    const int warp = (int) threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = acc;
    }
    __syncthreads();
    float tot = lane < kNormThreads / 32 ? warp_sums[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        tot += __shfl_xor(tot, off);
    }
    const float inv = 1.0f / sqrtf(tot / (float) n + eps);
    const float4 * w4 = reinterpret_cast<const float4 *>(w);
    float4 * s4 = sum_out != nullptr ? reinterpret_cast<float4 *>(sum_out + row * n) : nullptr;
    __half2 * y2 = reinterpret_cast<__half2 *>(y + row * n);
#pragma unroll
    for (int k = 0; k < kNormVecs; ++k) {
        const long long i = threadIdx.x + (long long) k * kNormThreads;
        if (i < n4) {
            if (s4 != nullptr) {
                s4[i] = v[k];
            }
            const float4 g = w4[i];
            y2[2 * i] = __floats2half2_rn(v[k].x * inv * g.x, v[k].y * inv * g.y);
            y2[2 * i + 1] = __floats2half2_rn(v[k].z * inv * g.z, v[k].w * inv * g.w);
        }
    }
}

__global__ void swiglu_f16_kernel(const float * __restrict__ x, const float * __restrict__ y,
                                  __half * __restrict__ out, const long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = x[i];
        out[i] = __float2half((g / (1.0f + expf(-g))) * y[i]);
    }
}

__global__ void wave_size_kernel(int * out) {
    if (threadIdx.x == 0) {
        *out = __builtin_amdgcn_wavefrontsize();
    }
}

} // namespace

int kernel_wave_size() {
    int * dev = nullptr;
    int host = 0;
    if (hipMalloc(&dev, sizeof(int)) != hipSuccess) {
        return 0;
    }
    wave_size_kernel<<<1, 64, 0, nullptr>>>(dev);
    const bool ok = hipGetLastError() == hipSuccess &&
                    hipMemcpy(&host, dev, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess;
    (void) hipFree(dev);
    return ok ? host : 0;
}

bool cast_f32_to_f16(const float * src, void * dst, const int64_t n, hipStream_t stream) {
    if (src == nullptr || dst == nullptr || n <= 0) {
        return false;
    }
    const long long grid = (n + kBlock - 1) / kBlock;
    cast_f32_to_f16_kernel<<<(unsigned) grid, kBlock, 0, stream>>>(
        src, reinterpret_cast<__half *>(dst), (long long) n);
    return hipGetLastError() == hipSuccess;
}

bool rms_norm(const float * x, const float * w, float * y, const int64_t rows, const int64_t n,
              const float eps, const float scale, hipStream_t stream) {
    if (x == nullptr || y == nullptr || rows <= 0 || n <= 0) {
        return false;
    }
    rms_norm_kernel<<<(unsigned) rows, kBlock, kBlock * sizeof(float), stream>>>(
        x, w, y, (long long) n, eps, scale);
    return hipGetLastError() == hipSuccess;
}

bool add_inplace(float * x, const float * y, const int64_t n, hipStream_t stream) {
    if (x == nullptr || y == nullptr || n <= 0) {
        return false;
    }
    add_inplace_kernel<<<(unsigned) ((n + kBlock - 1) / kBlock), kBlock, 0, stream>>>(x, y,
                                                                                      (long long) n);
    return hipGetLastError() == hipSuccess;
}

bool add_out(const float * x, const float * y, float * out, const int64_t n, hipStream_t stream) {
    if (x == nullptr || y == nullptr || out == nullptr || n <= 0) {
        return false;
    }
    add_out_kernel<<<(unsigned) ((n + kBlock - 1) / kBlock), kBlock, 0, stream>>>(x, y, out,
                                                                                  (long long) n);
    return hipGetLastError() == hipSuccess;
}

bool add_rms_norm_f16(const float * a, const float * b, float * sum_out, const float * w,
                      void * y_f16, const int64_t rows, const int64_t n, const float eps,
                      hipStream_t stream) {
    if (a == nullptr || w == nullptr || y_f16 == nullptr || rows <= 0 || n <= 0 || n % 4 != 0 ||
        n > 4LL * kNormThreads * kNormVecs) {
        return false;
    }
    add_rms_norm_f16_kernel<<<(unsigned) rows, kNormThreads, 0, stream>>>(
        a, b, sum_out, w, reinterpret_cast<__half *>(y_f16), (long long) n, eps);
    return hipGetLastError() == hipSuccess;
}

bool swiglu_f16(const float * x, const float * y, void * out_f16, const int64_t n,
                hipStream_t stream) {
    if (x == nullptr || y == nullptr || out_f16 == nullptr || n <= 0) {
        return false;
    }
    swiglu_f16_kernel<<<(unsigned) ((n + kBlock - 1) / kBlock), kBlock, 0, stream>>>(
        x, y, reinterpret_cast<__half *>(out_f16), (long long) n);
    return hipGetLastError() == hipSuccess;
}

} // namespace omph::kernels
