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

__global__ void mul_inplace_kernel(float * __restrict__ x, const float * __restrict__ y,
                                   const long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] *= y[i];
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


// Same as rms_norm_kernel but writes f16: the activation only ever feeds a GEMV,
// so the f32 round trip through global memory is wasted work.
__global__ void rms_norm_f16_kernel(const float * __restrict__ x, const float * __restrict__ w,
                                    __half * __restrict__ y, const long long n, const float eps,
                                    const float scale) {
    extern __shared__ float sdata[];
    const long long row = blockIdx.x;
    const float * xr = x + row * n;
    __half * yr = y + row * n;
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
        yr[i] = __float2half(xr[i] * inv * (w != nullptr ? w[i] : 1.0f) * scale);
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

} // namespace

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

bool mul_inplace(float * x, const float * y, const int64_t n, hipStream_t stream) {
    if (x == nullptr || y == nullptr || n <= 0) {
        return false;
    }
    mul_inplace_kernel<<<(unsigned) ((n + kBlock - 1) / kBlock), kBlock, 0, stream>>>(x, y,
                                                                                      (long long) n);
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

bool rms_norm_f16(const float * x, const float * w, void * y_f16, const int64_t rows,
                  const int64_t n, const float eps, const float scale, hipStream_t stream) {
    if (x == nullptr || y_f16 == nullptr || rows <= 0 || n <= 0) {
        return false;
    }
    rms_norm_f16_kernel<<<(unsigned) rows, kBlock, kBlock * sizeof(float), stream>>>(
        x, w, reinterpret_cast<__half *>(y_f16), (long long) n, eps, scale);
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
