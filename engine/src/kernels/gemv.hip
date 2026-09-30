// Fused dequant+dot GEMV for the repacked Q4_K layout.
//
// One warp per output row (8 warps per workgroup). The activation vector is
// staged in LDS as f16; each lane walks 32-weight sub-blocks in strides of 32
// and loads 16 contiguous, aligned bytes per sub-block. Weights are dequantized
// in registers, so quantized bytes are read exactly once and nothing else
// touches memory.
#include "kernels/gemv.hh"

#include "format/repack.hh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace omph::kernels {
namespace {

__global__ void gemv_q4k_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ sc,
                                const uint8_t * __restrict__ mn, const uint8_t * __restrict__ d,
                                const uint8_t * __restrict__ dmin, const __half * __restrict__ x,
                                float * __restrict__ y, const long long rows,
                                const long long n_sub) {
    extern __shared__ __half xs[];
    const long long k = n_sub * 32;
    for (long long i = threadIdx.x; i < k; i += blockDim.x) {
        xs[i] = x[i];
    }
    __syncthreads();

    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row = (long long) blockIdx.x * (blockDim.x / 32) + warp;
    if (row >= rows) {
        return;
    }
    const long long blocks = n_sub / 8;
    const uint8_t * rqs = qs + row * blocks * 128;
    const uint8_t * rsc = sc + row * blocks * 8;
    const uint8_t * rmn = mn + row * blocks * 8;
    const uint8_t * rd = d + row * blocks * 2;
    const uint8_t * rdmin = dmin + row * blocks * 2;

    float acc = 0.0f;
    for (long long s = lane; s < n_sub; s += 32) {
        const long long blk = s / 8;
        const int sub = (int) (s % 8);
        const float ds = __half2float(*reinterpret_cast<const __half *>(rd + 2 * blk)) *
                         (float) rsc[blk * 8 + sub];
        const float ms = __half2float(*reinterpret_cast<const __half *>(rdmin + 2 * blk)) *
                         (float) rmn[blk * 8 + sub];
        const uint4 p = *reinterpret_cast<const uint4 *>(rqs + s * 16);
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(&p);
        const __half * xv = xs + s * 32;
        float part = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t b = bytes[i];
            const float w0 = fmaf(ds, (float) (b & 0xF), -ms);
            const float w1 = fmaf(ds, (float) (b >> 4), -ms);
            part = fmaf(w0, __half2float(xv[2 * i]), part);
            part = fmaf(w1, __half2float(xv[2 * i + 1]), part);
        }
        acc += part;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    if (lane == 0) {
        y[row] = acc;
    }
}

} // namespace

bool gemv_q4k(const void * packed, const void * x_f16, float * y, const int64_t rows,
              const int64_t k, hipStream_t stream) {
    if (packed == nullptr || x_f16 == nullptr || y == nullptr || rows <= 0 || k <= 0 ||
        k % 256 != 0) {
        return false;
    }
    const int64_t n_blocks = rows * (k / 256);
    const omph::format::Q4kLayout l = omph::format::q4k_layout(n_blocks);
    const size_t smem = (size_t) k * 2;
    if (smem > 64 * 1024) {
        return false;
    }
    const auto * base = static_cast<const uint8_t *>(packed);
    const int64_t n_sub = k / 32;
    const dim3 grid((unsigned) ((rows + 7) / 8));
    gemv_q4k_kernel<<<grid, 256, smem, stream>>>(
        base + l.qs_off, base + l.sc_off, base + l.mn_off, base + l.d_off, base + l.dmin_off,
        static_cast<const __half *>(x_f16), y, rows, n_sub);
    return hipGetLastError() == hipSuccess;
}

} // namespace omph::kernels
