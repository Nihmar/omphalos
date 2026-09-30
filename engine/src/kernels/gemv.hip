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

constexpr long long kQ4kChunkSub = 256;  // 8192 weights staged per LDS refill
constexpr long long kQ4kRows = 4;        // output rows computed per warp

__global__ void gemv_q4k_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ sc,
                                const uint8_t * __restrict__ mn, const uint8_t * __restrict__ d,
                                const uint8_t * __restrict__ dmin, const __half * __restrict__ x,
                                float * __restrict__ y, const long long rows,
                                const long long n_sub) {
    extern __shared__ __half xs[];  // activation chunk

    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    // kQ4kRows consecutive output rows per warp: one activation value feeds all
    // of them and the accumulators give independent FMA chains.
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * kQ4kRows;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;

    float acc[kQ4kRows] = {};
    for (long long base = 0; base < n_sub; base += kQ4kChunkSub) {
        const long long len = n_sub - base < kQ4kChunkSub ? n_sub - base : kQ4kChunkSub;
        for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
            xs[i] = x[base * 32 + i];
        }
        __syncthreads();
        if (row0 >= rows) {
            __syncthreads();  // keep the barrier count identical across warps
            continue;
        }
        for (long long s = base + lane; s < base + len; s += 32) {
            const long long blk = s / 8;
            const int sub = (int) (s % 8);
            const __half * xv = xs + (s - base) * 32;
            uint4 q[kQ4kRows];
            float ds[kQ4kRows];
            float ms[kQ4kRows];
#pragma unroll
            for (int r = 0; r < kQ4kRows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;
                q[r] = *reinterpret_cast<const uint4 *>(qs + row * blocks * 128 + s * 16);
                ds[r] = __half2float(*reinterpret_cast<const __half *>(d + row * blocks * 2 +
                                                                       2 * blk)) *
                        (float) sc[row * blocks * 8 + blk * 8 + sub];
                ms[r] = __half2float(
                            *reinterpret_cast<const __half *>(dmin + row * blocks * 2 + 2 * blk)) *
                        (float) mn[row * blocks * 8 + blk * 8 + sub];
            }
            float part[kQ4kRows] = {};
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                // byte i: value 2i in the low nibble, 2i+1 in the high one
                const float x0 = __half2float(xv[2 * i]);
                const float x1 = __half2float(xv[2 * i + 1]);
#pragma unroll
                for (int r = 0; r < kQ4kRows; ++r) {
                    const uint8_t b = reinterpret_cast<const uint8_t *>(&q[r])[i];
                    part[r] = fmaf(fmaf(ds[r], (float) (b & 0xF), -ms[r]), x0, part[r]);
                    part[r] = fmaf(fmaf(ds[r], (float) (b >> 4), -ms[r]), x1, part[r]);
                }
            }
#pragma unroll
            for (int r = 0; r < kQ4kRows; ++r) {
                acc[r] += part[r];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kQ4kRows; ++r) {
        float a = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a += __shfl_down(a, off);
        }
        if (lane == 0 && row0 + r < rows) {
            y[row0 + r] = a;
        }
    }
}

__device__ const int8_t kIq4Table[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                         1,    13,   25,  38,  53,  69,  89,  113};

constexpr long long kIq4ChunkSub = 256;  // 8192 weights staged per LDS refill
constexpr long long kIq4Rows = 4;        // output rows computed per warp

__global__ void gemv_iq4_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ sc,
                                const uint8_t * __restrict__ d, const __half * __restrict__ x,
                                float * __restrict__ y, const long long rows,
                                const long long n_sub) {
    extern __shared__ uint8_t smem[];
    __half * xs = reinterpret_cast<__half *>(smem);  // activation chunk
    float * table = reinterpret_cast<float *>(smem + kIq4ChunkSub * 64);
    if (threadIdx.x < 16) {
        table[threadIdx.x] = (float) kIq4Table[threadIdx.x];
    }

    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    // Each warp owns kIq4Rows consecutive output rows: the activation value is
    // loaded once and reused by all of them (outer product), and the four
    // accumulators give independent FMA chains.
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * kIq4Rows;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;

    float acc[kIq4Rows] = {};
    for (long long base = 0; base < n_sub; base += kIq4ChunkSub) {
        const long long len = n_sub - base < kIq4ChunkSub ? n_sub - base : kIq4ChunkSub;
        for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) {
            xs[i] = x[base * 32 + i];
        }
        __syncthreads();
        if (row0 >= rows) {
            __syncthreads();  // keep the barrier count identical across warps
            continue;
        }
        for (long long s = base + lane; s < base + len; s += 32) {
            const long long blk = s / 8;
            const int sub = (int) (s % 8);
            const __half * xv = xs + (s - base) * 32;
            uint4 q[kIq4Rows];
            float dl[kIq4Rows];
#pragma unroll
            for (int r = 0; r < kIq4Rows; ++r) {
                const long long row = row0 + r <= rlast ? row0 + r : rlast;  // clamped, never
                                                                             // used when masked
                q[r] = *reinterpret_cast<const uint4 *>(
                    qs + row * blocks * 128 + s * 16);
                dl[r] = __half2float(*reinterpret_cast<const __half *>(
                            d + row * blocks * 2 + 2 * blk)) *
                        (float) ((int) sc[row * blocks * 8 + blk * 8 + sub] - 32);
            }
            float part[kIq4Rows] = {};
#pragma unroll
            for (int v = 0; v < 16; ++v) {
                const float xval = __half2float(xv[v]);
#pragma unroll
                for (int r = 0; r < kIq4Rows; ++r) {
                    const uint8_t code = reinterpret_cast<const uint8_t *>(&q[r])[v] & 0xF;
                    part[r] = fmaf(table[code], xval, part[r]);
                }
            }
#pragma unroll
            for (int v = 0; v < 16; ++v) {
                const float xval = __half2float(xv[v + 16]);
#pragma unroll
                for (int r = 0; r < kIq4Rows; ++r) {
                    const uint8_t code = reinterpret_cast<const uint8_t *>(&q[r])[v] >> 4;
                    part[r] = fmaf(table[code], xval, part[r]);
                }
            }
#pragma unroll
            for (int r = 0; r < kIq4Rows; ++r) {
                acc[r] = fmaf(dl[r], part[r], acc[r]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kIq4Rows; ++r) {
        float a = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a += __shfl_down(a, off);
        }
        if (lane == 0 && row0 + r < rows) {
            y[row0 + r] = a;
        }
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
    const size_t smem = (size_t) kQ4kChunkSub * 64;
    const auto * base = static_cast<const uint8_t *>(packed);
    const int64_t n_sub = k / 32;
    const dim3 grid((unsigned) ((rows + 8 * kQ4kRows - 1) / (8 * kQ4kRows)));
    gemv_q4k_kernel<<<grid, 256, smem, stream>>>(
        base + l.qs_off, base + l.sc_off, base + l.mn_off, base + l.d_off, base + l.dmin_off,
        static_cast<const __half *>(x_f16), y, rows, n_sub);
    return hipGetLastError() == hipSuccess;
}

bool gemv_iq4_xs(const void * packed, const void * x_f16, float * y, const int64_t rows,
                 const int64_t k, hipStream_t stream) {
    if (packed == nullptr || x_f16 == nullptr || y == nullptr || rows <= 0 || k <= 0 ||
        k % 256 != 0) {
        return false;
    }
    const int64_t n_blocks = rows * (k / 256);
    const omph::format::Iq4Layout l = omph::format::iq4_layout(n_blocks);
    const int64_t n_sub = k / 32;
    const size_t smem = (size_t) kIq4ChunkSub * 64 + 16 * sizeof(float);
    if (smem > 64 * 1024) {
        return false;
    }
    const auto * base = static_cast<const uint8_t *>(packed);
    const dim3 grid((unsigned) ((rows + 8 * kIq4Rows - 1) / (8 * kIq4Rows)));
    gemv_iq4_kernel<<<grid, 256, smem, stream>>>(base + l.qs_off, base + l.sc_off, base + l.d_off,
                                                 static_cast<const __half *>(x_f16), y, rows,
                                                 n_sub);
    return hipGetLastError() == hipSuccess;
}

} // namespace omph::kernels
