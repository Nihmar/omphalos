// Read bandwidth of the GEMV access patterns (#63), no decoding: a warp walks
// R rows' weights in sub-block steps, each lane loading B bytes per row per step,
// over a ~3.4 GB matrix (IQ3_S-sized). Layouts:
//   rows        each row contiguous; a warp's R rows are R separate streams (today)
//   interleaved the R rows a warp reads together are interleaved per sub-block:
//               one contiguous stream per warp
//   hipcc -O3 --offload-arch=gfx1200 -o stream_layout bench/gemv/stream_layout.hip && ./stream_layout
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>

template <int R, int B, bool kInter>
__global__ void walk(const unsigned char * __restrict__ w, const long long rows, const long long n_sub,
                     float * __restrict__ y) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * R;
    if (row0 >= rows) return;
    unsigned acc = 0;
    for (long long s = lane; s < n_sub; s += 32) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const long long off = kInter ? (((row0 / R) * n_sub + s) * R + r) * B : ((row0 + r) * n_sub + s) * B;
            if constexpr (B == 16) {
                const uint4 v = *reinterpret_cast<const uint4 *>(w + off);
                acc ^= v.x ^ v.y ^ v.z ^ v.w;
            } else if constexpr (B == 8) {
                const uint2 v = *reinterpret_cast<const uint2 *>(w + off);
                acc ^= v.x ^ v.y;
            } else {
                acc ^= *reinterpret_cast<const unsigned *>(w + off);
            }
        }
    }
    if (acc == 0x12345678u) y[0] = 1.0f;  // keep the loads
}

// the GEMV's shape: each workgroup stages x (k halves) in LDS, then its 8 warps
// walk R rows each; `groups` row groups per workgroup (1: today's one-shot
// workgroups; more: x staged once for several groups)
template <int R, int B>
__global__ void walk_staged(const unsigned char * __restrict__ w, const __half * __restrict__ x,
                            const long long rows, const long long n_sub, const int groups, float * __restrict__ y) {
    extern __shared__ __half xs[];
    for (long long i = threadIdx.x; i < n_sub * 32; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    unsigned acc = 0;
    for (int g = 0; g < groups; ++g) {
        const long long row0 = (((long long) blockIdx.x * groups + g) * (blockDim.x / 32) + warp) * R;
        if (row0 >= rows) break;
        for (long long s = lane; s < n_sub; s += 32) {
            const uint4 xv = *reinterpret_cast<const uint4 *>(xs + s * 32);
            acc ^= xv.x;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const long long off = ((row0 + r) * n_sub + s) * B;
                const uint2 v = *reinterpret_cast<const uint2 *>(w + off);
                acc ^= v.x ^ v.y;
            }
        }
    }
    if (acc == 0x12345678u) y[0] = 1.0f;
}

double run_staged(const unsigned char * w, const __half * x, long long bytes, int groups, float * y) {
    const long long n_sub = 5120 / 32;
    const long long rows = bytes / (n_sub * 8) / 4 * 4;
    const long long wgs = (rows / 4 + 7) / 8;
    const dim3 grid((unsigned) ((wgs + groups - 1) / groups));
    hipEvent_t a, b;
    (void) hipEventCreate(&a);
    (void) hipEventCreate(&b);
    for (int i = 0; i < 20; ++i) walk_staged<4, 8><<<grid, 256, n_sub * 64>>>(w, x, rows, n_sub, groups, y);
    (void) hipEventRecord(a);
    for (int i = 0; i < 20; ++i) walk_staged<4, 8><<<grid, 256, n_sub * 64>>>(w, x, rows, n_sub, groups, y);
    (void) hipEventRecord(b);
    (void) hipEventSynchronize(b);
    float ms = 0;
    (void) hipEventElapsedTime(&ms, a, b);
    return (double) rows * n_sub * 8 / (ms / 20) / 1e6;
}

template <int R, int B, bool kInter>
double run(const unsigned char * w, long long bytes, float * y) {
    const long long n_sub = 5120 / 32;  // a 5120-wide row
    const long long rows = bytes / (n_sub * B) / R * R;
    const dim3 grid((unsigned) ((rows / R + 7) / 8));
    hipEvent_t a, b;
    (void) hipEventCreate(&a);
    (void) hipEventCreate(&b);
    for (int i = 0; i < 20; ++i) walk<R, B, kInter><<<grid, 256>>>(w, rows, n_sub, y);  // warm-up
    (void) hipEventRecord(a);
    for (int i = 0; i < 20; ++i) walk<R, B, kInter><<<grid, 256>>>(w, rows, n_sub, y);
    (void) hipEventRecord(b);
    (void) hipEventSynchronize(b);
    float ms = 0;
    (void) hipEventElapsedTime(&ms, a, b);
    return (double) rows * n_sub * B / (ms / 20) / 1e6;
}

int main() {
    const long long bytes = 3474LL << 20;
    unsigned char * w = nullptr;
    float * y = nullptr;
    if (hipMalloc(&w, bytes + (1 << 20)) != hipSuccess || hipMalloc(&y, 4) != hipSuccess) return 1;
    (void) hipMemset(w, 1, bytes);
    std::printf("GB/s reading %lld MiB (bytes per lane per row per step, rows per warp)\n", bytes >> 20);
    std::printf("B=8  R=4  rows %.1f  interleaved %.1f\n", run<4, 8, false>(w, bytes, y), run<4, 8, true>(w, bytes, y));
    std::printf("B=16 R=4  rows %.1f  interleaved %.1f\n", run<4, 16, false>(w, bytes, y), run<4, 16, true>(w, bytes, y));
    std::printf("B=4  R=4  rows %.1f  interleaved %.1f\n", run<4, 4, false>(w, bytes, y), run<4, 4, true>(w, bytes, y));
    std::printf("B=8  R=2  rows %.1f  interleaved %.1f\n", run<2, 8, false>(w, bytes, y), run<2, 8, true>(w, bytes, y));
    std::printf("B=8  R=8  rows %.1f  interleaved %.1f\n", run<8, 8, false>(w, bytes, y), run<8, 8, true>(w, bytes, y));
    std::printf("B=16 R=1  rows %.1f\n", run<1, 16, false>(w, bytes, y));
    __half * x = nullptr;
    if (hipMalloc(&x, 5120 * 2) != hipSuccess) return 1;
    (void) hipMemset(x, 0, 5120 * 2);
    for (int g : {1, 2, 4, 8, 16}) {
        std::printf("x staged in LDS, %2d row group(s) per workgroup: %.1f\n", g, run_staged(w, x, bytes, g, y));
    }
    return 0;
}
