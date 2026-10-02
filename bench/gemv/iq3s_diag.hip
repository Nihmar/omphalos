// Where the IQ3_S GEMV's time goes (#174): the engine's three IQ3_S kernels
// (single token with persistent warps, single token with x chunks for
// ffn_down, the 4-token verification kernel), copied with ablation switches,
// run over random weights shaped like the model's 144 IQ3_S tensors.
//
//   full     the kernel as in gemv.hip
//   no-w     no weight loads: the codes come from a hash of (row, sub-block)
//   no-grid  no LDS grid lookups: the 9-bit index stands in for the grid word
//   no-x     no activation loads (LDS or global): constants
//   loads    weight loads only (all five arrays), no decoding, no x, no dots
//   empty    an empty kernel with the same grid: the launch cost alone
//   sgpr     full, with the warp-uniform row bases in SGPRs (32-bit lane offsets)
//
// Plus per-warp timestamps (wall_clock64) of the full kernel: the staging
// before the loop, the loop, the reduction and stores, and per launch the
// spread of the warps' start and end times.
//
//   hipcc -O3 --offload-arch=gfx1200 -std=c++20 -I engine/src -o /var/tmp/iq3s_diag bench/gemv/iq3s_diag.hip
//   /var/tmp/iq3s_diag
#include "kernels/quant_decode.hh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using omph::kernels::qd::grid_pair_to_half2;
using omph::kernels::qd::half2_v;

namespace {

__device__ inline float dot2(const uint32_t w, const uint32_t x, const float acc) {
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(half2_v, w), __builtin_bit_cast(half2_v, x), acc, false);
}

constexpr long long kChunkSub = 256;
constexpr int kRows = 4;

enum : int { kFull = 0, kNoW = 1, kNoGrid = 2, kNoX = 4, kLoads = 8, kTime = 16, kSgpr = 32 };

struct W {
    const uint8_t * qs;
    const uint8_t * qh;
    const uint8_t * signs;
    const uint8_t * scales;
    const uint8_t * d;
};

__device__ inline unsigned long long stamp() { return wall_clock64(); }

// one sub-block of R rows: the weight loads (or their stand-ins)
template <int M, int R>
__device__ inline void load_rows(const W & w, const long long row0, const long long rlast, const long long blocks,
                                 const long long s, uint2 * q, uint32_t * qhb, uint32_t * sg, float * dbr) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const long long row = row0 + r <= rlast ? row0 + r : rlast;
        if constexpr (M & kNoW) {
            const uint32_t h = (uint32_t) (row * 0x9E3779B1u) ^ (uint32_t) (s * 0x85EBCA77u);
            q[r] = make_uint2(h, h * 0xC2B2AE3Du);
            qhb[r] = h >> 24;
            sg[r] = h * 0x27D4EB2Fu;
            dbr[r] = 0.01f * (float) ((h >> 8) & 15);
        } else if constexpr (M & kSgpr) {
            // the row is warp-uniform: its base pointers in SGPRs, 32-bit lane offsets
            const long long ru = __builtin_amdgcn_readfirstlane((int) row);
            const long long bu = __builtin_amdgcn_readfirstlane((int) blocks);
            const uint32_t so = (uint32_t) s;
            q[r] = *reinterpret_cast<const uint2 *>(w.qs + ru * bu * 64 + so * 8u);
            qhb[r] = (w.qh + ru * bu * 8)[so];
            sg[r] = *reinterpret_cast<const uint32_t *>(w.signs + ru * bu * 32 + so * 4u);
            const uint8_t sc = (w.scales + ru * bu * 4)[so / 2u];
            const float dv = __half2float(*reinterpret_cast<const __half *>(w.d + ru * bu * 2 + (so / 8u) * 2u));
            dbr[r] = dv * (1.0f + 2.0f * (float) ((s & 1) ? (sc >> 4) : (sc & 0xF)));
        } else {
            q[r] = *reinterpret_cast<const uint2 *>(w.qs + row * blocks * 64 + s * 8);
            qhb[r] = w.qh[row * blocks * 8 + s];
            sg[r] = *reinterpret_cast<const uint32_t *>(w.signs + row * blocks * 32 + s * 4);
            const uint8_t sc = w.scales[row * blocks * 4 + s / 2];
            const float dv = __half2float(*reinterpret_cast<const __half *>(w.d + row * blocks * 2 + (s / 8) * 2));
            dbr[r] = dv * (1.0f + 2.0f * (float) ((s & 1) ? (sc >> 4) : (sc & 0xF)));
        }
    }
}

template <int M>
__device__ inline uint32_t grid_word(const uint32_t * grid, const uint32_t i) {
    if constexpr (M & kNoGrid) return i * 0x01010101u;
    else return grid[i];
}

// single token, x in LDS (xv: the sub-block's 4 uint4), R = 4: the gemv.hip loop body
template <int M>
__device__ inline void body1(const uint32_t * grid, const uint4 * xv, const uint2 * q, const uint32_t * qhb,
                             const uint32_t * sg, const float * dbr, float * acc, const long long s) {
    float part[kRows] = {};
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        uint4 xq;
        if constexpr (M & kNoX) {
            const uint32_t c = 0x3C003C00u ^ (uint32_t) (s + l);
            xq = make_uint4(c, c ^ 1u, c ^ 2u, c ^ 3u);
        } else {
            xq = xv[l];
        }
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const uint32_t lo = (q[r].x >> (16 * (l & 1))) & 0xFFFFu;
            const uint32_t word = l < 2 ? lo : (q[r].y >> (16 * (l & 1))) & 0xFFFFu;
            const uint32_t i1 = (word & 0xFFu) | (((qhb[r] >> (2 * l)) & 1u) << 8);
            const uint32_t i2 = (word >> 8) | (((qhb[r] >> (2 * l + 1)) & 1u) << 8);
            const uint32_t b1 = grid_word<M>(grid, i1);
            const uint32_t b2 = grid_word<M>(grid, i2);
            const uint2 g1 = make_uint2(grid_pair_to_half2<0x04010400u>(b1), grid_pair_to_half2<0x04030402u>(b1));
            const uint2 g2 = make_uint2(grid_pair_to_half2<0x04010400u>(b2), grid_pair_to_half2<0x04030402u>(b2));
            constexpr uint32_t kSign2 = 0x80008000u;
            float p = part[r];
            p = dot2(((sg[r] << (4 * l)) & kSign2) | g1.x, xq.x, p);
            p = dot2(((sg[r] << (4 * l + 1)) & kSign2) | g1.y, xq.y, p);
            p = dot2(((sg[r] << (4 * l + 2)) & kSign2) | g2.x, xq.z, p);
            p = dot2(((sg[r] << (4 * l + 3)) & kSign2) | g2.y, xq.w, p);
            part[r] = p;
        }
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = fmaf(dbr[r], part[r], acc[r]);
}

template <int R>
__device__ inline void fold_loads(const uint2 * q, const uint32_t * qhb, const uint32_t * sg, const float * dbr,
                                  float * acc) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
        acc[r] += __uint_as_float((q[r].x ^ q[r].y ^ qhb[r] ^ sg[r]) & 0x007FFFFFu) + dbr[r];
    }
}

__device__ inline void put_stamps(unsigned long long * ts, const unsigned long long * t) {
    const int lane = (int) threadIdx.x % 32;
    if (lane == 0) {
        const long long id = (long long) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
        for (int i = 0; i < 4; ++i) ts[id * 4 + i] = t[i];
    }
}

// gemv_iq3s_pw_kernel
template <int M>
__global__ void k_pw(const W w, const __half * __restrict__ x, float * __restrict__ y, const long long rows,
                     const long long n_sub, unsigned long long * ts) {
    unsigned long long t[4] = {};
    if constexpr (M & kTime) t[0] = stamp();
    extern __shared__ uint8_t smem[];
    __half * xs = reinterpret_cast<__half *>(smem);
    uint32_t * grid = reinterpret_cast<uint32_t *>(smem + n_sub * 64);
    for (int i = threadIdx.x; i < 512; i += blockDim.x) grid[i] = omph::quant::kIq3sGrid[i];
    for (long long i = threadIdx.x; i < n_sub * 32; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    if constexpr (M & kTime) t[1] = stamp();
    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;
    const long long n_groups = (rows + kRows - 1) / kRows;
    const long long stride = (long long) gridDim.x * (blockDim.x / 32);
    for (long long grp = (long long) blockIdx.x * (blockDim.x / 32) + warp; grp < n_groups; grp += stride) {
        const long long row0 = grp * kRows;
        float acc[kRows] = {};
        for (long long s = lane; s < n_sub; s += 32) {
            const uint4 * xv = reinterpret_cast<const uint4 *>(xs + s * 32);
            uint2 q[kRows];
            uint32_t qhb[kRows], sg[kRows];
            float dbr[kRows];
            load_rows<M, kRows>(w, row0, rlast, blocks, s, q, qhb, sg, dbr);
            if constexpr (M & kLoads) fold_loads<kRows>(q, qhb, sg, dbr, acc);
            else body1<M>(grid, xv, q, qhb, sg, dbr, acc, s);
        }
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            float v = acc[r];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off);
            if (lane == 0 && row0 + r < rows) y[row0 + r] = v;
        }
    }
    if constexpr (M & kTime) {
        t[2] = t[3] = stamp();
        put_stamps(ts, t);
    }
}

// gemv_iq3s_kernel (x in chunks of 256 sub-blocks: ffn_down)
template <int M>
__global__ void k_chunk(const W w, const __half * __restrict__ x, float * __restrict__ y, const long long rows,
                        const long long n_sub, unsigned long long * ts) {
    unsigned long long t[4] = {};
    if constexpr (M & kTime) t[0] = stamp();
    extern __shared__ uint8_t smem[];
    __half * xs = reinterpret_cast<__half *>(smem);
    uint32_t * grid = reinterpret_cast<uint32_t *>(smem + kChunkSub * 64);
    for (int i = threadIdx.x; i < 512; i += blockDim.x) grid[i] = omph::quant::kIq3sGrid[i];
    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long row0 = ((long long) blockIdx.x * (blockDim.x / 32) + warp) * kRows;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;
    float acc[kRows] = {};
    unsigned long long staged = 0;
    for (long long base = 0; base < n_sub; base += kChunkSub) {
        unsigned long long a = 0;
        if constexpr (M & kTime) a = stamp();
        const long long len = n_sub - base < kChunkSub ? n_sub - base : kChunkSub;
        for (long long i = threadIdx.x; i < len * 32; i += blockDim.x) xs[i] = x[base * 32 + i];
        __syncthreads();
        if constexpr (M & kTime) staged += stamp() - a;
        if (row0 >= rows) {
            __syncthreads();
            continue;
        }
        for (long long s = base + lane; s < base + len; s += 32) {
            const uint4 * xv = reinterpret_cast<const uint4 *>(xs + (s - base) * 32);
            uint2 q[kRows];
            uint32_t qhb[kRows], sg[kRows];
            float dbr[kRows];
            load_rows<M, kRows>(w, row0, rlast, blocks, s, q, qhb, sg, dbr);
            if constexpr (M & kLoads) fold_loads<kRows>(q, qhb, sg, dbr, acc);
            else body1<M>(grid, xv, q, qhb, sg, dbr, acc, s);
        }
        if constexpr (M & kTime) a = stamp();
        __syncthreads();
        if constexpr (M & kTime) staged += stamp() - a;
    }
    if constexpr (M & kTime) t[2] = stamp();
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off);
        if (lane == 0 && row0 + r < rows) y[row0 + r] = v;
    }
    if constexpr (M & kTime) {
        t[3] = stamp();
        t[1] = t[0] + staged;  // barrier + staging time, all chunks
        put_stamps(ts, t);
    }
}

// gemv_iq3s_ntg_kernel<4, 4>: x straight from global memory, grid in LDS
template <int M, int NT>
__global__ void k_ntg(const W w, const __half * __restrict__ x, float * __restrict__ y, const long long rows,
                      const long long n_sub, unsigned long long * ts) {
    constexpr int R = kRows;
    unsigned long long t[4] = {};
    if constexpr (M & kTime) t[0] = stamp();
    extern __shared__ uint8_t smem[];
    uint32_t * grid = reinterpret_cast<uint32_t *>(smem);
    for (int i = threadIdx.x; i < 512; i += blockDim.x) grid[i] = omph::quant::kIq3sGrid[i];
    __syncthreads();
    if constexpr (M & kTime) t[1] = stamp();
    const int warp = (int) threadIdx.x / 32;
    const int lane = (int) threadIdx.x % 32;
    const long long blocks = n_sub / 8;
    const long long rlast = rows - 1;
    const long long k = n_sub * 32;
    const long long n_groups = (rows + R - 1) / R;
    const long long stride = (long long) gridDim.x * (blockDim.x / 32);
    for (long long grp = (long long) blockIdx.x * (blockDim.x / 32) + warp; grp < n_groups; grp += stride) {
        const long long row0 = grp * R;
        float acc[R][NT] = {};
        for (long long s = lane; s < n_sub; s += 32) {
            uint2 q[R];
            uint32_t qhb[R], sg[R];
            float dbr[R];
            load_rows<M, R>(w, row0, rlast, blocks, s, q, qhb, sg, dbr);
            if constexpr (M & kLoads) {
                fold_loads<R>(q, qhb, sg, dbr, acc[0]);
                continue;
            }
            float part[R][NT] = {};
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                uint4 xq[NT];
#pragma unroll
                for (int tt = 0; tt < NT; ++tt) {
                    if constexpr (M & kNoX) {
                        const uint32_t c = 0x3C003C00u ^ (uint32_t) (s + l + tt);
                        xq[tt] = make_uint4(c, c ^ 1u, c ^ 2u, c ^ 3u);
                    } else {
                        xq[tt] = reinterpret_cast<const uint4 *>(x + tt * k + s * 32)[l];
                    }
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint32_t lo = (q[r].x >> (16 * (l & 1))) & 0xFFFFu;
                    const uint32_t word = l < 2 ? lo : (q[r].y >> (16 * (l & 1))) & 0xFFFFu;
                    const uint32_t i1 = (word & 0xFFu) | (((qhb[r] >> (2 * l)) & 1u) << 8);
                    const uint32_t i2 = (word >> 8) | (((qhb[r] >> (2 * l + 1)) & 1u) << 8);
                    const uint32_t b1 = grid_word<M>(grid, i1);
                    const uint32_t b2 = grid_word<M>(grid, i2);
                    constexpr uint32_t kSign2 = 0x80008000u;
                    const uint32_t w0 = ((sg[r] << (4 * l)) & kSign2) | grid_pair_to_half2<0x04010400u>(b1);
                    const uint32_t w1 = ((sg[r] << (4 * l + 1)) & kSign2) | grid_pair_to_half2<0x04030402u>(b1);
                    const uint32_t w2 = ((sg[r] << (4 * l + 2)) & kSign2) | grid_pair_to_half2<0x04010400u>(b2);
                    const uint32_t w3 = ((sg[r] << (4 * l + 3)) & kSign2) | grid_pair_to_half2<0x04030402u>(b2);
#pragma unroll
                    for (int tt = 0; tt < NT; ++tt) {
                        float p = part[r][tt];
                        p = dot2(w0, xq[tt].x, p);
                        p = dot2(w1, xq[tt].y, p);
                        p = dot2(w2, xq[tt].z, p);
                        p = dot2(w3, xq[tt].w, p);
                        part[r][tt] = p;
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int tt = 0; tt < NT; ++tt) acc[r][tt] = fmaf(dbr[r], part[r][tt], acc[r][tt]);
            }
        }
        if constexpr (M & kTime) t[2] = stamp();
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int tt = 0; tt < NT; ++tt) {
                float v = acc[r][tt];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off);
                if (lane == 0 && row0 + r < rows) y[tt * rows + row0 + r] = v;
            }
        }
    }
    if constexpr (M & kTime) {
        t[3] = stamp();
        put_stamps(ts, t);
    }
}

__global__ void k_empty() {}

struct Tensor {
    long long rows, k;
    std::string name;
    W w;
};

void check(const hipError_t e, const char * what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(1);
    }
}

int g_fit_pw = 0;

// one launch of the engine's kernel choice for this tensor
template <int M>
void launch(const Tensor & t, const int nt, const __half * x, float * y, unsigned long long * ts, bool empty,
            unsigned * n_warps = nullptr) {
    const long long n_sub = t.k / 32;
    dim3 grid;
    size_t smem;
    if (nt == 1 && n_sub <= kChunkSub) {
        const long long need = ((t.rows + kRows - 1) / kRows + 7) / 8;
        grid = dim3((unsigned) std::min<long long>(need, g_fit_pw));
        smem = (size_t) n_sub * 64 + 512 * 4;
        if (!empty) k_pw<M><<<grid, 256, smem>>>(t.w, x, y, t.rows, n_sub, ts);
    } else if (nt == 1) {
        grid = dim3((unsigned) ((t.rows + 8 * kRows - 1) / (8 * kRows)));
        smem = (size_t) kChunkSub * 64 + 512 * 4;
        if (!empty) k_chunk<M><<<grid, 256, smem>>>(t.w, x, y, t.rows, n_sub, ts);
    } else {
        grid = dim3((unsigned) ((t.rows + 8 * kRows - 1) / (8 * kRows)));
        smem = 512 * 4;
        if (!empty) k_ntg<M, 4><<<grid, 256, smem>>>(t.w, x, y, t.rows, n_sub, ts);
    }
    if (empty) k_empty<<<grid, 256, smem>>>();
    if (n_warps) *n_warps = grid.x * 8;
}

template <int M>
float pass_ms(const std::vector<const Tensor *> & ts, const int nt, const __half * x, float * y, bool empty,
              const int reps) {
    hipEvent_t a, b;
    check(hipEventCreate(&a), "event");
    check(hipEventCreate(&b), "event");
    std::vector<float> v;
    for (int i = 0; i < reps; ++i) {
        check(hipEventRecord(a, nullptr), "record");
        for (const Tensor * t : ts) launch<M>(*t, nt, x, y, nullptr, empty);
        check(hipEventRecord(b, nullptr), "record");
        check(hipEventSynchronize(b), "sync");
        float ms = 0;
        check(hipEventElapsedTime(&ms, a, b), "elapsed");
        v.push_back(ms);
    }
    check(hipGetLastError(), "launch");
    std::sort(v.begin(), v.end());
    (void) hipEventDestroy(a);
    (void) hipEventDestroy(b);
    return v[v.size() / 2];
}

} // namespace

int main() {
    const struct { long long rows, k; int n; const char * name; } shapes[] = {
        {1024, 5120, 7, "attn_k/v"},    {5120, 6144, 32, "attn_out/ssm_out"}, {5120, 17408, 22, "ffn_down"},
        {6144, 5120, 18, "attn_gate"}, {10240, 5120, 22, "attn_qkv"},        {12288, 5120, 3, "attn_q"},
        {17408, 5120, 40, "ffn_gate/up"},
    };
    hipDeviceProp_t prop{};
    check(hipGetDeviceProperties(&prop, 0), "props");
    int per_cu = 0;
    check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, k_pw<kFull>, 256, kChunkSub * 64 + 512 * 4),
          "occupancy");
    g_fit_pw = per_cu * prop.multiProcessorCount;
    int wall_khz = 0;
    check(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0), "wall clock");

    std::mt19937 rng(174);
    std::vector<Tensor> tensors;
    double total_bytes = 0;
    long long max_rows = 0;
    for (const auto & s : shapes) {
        for (int i = 0; i < s.n; ++i) {
            const long long nb = s.rows * (s.k / 256);
            const size_t sizes[5] = {(size_t) nb * 64, (size_t) nb * 8, (size_t) nb * 32, (size_t) nb * 4,
                                     (size_t) nb * 2};
            uint8_t * p[5];
            for (int a = 0; a < 5; ++a) {
                std::vector<uint8_t> h(sizes[a]);
                if (a == 4) {
                    auto * hh = reinterpret_cast<__half *>(h.data());
                    for (size_t j = 0; j < sizes[a] / 2; ++j) hh[j] = __float2half(0.001f * (float) (rng() % 64 + 1));
                } else {
                    for (auto & c : h) c = (uint8_t) rng();
                }
                check(hipMalloc(&p[a], sizes[a]), "malloc");
                check(hipMemcpy(p[a], h.data(), sizes[a], hipMemcpyHostToDevice), "copy");
                total_bytes += (double) sizes[a];
            }
            tensors.push_back({s.rows, s.k, s.name, {p[0], p[1], p[2], p[3], p[4]}});
            max_rows = std::max(max_rows, s.rows);
        }
    }
    std::vector<__half> xh(4 * 17408);
    for (auto & v : xh) v = __float2half((float) ((int) (rng() % 2001) - 1000) / 1000.0f);
    __half * x;
    float * y;
    unsigned long long * ts;
    check(hipMalloc(&x, xh.size() * 2), "malloc");
    check(hipMemcpy(x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice), "copy");
    check(hipMalloc(&y, 4 * max_rows * 4), "malloc");
    check(hipMalloc(&ts, (size_t) 1 << 22), "malloc");

    std::vector<const Tensor *> all;
    for (const Tensor & t : tensors) all.push_back(&t);
    std::printf("IQ3_S: %zu tensors, %.1f MiB of weights; %d CUs, persistent grid %d workgroups; wall clock %d kHz\n",
                tensors.size(), total_bytes / 1048576.0, prop.multiProcessorCount, g_fit_pw, wall_khz);
    // ~1.5 s of warm-up: the clock ramps up under load
    for (int i = 0; i < 100; ++i) (void) pass_ms<kFull>(all, 1, x, y, false, 1);

    const int reps = 9;
    for (const int nt : {1, 4}) {
        std::printf("\n== NT = %d: ms per pass (median of %d), GB/s of the weight bytes\n", nt, reps);
        std::printf("%-18s %4s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n", "shape", "n", "full", "GB/s", "no-w", "no-grid",
                    "no-x", "no-w+x", "loads", "empty", "sgpr");
        auto row = [&](const char * label, const std::vector<const Tensor *> & set) {
            double bytes = 0;
            for (const Tensor * t : set) bytes += (double) t->rows * (t->k / 256) * 110;
            const float f = pass_ms<kFull>(set, nt, x, y, false, reps);
            std::printf("%-18s %4zu %8.3f %8.1f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n", label, set.size(), f,
                        bytes / f / 1e6, pass_ms<kNoW>(set, nt, x, y, false, reps),
                        pass_ms<kNoGrid>(set, nt, x, y, false, reps), pass_ms<kNoX>(set, nt, x, y, false, reps),
                        pass_ms<kNoW | kNoX>(set, nt, x, y, false, reps),
                        pass_ms<kLoads>(set, nt, x, y, false, reps), pass_ms<kFull>(set, nt, x, y, true, reps),
                        pass_ms<kSgpr>(set, nt, x, y, false, reps));
        };
        for (const auto & s : shapes) {
            std::vector<const Tensor *> set;
            for (const Tensor & t : tensors) {
                if (t.rows == s.rows && t.k == s.k) set.push_back(&t);
            }
            row(s.name, set);
        }
        row("ALL", all);

        // per-warp timestamps, one launch per shape (cache-cold: other tensors in between)
        std::printf("\n   per launch (timestamps, us): staging = grid + x into LDS and barriers; loop; tail = "
                    "reductions + stores;\n   start spread = last warp start - first; end spread = last warp end - "
                    "first warp end; span = last end - first start\n");
        std::printf("   %-18s %6s %8s %8s %8s %8s %8s %8s %8s\n", "shape", "warps", "staging", "loop", "loop max",
                    "tail", "start sp", "end sp", "span");
        const double us = 1000.0 / wall_khz;
        for (const auto & s : shapes) {
            const Tensor * t = nullptr;
            for (const Tensor & tt : tensors) {
                if (tt.rows == s.rows && tt.k == s.k) t = &tt;
            }
            std::vector<double> st, lp, tl, t0, t3;
            for (int rep = 0; rep < 5; ++rep) {
                for (const Tensor * o : all) launch<kFull>(*o, nt, x, y, nullptr, false);  // evict
                unsigned n = 0;
                launch<kFull | kTime>(*t, nt, x, y, ts, false, &n);
                check(hipDeviceSynchronize(), "sync");
                std::vector<unsigned long long> h((size_t) n * 4);
                check(hipMemcpy(h.data(), ts, h.size() * 8, hipMemcpyDeviceToHost), "copy");
                for (unsigned i = 0; i < n; ++i) {
                    if (h[i * 4 + 3] == 0) continue;
                    st.push_back((double) (h[i * 4 + 1] - h[i * 4]) * us);
                    lp.push_back((double) (h[i * 4 + 2] - h[i * 4 + 1]) * us);
                    tl.push_back((double) (h[i * 4 + 3] - h[i * 4 + 2]) * us);
                    t0.push_back((double) h[i * 4]);
                    t3.push_back((double) h[i * 4 + 3]);
                }
                if (rep < 4) {
                    st.clear(), lp.clear(), tl.clear(), t0.clear(), t3.clear();
                    continue;
                }
                auto mean = [](const std::vector<double> & v) {
                    double a = 0;
                    for (double e : v) a += e;
                    return a / v.size();
                };
                const double f0 = *std::min_element(t0.begin(), t0.end());
                const double l0 = *std::max_element(t0.begin(), t0.end());
                const double f3 = *std::min_element(t3.begin(), t3.end());
                const double l3 = *std::max_element(t3.begin(), t3.end());
                std::printf("   %-18s %6zu %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f\n", s.name, t0.size(), mean(st),
                            mean(lp), *std::max_element(lp.begin(), lp.end()), mean(tl), (l0 - f0) * us,
                            (l3 - f3) * us, (l3 - f0) * us);
            }
        }
    }
    return 0;
}
