// #420 step 1: the prize of fusing the decode step's launches, measured on the
// memory pipeline alone.
//
// The decode step issues ~256 GEMV launches over ~11 GB (the 64 layers' grouped
// projections plus the 682 MiB head) plus ~450 smaller kernels. Each launch pays
// the memory pipeline's fill and drain (measured in stream_size.hip: ~8 us at
// 13-36 MiB, #203). This bench answers two questions on the real sizes:
//
//   A) a *chain* of one launch per segment: the total (the bytes plus every
//      launch's fill/drain and gap),
//   B) the same bytes as one launch with a device-side work queue over
//      128 KiB blocks (the fully fused ideal: the bytes and nothing else),
//   C) the chain cut into 2 / 8 / 32 launches (partial fusion granularities).
//
// A - B is the upper bound of what a persistent/fused step can win on the
// memory side; the GEMVs' own dequant and dot work is not part of it.
//   hipcc -O3 --offload-arch=gfx1200 -o /var/tmp/chain_size bench/launch/chain_size.hip && /var/tmp/chain_size
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace {

__device__ unsigned g_task;

__global__ __launch_bounds__(256) void rd(const uint4 * __restrict__ p, const long long n, uint4 * out) {
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc.x ^= v.x;
        acc.y ^= v.y;
        acc.z ^= v.z;
        acc.w ^= v.w;
    }
    if (acc.x == 0x12345678u && acc.y == 7u) {
        out[0] = acc;
    }
}

// One launch per work-queue group: the workgroups take 128 KiB blocks off a
// global counter, so no workgroup idles at a segment's tail and the memory
// pipeline never drains between segments.
__global__ __launch_bounds__(256) void rd_queue(const uint4 * __restrict__ pool, const long long * __restrict__ block_of,
                                                const long long * __restrict__ off_of, const long long nblocks,
                                                const long long blocks_per_wg, uint4 * out) {
    uint4 acc = make_uint4(0, 0, 0, 0);
    const long long base = (long long) blockIdx.x * blocks_per_wg;
    for (long long t = 0; t < blocks_per_wg; ++t) {
        const long long b = base + t;
        if (b >= nblocks) {
            break;
        }
        const long long o = off_of[b];
        const long long n = block_of[b];
        const uint4 * p = pool + o;
        for (long long i = threadIdx.x; i < n; i += 256) {
            const uint4 v = p[i];
            acc.x ^= v.x;
            acc.y ^= v.y;
            acc.z ^= v.z;
            acc.w ^= v.w;
        }
    }
    if (acc.x == 0x12345678u && acc.y == 7u) {
        out[0] = acc;
    }
}

// #456: the decode step's small kernels (delta_step, norms, attention prep /
// merge: ~165 launches, ~2.5 ms per step) leave the DRAM idle. A light
// side-stream kernel could read the *next* GEMV's head into the 32 MB MALL
// (Infinity Cache) during them, so that GEMV starts on hits instead of a cold
// ramp. The dummy below is one such gap (a dependent FMA chain, no memory), and
// rd_prefetch reads a head of `bytes` with cacheable loads (a non-temporal load
// would bypass the MALL and prefetch nothing).
__global__ __launch_bounds__(256) void lat_dummy(float * out, const int iters) {
    float x = (float) threadIdx.x * 1e-6f + (float) blockIdx.x * 1e-5f;
    for (int i = 0; i < iters; ++i) {
        x = fmaf(x, 1.000001f, 1e-7f);
    }
    if (x == 12345.678f) {
        out[0] = x;
    }
}

__global__ __launch_bounds__(256) void rd_prefetch(const uint4 * __restrict__ p, const long long n, uint4 * out) {
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc.x ^= v.x;
        acc.y ^= v.y;
        acc.z ^= v.z;
        acc.w ^= v.w;
    }
    if (acc.x == 0x12345678u && acc.y == 7u) {
        out[0] = acc;
    }
}

} // namespace

int main() {
    // The step's segment sizes: 64 layers x (qkv+gate 35 MiB, ssm_out 14, gate+up
    // 72, down 39) + the head 682 MiB ~ 11.0 GiB in 257 launches, against the
    // model's 11.28 GiB and ~256 GEMV launches (the profile of #203).
    std::vector<double> seg_mib;
    for (int l = 0; l < 64; ++l) {
        seg_mib.push_back(35.0);
        seg_mib.push_back(14.0);
        seg_mib.push_back(72.0);
        seg_mib.push_back(39.0);
    }
    seg_mib.push_back(682.0);
    const size_t block_bytes = 128 * 1024;
    size_t total = 0;
    for (const double m : seg_mib) {
        total += (size_t) (m * 1048576) / 16 * 16;
    }
    std::printf("chain: %zu segments, %.1f GiB\n", seg_mib.size(), total / 1073741824.0);

    uint4 * pool = nullptr;
    uint4 * out = nullptr;
    const size_t pool_bytes = (total + 1073741824) & ~(size_t) 15;  // the chain + a gibibyte of slack
    if (hipMalloc(&pool, pool_bytes) != hipSuccess || hipMalloc(&out, 64) != hipSuccess) {
        std::printf("malloc failed (needs %.1f GiB)\n", pool_bytes / 1073741824.0);
        return 1;
    }
    (void) hipMemset(pool, 1, pool_bytes);

    // the segment table (device pointers = offsets into the pool) and the block
    // table for the queue variant
    std::vector<long long> seg_off, seg_n;
    size_t at = 0;
    for (const double m : seg_mib) {
        const size_t bytes = (size_t) (m * 1048576) / 16 * 16;
        seg_off.push_back((long long) (at / 16));
        seg_n.push_back((long long) (bytes / 16));
        at += bytes;
    }
    std::vector<long long> blk_n, blk_off;
    for (size_t s = 0; s < seg_mib.size(); ++s) {
        const long long blocks = std::max<long long>(1, (seg_n[s] * 16) / (long long) block_bytes);
        for (long long b = 0; b < blocks; ++b) {
            blk_off.push_back(seg_off[s] + b * (long long) (block_bytes / 16));
            blk_n.push_back(std::min<long long>((long long) (block_bytes / 16),
                                                seg_n[s] - b * (long long) (block_bytes / 16)));
        }
    }
    long long * d_blk_n = nullptr;
    long long * d_blk_off = nullptr;
    (void) hipMalloc(&d_blk_n, blk_n.size() * 8);
    (void) hipMalloc(&d_blk_off, blk_off.size() * 8);
    (void) hipMemcpy(d_blk_n, blk_n.data(), blk_n.size() * 8, hipMemcpyHostToDevice);
    (void) hipMemcpy(d_blk_off, blk_off.data(), blk_off.size() * 8, hipMemcpyHostToDevice);
    std::printf("queue: %zu blocks of 128 KiB\n", blk_n.size());

    const unsigned grid = 16 * 16;
    hipEvent_t a, b;
    (void) hipEventCreate(&a);
    (void) hipEventCreate(&b);
    const auto warm = [&] {
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 2.0) {
            for (size_t s = 0; s < seg_mib.size(); ++s) {
                rd<<<grid, 256>>>(pool + seg_off[s], seg_n[s], out);
            }
            (void) hipDeviceSynchronize();
        }
    };
    warm();

    const auto time_ms = [&](const std::function<void()> & f) {
        std::vector<float> v;
        for (int rep = 0; rep < 5; ++rep) {
            (void) hipEventRecord(a);
            f();
            (void) hipEventRecord(b);
            (void) hipEventSynchronize(b);
            float ms = 0;
            (void) hipEventElapsedTime(&ms, a, b);
            v.push_back(ms);
        }
        std::sort(v.begin(), v.end());
        return (double) v[v.size() / 2];
    };

    // A: one launch per segment
    const double ms_a = time_ms([&] {
        for (size_t s = 0; s < seg_mib.size(); ++s) {
            rd<<<grid, 256>>>(pool + seg_off[s], seg_n[s], out);
        }
    });
    // C: the same chain in K launches (a slice of the segment list each)
    const auto chain_k = [&](const int K) {
        return time_ms([&] {
            const size_t per = (seg_mib.size() + (size_t) K - 1) / (size_t) K;
            for (int k = 0; k < K; ++k) {
                const size_t s0 = (size_t) k * per;
                const size_t s1 = std::min(s0 + per, seg_mib.size());
                for (size_t s = s0; s < s1; ++s) {
                    rd<<<grid, 256>>>(pool + seg_off[s], seg_n[s], out);
                }
            }
        });
    };
    // B: one queue launch for the whole chain
    const long long blocks_per_wg = (long long) ((blk_n.size() + grid - 1) / grid) + 1;
    const double ms_b = time_ms([&] {
        rd_queue<<<grid, 256>>>(pool, d_blk_n, d_blk_off, (long long) blk_n.size(), blocks_per_wg, out);
    });
    const double gb = total / 1073741824.0;
    const auto row = [&](const char * what, const double ms) {
        std::printf("%-22s %8.2f ms  %6.1f GB/s  (%.1f us per segment)\n", what, ms, gb / (ms * 1e-3),
                    ms * 1000.0 / (double) seg_mib.size());
    };
    row("A: launch per segment", ms_a);
    row("B: one queued launch", ms_b);
    row("C: 2 launches", chain_k(2));
    row("C: 8 launches", chain_k(8));
    row("C: 32 launches", chain_k(32));
    std::printf("prize A - B: %.2f ms (%.0f%% of A)\n", ms_a - ms_b, 100.0 * (ms_a - ms_b) / ms_a);

    // ---- #456: a latency-bound gap between segments, with and without a
    // side-stream prefetch of the next segment's head.
    hipStream_t side = nullptr;
    (void) hipStreamCreateWithFlags(&side, hipStreamNonBlocking);
    float * fout = nullptr;
    (void) hipMalloc(&fout, 64);
    // ~15 us of dependent FMA (the small kernels' measured cost, #203)
    const int iters = 350;  // calibrated for ~15 us: the small kernels' measured cost (#203)
    hipEvent_t ev_pre = nullptr;
    (void) hipEventCreateWithFlags(&ev_pre, hipEventDisableTiming);
    const std::vector<size_t> prefetch_mib = {0, 2, 4, 8, 16, 24};
    // a gap without the prefetch: the chain with the dummy everywhere (the head
    // row's cost includes the dummy; the deltas between rows are the effect)
    std::vector<double> base(seg_mib.size(), 0.0);
    const auto chain_gap = [&](const int p_mib) {
        const size_t bytes = (size_t) p_mib * 1048576;
        return time_ms([&] {
            for (size_t s = 0; s + 1 < seg_mib.size(); ++s) {
                rd<<<grid, 256>>>(pool + seg_off[s], seg_n[s], out);
                if (bytes > 0) {
                    const long long nn = (long long) std::min<size_t>(bytes, (size_t) seg_n[s + 1] * 16) / 16;
                    rd_prefetch<<<grid, 256, 0, side>>>(pool + seg_off[s + 1], nn, out);
                    (void) hipEventRecord(ev_pre, side);
                }
                lat_dummy<<<grid, 256>>>(fout, iters);
                if (bytes > 0) {
                    // a device-side wait, not a host sync (256 host syncs cost
                    // more than the whole chain)
                    (void) hipStreamWaitEvent(nullptr, ev_pre, 0);
                }
            }
        });
    };
    std::printf("\n#456: the chain with a %.0f us latency gap per segment (FMA dummy), prefetch of the next head:\n",
                (double) iters * 0.0002);
    for (const size_t mib : prefetch_mib) {
        const double ms = chain_gap((int) mib);
        std::printf("%-22s %8.2f ms  %6.1f GB/s  (%.1f us per segment)\n",
                    mib == 0 ? "no prefetch" : (std::string("prefetch ") + std::to_string(mib) + " MiB").c_str(),
                    ms, gb / (ms * 1e-3), ms * 1000.0 / (double) seg_mib.size());
    }
    // the dummy alone, for the gap's own cost
    const double ms_dummy = time_ms([&] {
        for (size_t s = 0; s + 1 < seg_mib.size(); ++s) {
            lat_dummy<<<grid, 256>>>(fout, iters);
        }
    });
    std::printf("the gap alone: %.2f ms for %zu gaps (%.1f us each)\n", ms_dummy, seg_mib.size() - 1,
                ms_dummy * 1000.0 / (double) (seg_mib.size() - 1));
    (void) hipStreamDestroy(side);
    return 0;
}
