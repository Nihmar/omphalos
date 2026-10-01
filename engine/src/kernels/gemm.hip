// f16 GEMM on the WMMA units (#132).
//
// 128 x 128 block tile (W rows x tokens), 8 waves as 2 x 4, 64 x 32 per wave
// (4 x 2 WMMA 16x16x16 tiles: 8 independent accumulators, what the units need
// to reach their ~100 TFLOPS), BK = 64. One padded LDS buffer; the next k-tile
// is prefetched into registers while the current one is multiplied. Tokens are
// the fastest grid dimension, so the blocks that share a W tile run together
// and W is read from DRAM once. 60-64 TFLOPS on this model's shapes, hipBLASLt
// ~49 (#132).
//
// gfx12 WMMA operand layout (wave32, verified): A (16 x 16, M x K) lane l holds
// row l % 16, k = (l / 16) * 8 .. + 7; B (K x N) lane l holds column l % 16,
// the same k range; D lane l holds column l % 16, rows (l / 16) * 8 .. + 7. Both
// W and x are k-contiguous, so every operand is one 16-byte load.
#include "kernels/gemm.hh"

#include "kernels/quant_decode.hh"

namespace omph::kernels {
namespace {

typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

constexpr int kBM = 128;  // W rows per block
constexpr int kBN = 128;  // tokens per block
constexpr int kBK = 64;
constexpr int kWM = 2;    // waves along W rows
constexpr int kWN = 4;    // waves along tokens
constexpr int kThreads = 32 * kWM * kWN;
constexpr int kLDK = kBK + 8;  // padded LDS rows (144 B): conflict-free 16-lane reads
constexpr int kTM = kBM / kWM / 16;
constexpr int kTN = kBN / kWN / 16;
constexpr int kChunks = kBK / 8;  // 16-byte chunks per tile row
constexpr int kACh = kBM * kChunks / kThreads;
constexpr int kBCh = kBN * kChunks / kThreads;

// The staging arrays must stay in registers: an aggregate copy or an
// if-zeroed element sent them to scratch, which serialized every global load.
template <int N>
__device__ __forceinline__ void load_tile(uint4 (&r)[N], const _Float16 * const (&p)[N],
                                          const unsigned (&mask)[N], const int k0) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        const uint4 v = *reinterpret_cast<const uint4 *>(p[i] + k0);
        r[i] = make_uint4(v.x & mask[i], v.y & mask[i], v.z & mask[i], v.w & mask[i]);
    }
}

template <int N>
__device__ __forceinline__ void store_tile(_Float16 * lds, const uint4 (&r)[N], const int (&o)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        *reinterpret_cast<uint4 *>(lds + o[i]) = r[i];
    }
}

__global__ __launch_bounds__(kThreads) void gemm_f16_kernel(
    const _Float16 * __restrict__ w, const _Float16 * __restrict__ x, float * __restrict__ y,
    const int n_out, const int k, const int tokens, const long long ldy) {
    __shared__ _Float16 as[kBM * kLDK];
    __shared__ _Float16 bs[kBN * kLDK];
    const int tid = (int) threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int wm = wave % kWM;
    const int wn = wave / kWM;
    const int m0 = (int) blockIdx.y * kBM;
    const int n0 = (int) blockIdx.x * kBN;

    // each thread stages 16-byte chunks; rows past the end read row 0, masked to 0
    uint4 ra[kACh];
    uint4 rb[kBCh];
    const _Float16 * wp[kACh];
    const _Float16 * xp[kBCh];
    unsigned am[kACh];
    unsigned bm[kBCh];
    int ao[kACh];
    int bo[kBCh];
#pragma unroll
    for (int i = 0; i < kACh; ++i) {
        const int c = tid + i * kThreads;
        const int r = c / kChunks;
        const int q = c % kChunks;
        const bool in = m0 + r < n_out;
        am[i] = in ? ~0u : 0u;
        wp[i] = w + (long long) (in ? m0 + r : 0) * k + q * 8;
        ao[i] = r * kLDK + q * 8;
    }
#pragma unroll
    for (int i = 0; i < kBCh; ++i) {
        const int c = tid + i * kThreads;
        const int r = c / kChunks;
        const int q = c % kChunks;
        const bool in = n0 + r < tokens;
        bm[i] = in ? ~0u : 0u;
        xp[i] = x + (long long) (in ? n0 + r : 0) * k + q * 8;
        bo[i] = r * kLDK + q * 8;
    }
    float8 acc[kTM][kTN];
#pragma unroll
    for (int i = 0; i < kTM; ++i) {
#pragma unroll
        for (int j = 0; j < kTN; ++j) {
            acc[i][j] = float8{};
        }
    }
    load_tile(ra, wp, am, 0);
    load_tile(rb, xp, bm, 0);
    store_tile(as, ra, ao);
    store_tile(bs, rb, bo);
    __syncthreads();
    const int nk = k / kBK;
    for (int kt = 0; kt < nk; ++kt) {
        if (kt + 1 < nk) {
            load_tile(ra, wp, am, (kt + 1) * kBK);
            load_tile(rb, xp, bm, (kt + 1) * kBK);
        }
        // not unrolled: unrolling hoists every k-step's LDS operands and spills
#pragma unroll 1
        for (int kk = 0; kk < kBK; kk += 16) {
            half8 a[kTM];
            half8 b[kTN];
#pragma unroll
            for (int i = 0; i < kTM; ++i) {
                a[i] = *reinterpret_cast<const half8 *>(
                    &as[(wm * kTM * 16 + i * 16 + (lane & 15)) * kLDK + kk + (lane >> 4) * 8]);
            }
#pragma unroll
            for (int j = 0; j < kTN; ++j) {
                b[j] = *reinterpret_cast<const half8 *>(
                    &bs[(wn * kTN * 16 + j * 16 + (lane & 15)) * kLDK + kk + (lane >> 4) * 8]);
            }
#pragma unroll
            for (int i = 0; i < kTM; ++i) {
#pragma unroll
                for (int j = 0; j < kTN; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], b[j],
                                                                                 acc[i][j]);
                }
            }
        }
        __syncthreads();  // the tile is consumed
        if (kt + 1 < nk) {
            store_tile(as, ra, ao);
            store_tile(bs, rb, bo);
        }
        __syncthreads();
    }
    // D: lane -> token (lane & 15), W rows (lane >> 4) * 8 .. + 7: 32 contiguous bytes of y
#pragma unroll
    for (int i = 0; i < kTM; ++i) {
        const int r = m0 + wm * kTM * 16 + i * 16 + (lane >> 4) * 8;
#pragma unroll
        for (int j = 0; j < kTN; ++j) {
            const int t = n0 + wn * kTN * 16 + j * 16 + (lane & 15);
            if (t < tokens && r < n_out) {
                float4 * dst = reinterpret_cast<float4 *>(y + t * ldy + r);
                dst[0] = make_float4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
                dst[1] = make_float4(acc[i][j][4], acc[i][j][5], acc[i][j][6], acc[i][j][7]);
            }
        }
    }
}

// --- fused dequant + GEMM (#141) ---------------------------------------------
//
// gemm_f16_kernel with the W tile decoded from the repacked layout instead of
// read as f16: a k-step's tile is 128 rows x 64 weights, i.e. 256 sub-blocks
// of 32, one per thread (row tid / 2, sub-block 2 kt + tid % 2). Each thread
// loads its next sub-block's words while the WMMAs of the current tile run,
// then decodes them into the tile's LDS rows. The f16 values and the LDS
// layout are the f16 path's, and so is the WMMA order: the output is
// bit-identical to dequantizing to f16 and calling gemm_f16. No f16 copy of
// W is written anywhere (the round trip through the cache cost ~23 % of a
// 512-token prefill, #135 / #140).
template <class Dec, int BN>
__global__ __launch_bounds__(kThreads) void gemm_q_kernel(
    const Dec dec, const _Float16 * __restrict__ x, float * __restrict__ y, const int n_out,
    const int k, const int tokens, const long long ldy) {
    static_assert(kBM * kBK / 32 == kThreads, "one sub-block per thread per k-step");
    // (the codebooks stay in global memory: an LDS copy measured slower, #141)
    __shared__ _Float16 as[kBM * kLDK];
    constexpr int TN = BN / kWN / 16;
    constexpr int BCh = BN * kChunks / kThreads;
    __shared__ _Float16 bs[BN * kLDK];
    const int tid = (int) threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int wm = wave % kWM;
    const int wn = wave / kWM;
    const int m0 = (int) blockIdx.y * kBM;
    const int n0 = (int) blockIdx.x * BN;

    // this thread's W row and sub-block slot
    const int ar = tid >> 1;
    const int au = tid & 1;
    const bool a_in = m0 + ar < n_out;
    const long long a_row = a_in ? m0 + ar : 0;
    _Float16 * a_dst = as + ar * kLDK + au * 32;
    typename Dec::Raw raw{};
    const auto put_a = [&](const int kt) {
        uint4 o[4];
        if (a_in) {
            float v[32];
            dec.decode(raw, 2LL * kt + au, v);
            qd::pack32(v, o);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                o[i] = make_uint4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            *reinterpret_cast<uint4 *>(a_dst + i * 8) = o[i];
        }
    };

    // x as in gemm_f16_kernel
    uint4 rb[BCh];
    const _Float16 * xp[BCh];
    unsigned bm[BCh];
    int bo[BCh];
#pragma unroll
    for (int i = 0; i < BCh; ++i) {
        const int c = tid + i * kThreads;
        const int r = c / kChunks;
        const int q = c % kChunks;
        const bool in = n0 + r < tokens;
        bm[i] = in ? ~0u : 0u;
        xp[i] = x + (long long) (in ? n0 + r : 0) * k + q * 8;
        bo[i] = r * kLDK + q * 8;
    }
    float8 acc[kTM][TN];
#pragma unroll
    for (int i = 0; i < kTM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = float8{};
        }
    }
    if (a_in) {
        raw = dec.load(a_row, au);
    }
    load_tile(rb, xp, bm, 0);
    put_a(0);
    store_tile(bs, rb, bo);
    __syncthreads();
    const int nk = k / kBK;
    for (int kt = 0; kt < nk; ++kt) {
        if (kt + 1 < nk) {
            if (a_in) {
                raw = dec.load(a_row, 2LL * (kt + 1) + au);
            }
            load_tile(rb, xp, bm, (kt + 1) * kBK);
        }
#pragma unroll 1
        for (int kk = 0; kk < kBK; kk += 16) {
            half8 a[kTM];
            half8 b[TN];
#pragma unroll
            for (int i = 0; i < kTM; ++i) {
                a[i] = *reinterpret_cast<const half8 *>(
                    &as[(wm * kTM * 16 + i * 16 + (lane & 15)) * kLDK + kk + (lane >> 4) * 8]);
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                b[j] = *reinterpret_cast<const half8 *>(
                    &bs[(wn * TN * 16 + j * 16 + (lane & 15)) * kLDK + kk + (lane >> 4) * 8]);
            }
#pragma unroll
            for (int i = 0; i < kTM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], b[j],
                                                                                 acc[i][j]);
                }
            }
        }
        __syncthreads();  // the tile is consumed
        if (kt + 1 < nk) {
            put_a(kt + 1);
            store_tile(bs, rb, bo);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < kTM; ++i) {
        const int r = m0 + wm * kTM * 16 + i * 16 + (lane >> 4) * 8;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int t = n0 + wn * TN * 16 + j * 16 + (lane & 15);
            if (t < tokens && r < n_out) {
                float4 * dst = reinterpret_cast<float4 *>(y + t * ldy + r);
                dst[0] = make_float4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
                dst[1] = make_float4(acc[i][j][4], acc[i][j][5], acc[i][j][6], acc[i][j][7]);
            }
        }
    }
}

} // namespace

bool gemm_f16(const void * w, const void * x, float * y, const int64_t n_out, const int64_t k,
              const int64_t tokens, const int64_t ldy, hipStream_t stream) {
    const auto aligned = [](const void * p) { return ((uintptr_t) p & 15) == 0; };
    if (w == nullptr || x == nullptr || y == nullptr || n_out <= 0 || tokens <= 0 || k <= 0 ||
        k % kBK != 0 || n_out % 16 != 0 || ldy < n_out || ldy % 4 != 0 || !aligned(w) ||
        !aligned(x) || !aligned(y) || n_out > INT32_MAX || tokens > INT32_MAX || k > INT32_MAX) {
        return false;
    }
    const dim3 grid((unsigned) ((tokens + kBN - 1) / kBN), (unsigned) ((n_out + kBM - 1) / kBM));
    gemm_f16_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const _Float16 *>(w),
                                                    static_cast<const _Float16 *>(x), y, (int) n_out,
                                                    (int) k, (int) tokens, (long long) ldy);
    return hipGetLastError() == hipSuccess;
}

bool gemm_q(const uint32_t type, const void * packed, const void * x, float * y,
            const int64_t n_out, const int64_t k, const int64_t tokens, const int64_t ldy,
            hipStream_t stream) {
    const auto aligned = [](const void * p) { return ((uintptr_t) p & 15) == 0; };
    if (packed == nullptr || x == nullptr || y == nullptr || n_out <= 0 || tokens <= 0 ||
        k <= 0 || k % 256 != 0 || n_out % 16 != 0 || ldy < n_out || ldy % 4 != 0 ||
        !aligned(x) || !aligned(y) || n_out > INT32_MAX || tokens > INT32_MAX ||
        k > INT32_MAX) {
        return false;
    }
    const auto * base = static_cast<const uint8_t *>(packed);
    const int64_t n_blocks = n_out * (k / 256);
    const long long blocks = k / 256;
    // 256 tokens per block halves the decode work per FLOP (512 tokens: 625 ->
    // 577 ms of prefill); up to 256 tokens the 128-wide tile keeps more blocks
    // busy and wins (T = 256: 327 vs 335 ms, #141).
    const auto launch = [&](const auto dec) {
        const auto * xh = static_cast<const _Float16 *>(x);
        if (tokens > 256) {
            const dim3 grid((unsigned) ((tokens + 255) / 256), (unsigned) ((n_out + kBM - 1) / kBM));
            gemm_q_kernel<decltype(dec), 256><<<grid, kThreads, 0, stream>>>(
                dec, xh, y, (int) n_out, (int) k, (int) tokens, (long long) ldy);
        } else {
            const dim3 grid((unsigned) ((tokens + 127) / 128), (unsigned) ((n_out + kBM - 1) / kBM));
            gemm_q_kernel<decltype(dec), 128><<<grid, kThreads, 0, stream>>>(
                dec, xh, y, (int) n_out, (int) k, (int) tokens, (long long) ldy);
        }
    };
    switch (type) {
        case 21: launch(qd::DecIq3s::make(base, n_blocks, blocks)); break;
        case 18: launch(qd::DecIq3xxs::make(base, n_blocks, blocks)); break;
        case 16: launch(qd::DecIq2xxs::make(base, n_blocks, blocks)); break;
        case 17: launch(qd::DecIq2xs::make(base, n_blocks, blocks)); break;
        case 22: launch(qd::DecIq2s::make(base, n_blocks, blocks)); break;
        case 23: launch(qd::DecIq4xs::make(base, n_blocks, blocks)); break;
        case 12: launch(qd::DecQ4k::make(base, n_blocks, blocks)); break;
        case 10: launch(qd::DecQ2k::make(base, n_blocks, blocks)); break;
        case 14: launch(qd::DecQ6k::make(base, n_blocks, blocks)); break;
        case 29: launch(qd::DecIq1m::make(base, n_blocks, blocks)); break;
        default: return false;
    }
    return hipGetLastError() == hipSuccess;
}

} // namespace omph::kernels
