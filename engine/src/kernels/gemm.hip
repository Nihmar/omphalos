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
// The output of 8 rows of one token (acc, rows r .. r + 7 at y + off): f32 by
// default, or with SWI (#221) the FFN's f16(silu(gate) * up) where the GEMM
// computes up: the gate rows read at the same offset, swiglu_f16's expression,
// so the result is bit-identical to writing up and running swiglu_f16.
struct Epi {
    const float * gate;
    __half * out;
};

template <bool SWI>
__device__ inline void store8(const float8 & a, float * __restrict__ y, const Epi & e, const long long off) {
    if constexpr (SWI) {
        const float4 g0 = reinterpret_cast<const float4 *>(e.gate + off)[0];
        const float4 g1 = reinterpret_cast<const float4 *>(e.gate + off)[1];
        const float g[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
        __half h[8];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            h[v] = __float2half((g[v] / (1.0f + expf(-g[v]))) * a[v]);
        }
        *reinterpret_cast<uint4 *>(e.out + off) = *reinterpret_cast<const uint4 *>(h);
    } else {
        float4 * dst = reinterpret_cast<float4 *>(y + off);
        dst[0] = make_float4(a[0], a[1], a[2], a[3]);
        dst[1] = make_float4(a[4], a[5], a[6], a[7]);
    }
}

template <class Dec, int BN, bool SWI>
__global__ __launch_bounds__(kThreads) void gemm_q_kernel(
    const Dec dec, const _Float16 * __restrict__ x, float * __restrict__ y, const int n_out,
    const int k, const int tokens, const long long ldy, const Epi epi) {
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
                store8<SWI>(acc[i][j], y, epi, t * ldy + r);
            }
        }
    }
}


// WMMA tiles (#178, #219): a workgroup = 8 waves x 16 rows (128 rows) x 256
// tokens. Each wave decodes its rows' weights straight into WMMA A operands,
// with the tile GEMVs' decoders (quant_decode.hh), and reuses them for 16
// token tiles; only x goes through LDS, one 32-weight sub-block at a time,
// double-buffered. IQ3_S: 52.7 vs 50.3 TFLOPS of the row-wise decode at 512
// tokens (bench/gemv/iq3s_wil.hip).
constexpr int kGtTok = 256;   // tokens per workgroup
constexpr int kGtXLd = 40;    // halves per staged token row (32 + 8 pad)
template <class TD, bool SWI>
__global__ __launch_bounds__(256) void gemm_tiles_kernel(const uint8_t * __restrict__ wt,
                                                        const _Float16 * __restrict__ x, float * __restrict__ y,
                                                        const long long blocks, const int tokens,
                                                        const long long ldy, const Epi epi) {
    __shared__ __attribute__((aligned(16))) uint8_t tab[TD::kLds];
    __shared__ _Float16 xs[2][kGtTok * kGtXLd];
    const int tid = (int) threadIdx.x;
    TD::init(tab);
    const int lane = tid & 31;
    const int wv = tid >> 5;
    const int h = lane >> 4;
    const long long tile = (long long) blockIdx.y * 8 + wv;
    const long long n0 = (long long) blockIdx.x * kGtTok;
    const long long k = blocks * 256;
    const long long tx = n0 + tid < tokens ? n0 + tid : 0;
    const uint4 * xrow = reinterpret_cast<const uint4 *>(x + tx * k);
    const auto stage = [&](const long long s, _Float16 * buf) {
        const uint4 * src = xrow + s * 4;
        uint4 * dst = reinterpret_cast<uint4 *>(buf + tid * kGtXLd);
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
    };
    float8 acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        acc[j] = float8{};
    }
    stage(0, xs[0]);
    __syncthreads();
    int buf = 0;
    for (long long b = 0; b < blocks; ++b) {
        const typename TD::Blk kb = TD::load(wt + (tile * blocks + b) * TD::kBytes, lane);
#pragma unroll 1
        for (int sb = 0; sb < 8; ++sb) {
            const long long s = b * 8 + sb;
            if (s + 1 < blocks * 8) {
                stage(s + 1, xs[buf ^ 1]);  // the next sub-block's x
            }
            const half8 a[2] = {TD::frag(tab, kb, sb, 0, h), TD::frag(tab, kb, sb, 1, h)};
            const _Float16 * cur = xs[buf];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
#pragma unroll
                for (int gl = 0; gl < 2; ++gl) {
                    const half8 bv = *reinterpret_cast<const half8 *>(
                        cur + (j * 16 + (lane & 15)) * kGtXLd + h * 16 + gl * 8);
                    acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[gl], bv, acc[j]);
                }
            }
            __syncthreads();  // this x buffer is consumed, the next one written
            buf ^= 1;
        }
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const long long t = n0 + j * 16 + (lane & 15);
        if (t < tokens) {
            store8<SWI>(acc[j], y, epi, t * ldy + tile * 16 + h * 8);
        }
    }
}

// WMMA tiles, 2 x 8 per wave (#326): a workgroup = 8 waves x 32 rows (256 rows)
// x 128 tokens. Each wave owns two 16-row tiles (all 8 waves one distinct
// pair, so the decode is not duplicated) and all 8 token tiles, i.e. 16
// accumulators like the kernel above; a B operand read from LDS then feeds two
// WMMAs instead of one (0.5 reads per WMMA against 1.0). The smaller token
// tile also halves the x buffer (20 KB against 40 KB), which takes occupancy
// from 1 to 2 workgroups per WGP. Measured 55.7 -> 60.0 TFLOPS on IQ3_S
// ffn_up at 512 tokens, +2 to +8 % on the other tile types (Q4_K keeps the
// kernel above: its decode is cheap and the two weight blocks cost it more
// than the halved B reads gain); every output bit-identical
// (bench/gemm/tile2.hip).
constexpr int kG2Tok = 128;   // tokens per workgroup
constexpr int kG2XLd = 40;    // halves per staged token row (32 + 8 pad)
template <class TD, bool SWI>
__global__ __launch_bounds__(256) void gemm_tiles2_kernel(const uint8_t * __restrict__ wt,
                                                         const _Float16 * __restrict__ x, float * __restrict__ y,
                                                         const long long blocks, const int tokens,
                                                         const long long ldy, const Epi epi) {
    __shared__ __attribute__((aligned(16))) uint8_t tab[TD::kLds];
    __shared__ _Float16 xs[2][kG2Tok * kG2XLd];
    const int tid = (int) threadIdx.x;
    TD::init(tab);
    const int lane = tid & 31;
    const int wv = tid >> 5;
    const int h = lane >> 4;
    const long long tile = (long long) blockIdx.y * 16 + wv * 2;
    const long long n0 = (long long) blockIdx.x * kG2Tok;
    const long long k = blocks * 256;
    const long long tx = n0 + tid < tokens ? n0 + tid : 0;
    const uint4 * xrow = reinterpret_cast<const uint4 *>(x + tx * k);
    const auto stage = [&](const long long s, _Float16 * buf) {
        if (tid >= kG2Tok) {
            return;
        }
        const uint4 * src = xrow + s * 4;
        uint4 * dst = reinterpret_cast<uint4 *>(buf + tid * kG2XLd);
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
    };
    float8 acc[2][8];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[i][j] = float8{};
        }
    }
    stage(0, xs[0]);
    __syncthreads();
    int buf = 0;
    for (long long b = 0; b < blocks; ++b) {
        const typename TD::Blk kb[2] = {TD::load(wt + (tile * blocks + b) * TD::kBytes, lane),
                                        TD::load(wt + ((tile + 1) * blocks + b) * TD::kBytes, lane)};
#pragma unroll 1
        for (int sb = 0; sb < 8; ++sb) {
            const half8 a[2][2] = {{TD::frag(tab, kb[0], sb, 0, h), TD::frag(tab, kb[0], sb, 1, h)},
                                   {TD::frag(tab, kb[1], sb, 0, h), TD::frag(tab, kb[1], sb, 1, h)}};
#pragma unroll
            for (int gl = 0; gl < 2; ++gl) {
                if (gl == 0) {
                    const long long s = b * 8 + sb;
                    if (s + 1 < blocks * 8) {
                        stage(s + 1, xs[buf ^ 1]);  // the next sub-block's x
                    }
                }
                const _Float16 * cur = xs[buf];
                half8 bv[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    bv[j] = *reinterpret_cast<const half8 *>(
                        cur + (j * 16 + (lane & 15)) * kG2XLd + h * 16 + gl * 8);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i][gl], bv[j], acc[i][j]);
                    }
                }
                if (gl == 1) {
                    __syncthreads();  // this x buffer is consumed, the next one written
                    buf ^= 1;
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const long long t = n0 + j * 16 + (lane & 15);
            if (t < tokens) {
                store8<SWI>(acc[i][j], y, epi, t * ldy + (tile + i) * 16 + h * 8);
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

namespace {

template <bool SWI>
bool gemm_q_impl(const uint32_t type, const void * packed, const void * x, float * y, const int64_t n_out,
                 const int64_t k, const int64_t tokens, const int64_t ldy, const Epi epi, hipStream_t stream) {
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
            gemm_q_kernel<decltype(dec), 256, SWI><<<grid, kThreads, 0, stream>>>(
                dec, xh, y, (int) n_out, (int) k, (int) tokens, (long long) ldy, epi);
        } else {
            const dim3 grid((unsigned) ((tokens + 127) / 128), (unsigned) ((n_out + kBM - 1) / kBM));
            gemm_q_kernel<decltype(dec), 128, SWI><<<grid, kThreads, 0, stream>>>(
                dec, xh, y, (int) n_out, (int) k, (int) tokens, (long long) ldy, epi);
        }
    };
    const auto tiles = [&](const auto td) {
        const dim3 grid((unsigned) ((tokens + kGtTok - 1) / kGtTok), (unsigned) (n_out / 128));
        gemm_tiles_kernel<decltype(td), SWI><<<grid, 256, 0, stream>>>(
            base, static_cast<const _Float16 *>(x), y, blocks, (int) tokens, (long long) ldy, epi);
    };
    // 2 x 8 rows/tokens per wave (#326); needs a whole 256-row workgroup
    const auto tiles2 = [&](const auto td) {
        const dim3 grid((unsigned) ((tokens + kG2Tok - 1) / kG2Tok), (unsigned) (n_out / 256));
        gemm_tiles2_kernel<decltype(td), SWI><<<grid, 256, 0, stream>>>(
            base, static_cast<const _Float16 *>(x), y, blocks, (int) tokens, (long long) ldy, epi);
    };
    switch (type) {
        case 21:
            if (n_out % 256 == 0) {  // the 2 x 8 tile GEMM (#326)
                tiles2(qd::TileIq3s{});
            } else if (n_out % 128 == 0) {  // the tile GEMM (#178)
                tiles(qd::TileIq3s{});
            } else {
                launch(qd::DecIq3sTile::make(base, n_blocks, blocks));
            }
            break;
        case 18:
            if (n_out % 256 == 0) {  // #326
                tiles2(qd::TileIq3xxs{});
            } else if (n_out % 128 == 0) {  // #219
                tiles(qd::TileIq3xxs{});
            } else {
                launch(qd::DecIq3xxsTile::make(base, n_blocks, blocks));
            }
            break;
        case 16:  // #253
            if (n_out % 256 == 0) tiles2(qd::TileIq2xxs{});
            else if (n_out % 128 == 0) tiles(qd::TileIq2xxs{});
            else if (n_out % 16 == 0) launch(qd::DecIq2xxsTile::make(base, n_blocks, blocks));
            else launch(qd::DecIq2xxs::make(base, n_blocks, blocks));
            break;
        case 17:  // #253
            if (n_out % 256 == 0) tiles2(qd::TileIq2xs{});
            else if (n_out % 128 == 0) tiles(qd::TileIq2xs{});
            else if (n_out % 16 == 0) launch(qd::DecIq2xsTile::make(base, n_blocks, blocks));
            else launch(qd::DecIq2xs::make(base, n_blocks, blocks));
            break;
        case 22:  // #253
            if (n_out % 256 == 0) tiles2(qd::TileIq2s{});
            else if (n_out % 128 == 0) tiles(qd::TileIq2s{});
            else if (n_out % 16 == 0) launch(qd::DecIq2sTile::make(base, n_blocks, blocks));
            else launch(qd::DecIq2s::make(base, n_blocks, blocks));
            break;
        case 23:  // the tiles (#244) when rows % 16 == 0, the GEMM's tile kernel when % 128
            if (n_out % 256 == 0) tiles2(qd::TileIq4xs{});
            else if (n_out % 128 == 0) tiles(qd::TileIq4xs{});
            else if (n_out % 16 == 0) launch(qd::DecIq4xsTile::make(base, n_blocks, blocks));
            else launch(qd::DecIq4xs::make(base, n_blocks, blocks));
            break;
        case 12:
            if (n_out % 128 == 0) tiles(qd::TileQ4k{});
            else if (n_out % 16 == 0) launch(qd::DecQ4kTile::make(base, n_blocks, blocks));
            else launch(qd::DecQ4k::make(base, n_blocks, blocks));
            break;
        case 10:  // #253
            if (n_out % 256 == 0) tiles2(qd::TileQ2k{});
            else if (n_out % 128 == 0) tiles(qd::TileQ2k{});
            else if (n_out % 16 == 0) launch(qd::DecQ2kTile::make(base, n_blocks, blocks));
            else launch(qd::DecQ2k::make(base, n_blocks, blocks));
            break;
        case 14:  // #258
            if (n_out % 256 == 0) tiles2(qd::TileQ6k{});
            else if (n_out % 128 == 0) tiles(qd::TileQ6k{});
            else if (n_out % 16 == 0) launch(qd::DecQ6kTile::make(base, n_blocks, blocks));
            else launch(qd::DecQ6k::make(base, n_blocks, blocks));
            break;
        case 29:  // #275
            if (n_out % 256 == 0) tiles2(qd::TileIq1m{});
            else if (n_out % 128 == 0) tiles(qd::TileIq1m{});
            else if (n_out % 16 == 0) launch(qd::DecIq1mTile::make(base, n_blocks, blocks));
            else launch(qd::DecIq1m::make(base, n_blocks, blocks));
            break;
        default: return false;
    }
    return hipGetLastError() == hipSuccess;
}

} // namespace

bool gemm_q(const uint32_t type, const void * packed, const void * x, float * y, const int64_t n_out,
            const int64_t k, const int64_t tokens, const int64_t ldy, hipStream_t stream) {
    return gemm_q_impl<false>(type, packed, x, y, n_out, k, tokens, ldy, Epi{nullptr, nullptr}, stream);
}

bool gemm_q_swiglu(const uint32_t type, const void * packed, const void * x, const float * gate, void * out_f16,
                   const int64_t n_out, const int64_t k, const int64_t tokens, hipStream_t stream) {
    const auto aligned = [](const void * p) { return ((uintptr_t) p & 15) == 0; };
    if (gate == nullptr || out_f16 == nullptr || !aligned(gate) || !aligned(out_f16) || n_out % 8 != 0) {
        return false;
    }
    // y only passes the shared checks: the SWI stores go to out_f16
    return gemm_q_impl<true>(type, packed, x, const_cast<float *>(gate), n_out, k, tokens, n_out,
                             Epi{gate, static_cast<__half *>(out_f16)}, stream);
}

} // namespace omph::kernels
