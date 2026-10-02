// M9 stage A (#178): an IQ3_S layout made for WMMA GEMVs, with the kernel that
// reads it, against the engine's IQ3_S kernels (iq3s_diag.hip's copies), on
// random weights shaped like the model's 144 IQ3_S tensors.
//
// Layout: 16-row tiles; per (tile, 256-weight block) 1760 bytes, the same as
// the 16 rows' GGUF blocks, so no extra bytes:
//   qs     1024 B  lane l (row l % 16, half h = l / 16): 32 B, sub-block sb's
//                  qs word h (the index bytes of groups 2h, 2h + 1) at 4 sb
//   signs   512 B  lane l: 16 B, sub-block sb's sign bytes (h = 0: bytes 1, 3;
//                  h = 1: bytes 0, 2 of the sign word) at 2 sb
//   qh      128 B  lane l: 4 B, sub-block sb's qh nibble h at bits 4 sb
//   scales   64 B  row r: the block's 4 scale bytes
//   d        32 B  row r: the block's f16 scale
// Lane h of a row takes weights 16 h .. 16 h + 15 of each sub-block (two
// 8-weight WMMA k steps); x is read with the same k permutation, so the dot
// products are unchanged. Every lane's loads are contiguous and wide, and no
// two lanes read the same bytes except the 6-byte row metadata.
//   hipcc -O3 --offload-arch=gfx1200 -std=c++20 -I engine/src -I bench/gemv -o /var/tmp/iq3s_wil bench/gemv/iq3s_wil.hip
#define IQ3S_DIAG_NO_MAIN
#include <cstring>

#include "iq3s_diag.hip"

namespace {

typedef _Float16 wh8 __attribute__((ext_vector_type(8)));
typedef float wf8 __attribute__((ext_vector_type(8)));
typedef _Float16 wh2 __attribute__((ext_vector_type(2)));

constexpr long long kTileBytes = 1760;  // per 16-row tile and 256-weight block

// GGUF-layout host arrays -> the tiled layout
std::vector<uint8_t> to_tiles(const long long rows, const long long blocks, const std::vector<uint8_t> & qs,
                              const std::vector<uint8_t> & qh, const std::vector<uint8_t> & sg,
                              const std::vector<uint8_t> & sc, const std::vector<uint8_t> & d) {
    std::vector<uint8_t> out((size_t) (rows / 16) * blocks * kTileBytes);
    for (long long tile = 0; tile < rows / 16; ++tile) {
        for (long long b = 0; b < blocks; ++b) {
            uint8_t * o = out.data() + (tile * blocks + b) * kTileBytes;
            for (int l = 0; l < 32; ++l) {
                const long long row = tile * 16 + (l & 15);
                const int h = l >> 4;
                uint32_t qhw = 0;
                for (int sb = 0; sb < 8; ++sb) {
                    const long long s = b * 8 + sb;  // sub-block in the row
                    std::memcpy(o + l * 32 + sb * 4, &qs[(row * blocks * 8 + s) * 8 + h * 4], 4);
                    const uint8_t * w = &sg[(row * blocks * 8 + s) * 4];
                    o[1024 + l * 16 + sb * 2] = h == 0 ? w[1] : w[0];
                    o[1024 + l * 16 + sb * 2 + 1] = h == 0 ? w[3] : w[2];
                    qhw |= (uint32_t) ((qh[row * blocks * 8 + s] >> (4 * h)) & 0xF) << (4 * sb);
                }
                std::memcpy(o + 1536 + l * 4, &qhw, 4);
            }
            for (int r = 0; r < 16; ++r) {
                const long long row = tile * 16 + r;
                std::memcpy(o + 1664 + r * 4, &sc[(row * blocks + b) * 4], 4);
                std::memcpy(o + 1728 + r * 2, &d[(row * blocks + b) * 2], 2);
            }
        }
    }
    return out;
}

// grid: (rows / 16 / (8 / KW)); 256 threads: (8 / KW) row tiles x KW k slices.
// NT tokens (1..16), x [NT][k] f16, y [NT][rows]. Sums in LDS in a fixed order.
template <int NT, int KW>
__global__ __launch_bounds__(256) void k_wil(const uint8_t * __restrict__ wt, const __half * __restrict__ x,
                                            float * __restrict__ y, const long long rows, const long long blocks) {
    constexpr int TILES = 8 / KW;
    __shared__ uint2 grid[512];  // as half2 pairs
    __shared__ float red[8][16][17];
    const int tid = (int) threadIdx.x;
    for (int i = tid; i < 512; i += blockDim.x) {
        const uint32_t g = omph::quant::kIq3sGrid[i];
        grid[i] = make_uint2(grid_pair_to_half2<0x04010400u>(g), grid_pair_to_half2<0x04030402u>(g));
    }
    __syncthreads();
    const int lane = tid & 31;
    const int wv = tid >> 5;
    const int tl = wv / KW;
    const int kp = wv % KW;
    const int h = lane >> 4;
    const int r = lane & 15;
    const int t = lane & 15;  // B column: the token
    const long long tile = (long long) blockIdx.x * TILES + tl;
    const long long k = blocks * 256;
    const long long per = (blocks + KW - 1) / KW;
    const long long b0 = kp * per;
    const long long b1 = b0 + per < blocks ? b0 + per : blocks;
    wf8 acc = {};
    for (long long b = b0; b < b1; ++b) {
        const uint8_t * o = wt + (tile * blocks + b) * kTileBytes;
        const uint4 q0 = *reinterpret_cast<const uint4 *>(o + lane * 32);
        const uint4 q1 = *reinterpret_cast<const uint4 *>(o + lane * 32 + 16);
        const uint4 s4 = *reinterpret_cast<const uint4 *>(o + 1024 + lane * 16);
        const uint32_t qhw = *reinterpret_cast<const uint32_t *>(o + 1536 + lane * 4);
        const uint32_t scw = *reinterpret_cast<const uint32_t *>(o + 1664 + r * 4);
        const float dv = __half2float(*reinterpret_cast<const __half *>(o + 1728 + r * 2));
        const uint32_t qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        const uint32_t sw[4] = {s4.x, s4.y, s4.z, s4.w};
#pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            const uint32_t q = qw[sb];
            const uint32_t qhn = (qhw >> (4 * sb)) & 0xFu;
            const uint32_t s16 = (sw[sb >> 1] >> (16 * (sb & 1))) & 0xFFFFu;
            // the sign word with this half's bytes back at their positions
            const uint32_t sg = h == 0 ? (((s16 & 0xFFu) << 8) | ((s16 >> 8) << 24))
                                       : ((s16 & 0xFFu) | ((s16 >> 8) << 16));
            const uint32_t sc = (scw >> (8 * (sb >> 1))) & 0xFFu;
            const float dbr = dv * (1.0f + 2.0f * (float) ((sb & 1) ? (sc >> 4) : (sc & 0xF)));
            const _Float16 dh = (_Float16) dbr;
            const wh2 d2 = {dh, dh};
            const long long s = b * 8 + sb;
#pragma unroll
            for (int gl = 0; gl < 2; ++gl) {
                const int g = 2 * h + gl;  // the group in the sub-block
                const uint32_t word = (q >> (16 * gl)) & 0xFFFFu;
                const uint32_t i1 = (word & 0xFFu) | (((qhn >> (2 * gl)) & 1u) << 8);
                const uint32_t i2 = (word >> 8) | (((qhn >> (2 * gl + 1)) & 1u) << 8);
                const uint2 g1 = grid[i1];
                const uint2 g2 = grid[i2];
                constexpr uint32_t kSign2 = 0x80008000u;
                const uint32_t wv4[4] = {((sg << (4 * g)) & kSign2) | g1.x, ((sg << (4 * g + 1)) & kSign2) | g1.y,
                                         ((sg << (4 * g + 2)) & kSign2) | g2.x, ((sg << (4 * g + 3)) & kSign2) | g2.y};
                wh8 a;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const wh2 p = __builtin_bit_cast(wh2, wv4[i]) * d2;
                    a[2 * i] = p[0];
                    a[2 * i + 1] = p[1];
                }
                wh8 bv = {};
                if (t < NT) bv = *reinterpret_cast<const wh8 *>(x + t * k + s * 32 + h * 16 + gl * 8);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bv, acc);
            }
        }
    }
#pragma unroll
    for (int v = 0; v < 8; ++v) red[wv][h * 8 + v][t] = acc[v];
    __syncthreads();
    if (tid < TILES * 16 * NT) {
        const int tt2 = tid / (16 * NT);
        const int m = (tid / NT) % 16;
        const int tt = tid % NT;
        float sum = red[tt2 * KW][m][tt];
#pragma unroll
        for (int i = 1; i < KW; ++i) sum += red[tt2 * KW + i][m][tt];
        const long long row = ((long long) blockIdx.x * TILES + tt2) * 16 + m;
        if (row < rows) y[tt * rows + row] = sum;
    }
}

// The same tiled layout as a prefill GEMM: a workgroup = 8 waves x 16 rows
// (128 rows) x 256 tokens; each wave decodes its rows' weights straight into
// A operands (no LDS round trip for W) and reuses them for 16 token tiles;
// x is staged in LDS one sub-block (32 k) at a time, double-buffered.
constexpr int kGT = 256;    // tokens per workgroup
constexpr int kXLd = 40;    // halves per staged token row (32 + 8 pad)
__global__ __launch_bounds__(256) void k_wgemm(const uint8_t * __restrict__ wt, const __half * __restrict__ x,
                                              float * __restrict__ y, const long long rows, const long long blocks,
                                              const long long tokens) {
    __shared__ uint2 grid[512];
    __shared__ _Float16 xs[2][kGT * kXLd];
    const int tid = (int) threadIdx.x;
    for (int i = tid; i < 512; i += blockDim.x) {
        const uint32_t g = omph::quant::kIq3sGrid[i];
        grid[i] = make_uint2(grid_pair_to_half2<0x04010400u>(g), grid_pair_to_half2<0x04030402u>(g));
    }
    const int lane = tid & 31;
    const int wv = tid >> 5;
    const int h = lane >> 4;
    const int r = lane & 15;
    const long long tile = (long long) blockIdx.y * 8 + wv;
    const long long n0 = (long long) blockIdx.x * kGT;
    const long long k = blocks * 256;
    // staging: thread tid stages token tid's 32 halves (64 B) of a sub-block
    const long long tx = n0 + tid < tokens ? n0 + tid : 0;
    const uint4 * xrow = reinterpret_cast<const uint4 *>(x + tx * k);
    const auto stage = [&](const long long s, _Float16 * buf) {
        const uint4 * src = xrow + s * 4;
        uint4 * dst = reinterpret_cast<uint4 *>(buf + tid * kXLd);
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
    };
    wf8 acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j] = wf8{};
    stage(0, xs[0]);
    __syncthreads();
    int buf = 0;
    for (long long b = 0; b < blocks; ++b) {
        const uint8_t * o = wt + (tile * blocks + b) * kTileBytes;
        const uint4 q0 = *reinterpret_cast<const uint4 *>(o + lane * 32);
        const uint4 q1 = *reinterpret_cast<const uint4 *>(o + lane * 32 + 16);
        const uint4 s4 = *reinterpret_cast<const uint4 *>(o + 1024 + lane * 16);
        const uint32_t qhw = *reinterpret_cast<const uint32_t *>(o + 1536 + lane * 4);
        const uint32_t scw = *reinterpret_cast<const uint32_t *>(o + 1664 + r * 4);
        const float dv = __half2float(*reinterpret_cast<const __half *>(o + 1728 + r * 2));
        const uint32_t qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        const uint32_t sw[4] = {s4.x, s4.y, s4.z, s4.w};
#pragma unroll 1
        for (int sb = 0; sb < 8; ++sb) {
            const long long s = b * 8 + sb;
            if (s + 1 < blocks * 8) stage(s + 1, xs[buf ^ 1]);  // the next sub-block's x
            const uint32_t q = qw[sb];
            const uint32_t qhn = (qhw >> (4 * sb)) & 0xFu;
            const uint32_t s16 = (sw[sb >> 1] >> (16 * (sb & 1))) & 0xFFFFu;
            const uint32_t sg = h == 0 ? (((s16 & 0xFFu) << 8) | ((s16 >> 8) << 24))
                                       : ((s16 & 0xFFu) | ((s16 >> 8) << 16));
            const uint32_t sc = (scw >> (8 * (sb >> 1))) & 0xFFu;
            const float dbr = dv * (1.0f + 2.0f * (float) ((sb & 1) ? (sc >> 4) : (sc & 0xF)));
            const _Float16 dh = (_Float16) dbr;
            const wh2 d2 = {dh, dh};
            wh8 a[2];
#pragma unroll
            for (int gl = 0; gl < 2; ++gl) {
                const int g = 2 * h + gl;
                const uint32_t word = (q >> (16 * gl)) & 0xFFFFu;
                const uint32_t i1 = (word & 0xFFu) | (((qhn >> (2 * gl)) & 1u) << 8);
                const uint32_t i2 = (word >> 8) | (((qhn >> (2 * gl + 1)) & 1u) << 8);
                const uint2 g1 = grid[i1];
                const uint2 g2 = grid[i2];
                constexpr uint32_t kSign2 = 0x80008000u;
                const uint32_t wv4[4] = {((sg << (4 * g)) & kSign2) | g1.x, ((sg << (4 * g + 1)) & kSign2) | g1.y,
                                         ((sg << (4 * g + 2)) & kSign2) | g2.x, ((sg << (4 * g + 3)) & kSign2) | g2.y};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const wh2 p = __builtin_bit_cast(wh2, wv4[i]) * d2;
                    a[gl][2 * i] = p[0];
                    a[gl][2 * i + 1] = p[1];
                }
            }
            const _Float16 * cur = xs[buf];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
#pragma unroll
                for (int gl = 0; gl < 2; ++gl) {
                    const wh8 bv = *reinterpret_cast<const wh8 *>(cur + (j * 16 + (lane & 15)) * kXLd + h * 16 + gl * 8);
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
#pragma unroll
            for (int v = 0; v < 8; ++v) y[t * rows + tile * 16 + h * 8 + v] = acc[j][v];
        }
    }
}

struct TT {  // a tensor in both layouts
    Tensor t;
    uint8_t * tiles;
};

template <int NT, int KW>
void launch_wil(const TT & t, const __half * x, float * y) {
    constexpr int TILES = 8 / KW;
    k_wil<NT, KW><<<(unsigned) (t.t.rows / 16 / TILES), 256>>>(t.tiles, x, y, t.t.rows, t.t.k / 256);
}

template <int NT, int KW>
float wil_ms(const std::vector<TT> & ts, const __half * x, float * y, int reps) {
    hipEvent_t a, b;
    (void) hipEventCreate(&a);
    (void) hipEventCreate(&b);
    std::vector<float> v;
    for (int i = 0; i < reps; ++i) {
        (void) hipEventRecord(a, nullptr);
        for (const TT & t : ts) launch_wil<NT, KW>(t, x, y);
        (void) hipEventRecord(b, nullptr);
        (void) hipEventSynchronize(b);
        float ms = 0;
        (void) hipEventElapsedTime(&ms, a, b);
        v.push_back(ms);
    }
    check(hipGetLastError(), "wil launch");
    std::sort(v.begin(), v.end());
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
    check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, k_pw<kFull>, 256, kChunkSub * 64 + 512 * 4), "occ");
    g_fit_pw = per_cu * prop.multiProcessorCount;
    std::mt19937 rng(174);
    std::vector<TT> tensors;
    long long max_rows = 0;
    for (const auto & s : shapes) {
        for (int i = 0; i < s.n; ++i) {
            const long long blocks = s.k / 256;
            const long long nb = s.rows * blocks;
            const size_t sizes[5] = {(size_t) nb * 64, (size_t) nb * 8, (size_t) nb * 32, (size_t) nb * 4, (size_t) nb * 2};
            std::vector<uint8_t> h[5];
            uint8_t * p[5];
            for (int a = 0; a < 5; ++a) {
                h[a].resize(sizes[a]);
                if (a == 4) {
                    auto * hh = reinterpret_cast<__half *>(h[a].data());
                    for (size_t j = 0; j < sizes[a] / 2; ++j) hh[j] = __float2half(0.001f * (float) (rng() % 64 + 1));
                } else {
                    for (auto & c : h[a]) c = (uint8_t) rng();
                }
                check(hipMalloc(&p[a], sizes[a]), "malloc");
                check(hipMemcpy(p[a], h[a].data(), sizes[a], hipMemcpyHostToDevice), "copy");
            }
            const std::vector<uint8_t> til = to_tiles(s.rows, blocks, h[0], h[1], h[2], h[3], h[4]);
            uint8_t * pt;
            check(hipMalloc(&pt, til.size()), "malloc");
            check(hipMemcpy(pt, til.data(), til.size(), hipMemcpyHostToDevice), "copy");
            tensors.push_back({{s.rows, s.k, s.name, {p[0], p[1], p[2], p[3], p[4]}}, pt});
            max_rows = std::max(max_rows, s.rows);
        }
    }
    std::vector<__half> xh(16 * 17408);
    for (auto & v : xh) v = __float2half((float) ((int) (rng() % 2001) - 1000) / 1000.0f);
    __half * x;
    float * y;
    float * y2;
    check(hipMalloc(&x, xh.size() * 2), "malloc");
    check(hipMemcpy(x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice), "copy");
    check(hipMalloc(&y, 16 * max_rows * 4), "malloc");
    check(hipMalloc(&y2, 16 * max_rows * 4), "malloc");
    std::vector<const Tensor *> all;
    for (const TT & t : tensors) all.push_back(&t.t);
    for (int i = 0; i < 100; ++i) (void) pass_ms<kFull>(all, 1, x, y, false, 1);  // warm-up

    double worst = 0.0;
    int differ = 0;
    for (const int idx : {0, 10, 60, 143}) {
        const TT & t = tensors[(size_t) idx];
        std::vector<float> ref(t.t.rows), got(t.t.rows), g16((size_t) 16 * t.t.rows);
        launch<kFull>(t.t, 1, x, y, nullptr, false);
        check(hipMemcpy(ref.data(), y, t.t.rows * 4, hipMemcpyDeviceToHost), "copy");
        launch_wil<1, 8>(t, x, y2);
        check(hipMemcpy(got.data(), y2, t.t.rows * 4, hipMemcpyDeviceToHost), "copy");
        launch_wil<16, 8>(t, x, y2);
        check(hipMemcpy(g16.data(), y2, 16 * t.t.rows * 4, hipMemcpyDeviceToHost), "copy");
        double mx = 0.0, md = 0.0;
        for (long long rr = 0; rr < t.t.rows; ++rr) {
            mx = std::max(mx, (double) std::fabs(ref[rr]));
            md = std::max(md, (double) std::fabs(ref[rr] - got[rr]));
            if (std::memcmp(&got[rr], &g16[rr], 4) != 0) ++differ;
        }
        worst = std::max(worst, md / mx);
    }
    std::printf("tiled WMMA vs the engine's kernel: max|d| / max|ref| = %.2e; NT = 16 token 0 vs NT = 1: %d rows differ\n",
                worst, differ);
    const int reps = 9;
    std::printf("all 144 IQ3_S tensors, ms per pass (median of %d)\n", reps);
    std::printf("  engine        NT = 1 %.3f   NT = 4 %.3f\n", pass_ms<kFull>(all, 1, x, y, false, reps),
                pass_ms<kFull>(all, 4, x, y, false, reps));
    std::printf("  tiled KW=2    NT = 1 %.3f   NT = 4 %.3f\n", wil_ms<1, 2>(tensors, x, y, reps),
                wil_ms<4, 2>(tensors, x, y, reps));
    std::printf("  tiled KW=4    NT = 1 %.3f   NT = 4 %.3f\n", wil_ms<1, 4>(tensors, x, y, reps),
                wil_ms<4, 4>(tensors, x, y, reps));
    std::printf("  tiled KW=8    NT = 1 %.3f   NT = 4 %.3f   NT = 8 %.3f   NT = 16 %.3f\n", wil_ms<1, 8>(tensors, x, y, reps),
                wil_ms<4, 8>(tensors, x, y, reps), wil_ms<8, 8>(tensors, x, y, reps), wil_ms<16, 8>(tensors, x, y, reps));
    // prefill GEMM on the tiled layout, T = 512
    {
        const long long T = 512;
        __half * xg;
        float * yg;
        check(hipMalloc(&xg, (size_t) T * 17408 * 2), "malloc");
        check(hipMalloc(&yg, (size_t) T * max_rows * 4), "malloc");
        std::vector<__half> xgh((size_t) T * 17408);
        for (auto & v : xgh) v = __float2half((float) ((int) (rng() % 2001) - 1000) / 1000.0f);
        check(hipMemcpy(xg, xgh.data(), xgh.size() * 2, hipMemcpyHostToDevice), "copy");
        double flop = 0;
        for (const TT & t : tensors) flop += 2.0 * t.t.rows * t.t.k * T;
        hipEvent_t a, b;
        (void) hipEventCreate(&a);
        (void) hipEventCreate(&b);
        std::vector<float> v;
        for (int rep = 0; rep < 5; ++rep) {
            (void) hipEventRecord(a, nullptr);
            for (const TT & t : tensors) {
                k_wgemm<<<dim3((unsigned) ((T + kGT - 1) / kGT), (unsigned) (t.t.rows / 128)), 256>>>(
                    t.tiles, xg, yg, t.t.rows, t.t.k / 256, T);
            }
            (void) hipEventRecord(b, nullptr);
            (void) hipEventSynchronize(b);
            float ms = 0;
            (void) hipEventElapsedTime(&ms, a, b);
            v.push_back(ms);
        }
        check(hipGetLastError(), "wgemm");
        std::sort(v.begin(), v.end());
        std::printf("  tiled GEMM T = 512: %.3f ms -> %.1f TFLOPS (engine gemm_q on the real IQ3_S tensors: 171.5 ms, 50.3)\n",
                    v[2], flop / v[2] / 1e9);
        // check: token 0 of the GEMM against the tiled GEMV on the same x row
        const TT & t = tensors[60];
        k_wgemm<<<dim3(2, (unsigned) (t.t.rows / 128)), 256>>>(t.tiles, xg, yg, t.t.rows, t.t.k / 256, T);
        k_wil<1, 8><<<(unsigned) (t.t.rows / 16), 256>>>(t.tiles, xg, y2, t.t.rows, t.t.k / 256);
        std::vector<float> g1(t.t.rows), g2(t.t.rows);
        check(hipMemcpy(g1.data(), yg, t.t.rows * 4, hipMemcpyDeviceToHost), "copy");
        check(hipMemcpy(g2.data(), y2, t.t.rows * 4, hipMemcpyDeviceToHost), "copy");
        double md = 0, mx = 0;
        for (long long rr = 0; rr < t.t.rows; ++rr) {
            md = std::max(md, (double) std::fabs(g1[rr] - g2[rr]));
            mx = std::max(mx, (double) std::fabs(g2[rr]));
        }
        std::printf("  GEMM token 0 vs GEMV: max|d| / max|ref| = %.2e (k order differs: split-K sums)\n", md / mx);
    }
    return 0;
}
