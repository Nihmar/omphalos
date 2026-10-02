// IQ3_S GEMV on WMMA tiles, a prototype (#206): the engine's decode, the
// products on v_wmma (16 rows x up to 16 tokens per wave), the k range split
// over the workgroup's waves and summed in LDS in a fixed order. A WMMA output
// column does not depend on the other columns, so the rows are bit-identical
// for 1 and N tokens by construction (checked below). The weights enter the
// tiles as f16 (scale x grid value), as the prefill's dequantized GEMM: not
// the current GEMVs' arithmetic, so their outputs are compared by error.
//
// Against the engine's kernels (bench/gemv/iq3s_diag.hip's copies), random
// weights shaped like the model's 144 IQ3_S tensors, ms per pass.
//   hipcc -O3 --offload-arch=gfx1200 -std=c++20 -I engine/src -o /var/tmp/iq3s_wmma bench/gemv/iq3s_wmma.hip
#define IQ3S_DIAG_NO_MAIN
#include <cstring>
#include "iq3s_diag.hip"

namespace {

typedef _Float16 wh8 __attribute__((ext_vector_type(8)));
typedef float wf8 __attribute__((ext_vector_type(8)));

// grid: (ceil(rows / 16)); 32 * KW threads. NT tokens (1..16), x [NT][k] f16, y [NT][rows].
template <int NT, int KW>
__global__ __launch_bounds__(32 * KW) void k_wmma(const W w, const __half * __restrict__ x, float * __restrict__ y,
                                                  const long long rows, const long long n_sub) {
    __shared__ uint32_t grid[512];
    __shared__ float red[KW][16][17];
    const int tid = (int) threadIdx.x;
    for (int i = tid; i < 512; i += blockDim.x) grid[i] = omph::quant::kIq3sGrid[i];
    __syncthreads();
    const int lane = tid & 31;
    const int wv = tid >> 5;
    const int hk = lane >> 4;  // the k half of the A / B operands
    const int t = lane & 15;   // B column: the token
    const long long row0 = (long long) blockIdx.x * 16;
    const long long row = row0 + (lane & 15) < rows ? row0 + (lane & 15) : rows - 1;
    const long long blocks = n_sub / 8;
    const long long rb = row * blocks;
    const long long k = n_sub * 32;
    const long long per = (n_sub + KW - 1) / KW;
    const long long s0 = wv * per;
    const long long s1 = s0 + per < n_sub ? s0 + per : n_sub;
    wf8 acc = {};
    for (long long s = s0; s < s1; ++s) {
        const uint2 q = *reinterpret_cast<const uint2 *>(w.qs + rb * 64 + s * 8);
        const uint32_t qhb = w.qh[rb * 8 + s];
        const uint32_t sg = *reinterpret_cast<const uint32_t *>(w.signs + rb * 32 + s * 4);
        const uint8_t sc = w.scales[rb * 4 + s / 2];
        const float dv = __half2float(*reinterpret_cast<const __half *>(w.d + rb * 2 + (s / 8) * 2));
        const float dbr = dv * (1.0f + 2.0f * (float) ((s & 1) ? (sc >> 4) : (sc & 0xF)));
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int g = 2 * j + hk;  // 8 weights g * 8 .. + 7 of the sub-block
            const uint32_t word = ((g < 2 ? q.x : q.y) >> (16 * (g & 1))) & 0xFFFFu;
            const uint32_t i1 = (word & 0xFFu) | (((qhb >> (2 * g)) & 1u) << 8);
            const uint32_t i2 = (word >> 8) | (((qhb >> (2 * g + 1)) & 1u) << 8);
            const uint32_t b1 = grid[i1];
            const uint32_t b2 = grid[i2];
            constexpr uint32_t kSign2 = 0x80008000u;
            const uint32_t wv4[4] = {((sg << (4 * g)) & kSign2) | grid_pair_to_half2<0x04010400u>(b1),
                                     ((sg << (4 * g + 1)) & kSign2) | grid_pair_to_half2<0x04030402u>(b1),
                                     ((sg << (4 * g + 2)) & kSign2) | grid_pair_to_half2<0x04010400u>(b2),
                                     ((sg << (4 * g + 3)) & kSign2) | grid_pair_to_half2<0x04030402u>(b2)};
            const _Float16 * hv = reinterpret_cast<const _Float16 *>(wv4);
            wh8 a;
#pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = (_Float16) ((float) hv[i] * dbr);
            wh8 b = {};
            if (t < NT) b = *reinterpret_cast<const wh8 *>(x + t * k + s * 32 + g * 8);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
        }
    }
    // D: lane -> token lane % 16, rows (lane / 16) * 8 + v; the waves' partials summed in order
#pragma unroll
    for (int v = 0; v < 8; ++v) red[wv][hk * 8 + v][t] = acc[v];
    __syncthreads();
    if (tid < 16 * NT) {
        const int m = tid / NT;
        const int tt = tid % NT;
        float sum = red[0][m][tt];
#pragma unroll
        for (int i = 1; i < KW; ++i) sum += red[i][m][tt];
        if (row0 + m < rows) y[tt * rows + row0 + m] = sum;
    }
}

template <int NT, int KW>
void launch_wmma(const Tensor & t, const __half * x, float * y) {
    k_wmma<NT, KW><<<(unsigned) ((t.rows + 15) / 16), 32 * KW>>>(t.w, x, y, t.rows, t.k / 32);
}

template <int NT, int KW>
float wmma_ms(const std::vector<const Tensor *> & ts, const __half * x, float * y, int reps) {
    hipEvent_t a, b;
    (void) hipEventCreate(&a);
    (void) hipEventCreate(&b);
    std::vector<float> v;
    for (int i = 0; i < reps; ++i) {
        (void) hipEventRecord(a, nullptr);
        for (const Tensor * t : ts) launch_wmma<NT, KW>(*t, x, y);
        (void) hipEventRecord(b, nullptr);
        (void) hipEventSynchronize(b);
        float ms = 0;
        (void) hipEventElapsedTime(&ms, a, b);
        v.push_back(ms);
    }
    check(hipGetLastError(), "wmma launch");
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// v2: one 256-weight block per iteration per row (wide contiguous loads), the
// workgroup's 8 waves as (8 / KW) row tiles x KW k slices (whole blocks).
template <int NT, int KW>
__global__ __launch_bounds__(256) void k_wmma2(const W w, const __half * __restrict__ x, float * __restrict__ y,
                                              const long long rows, const long long n_sub) {
    constexpr int TILES = 8 / KW;
    __shared__ uint32_t grid[512];
    __shared__ float red[8][16][17];
    const int tid = (int) threadIdx.x;
    for (int i = tid; i < 512; i += blockDim.x) grid[i] = omph::quant::kIq3sGrid[i];
    __syncthreads();
    const int lane = tid & 31;
    const int wv = tid >> 5;
    const int tile = wv / KW;
    const int kp = wv % KW;
    const int hk = lane >> 4;
    const int t = lane & 15;
    const long long row0 = ((long long) blockIdx.x * TILES + tile) * 16;
    const long long row = row0 + (lane & 15) < rows ? row0 + (lane & 15) : rows - 1;
    const long long blocks = n_sub / 8;
    const long long rb = row * blocks;
    const long long k = n_sub * 32;
    const long long per = (blocks + KW - 1) / KW;
    const long long b0 = kp * per;
    const long long b1 = b0 + per < blocks ? b0 + per : blocks;
    wf8 acc = {};
    for (long long blk = b0; blk < b1; ++blk) {
        const uint4 * qv = reinterpret_cast<const uint4 *>(w.qs + (rb + blk) * 64);
        const uint4 q4[4] = {qv[0], qv[1], qv[2], qv[3]};
        const uint2 qh8 = *reinterpret_cast<const uint2 *>(w.qh + (rb + blk) * 8);
        const uint4 * sv = reinterpret_cast<const uint4 *>(w.signs + (rb + blk) * 32);
        const uint4 s4[2] = {sv[0], sv[1]};
        const uint32_t sc4 = *reinterpret_cast<const uint32_t *>(w.scales + (rb + blk) * 4);
        const float dv = __half2float(*reinterpret_cast<const __half *>(w.d + (rb + blk) * 2));
        const uint32_t * qw = reinterpret_cast<const uint32_t *>(q4);
        const uint32_t * sw = reinterpret_cast<const uint32_t *>(s4);
        const uint8_t * qhb8 = reinterpret_cast<const uint8_t *>(&qh8);
#pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            const long long s = blk * 8 + sb;
            const uint32_t qx = qw[2 * sb], qy = qw[2 * sb + 1];
            const uint32_t qhb = qhb8[sb];
            const uint32_t sg = sw[sb];
            const uint32_t sc = (sc4 >> (8 * (sb / 2))) & 0xFFu;
            const float dbr = dv * (1.0f + 2.0f * (float) ((sb & 1) ? (sc >> 4) : (sc & 0xF)));
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int g = 2 * j + hk;
                const uint32_t word = ((g < 2 ? qx : qy) >> (16 * (g & 1))) & 0xFFFFu;
                const uint32_t i1 = (word & 0xFFu) | (((qhb >> (2 * g)) & 1u) << 8);
                const uint32_t i2 = (word >> 8) | (((qhb >> (2 * g + 1)) & 1u) << 8);
                const uint32_t bb1 = grid[i1];
                const uint32_t bb2 = grid[i2];
                constexpr uint32_t kSign2 = 0x80008000u;
                const uint32_t wv4[4] = {((sg << (4 * g)) & kSign2) | grid_pair_to_half2<0x04010400u>(bb1),
                                         ((sg << (4 * g + 1)) & kSign2) | grid_pair_to_half2<0x04030402u>(bb1),
                                         ((sg << (4 * g + 2)) & kSign2) | grid_pair_to_half2<0x04010400u>(bb2),
                                         ((sg << (4 * g + 3)) & kSign2) | grid_pair_to_half2<0x04030402u>(bb2)};
                const _Float16 * hv = reinterpret_cast<const _Float16 *>(wv4);
                wh8 a;
#pragma unroll
                for (int i = 0; i < 8; ++i) a[i] = (_Float16) ((float) hv[i] * dbr);
                wh8 b = {};
                if (t < NT) b = *reinterpret_cast<const wh8 *>(x + t * k + s * 32 + g * 8);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
            }
        }
    }
#pragma unroll
    for (int v = 0; v < 8; ++v) red[wv][hk * 8 + v][t] = acc[v];
    __syncthreads();
    if (tid < TILES * 16 * NT) {
        const int tl = tid / (16 * NT);
        const int m = (tid / NT) % 16;
        const int tt = tid % NT;
        float sum = red[tl * KW][m][tt];
#pragma unroll
        for (int i = 1; i < KW; ++i) sum += red[tl * KW + i][m][tt];
        const long long r0 = ((long long) blockIdx.x * TILES + tl) * 16;
        if (r0 + m < rows) y[tt * rows + r0 + m] = sum;
    }
}

template <int NT, int KW>
__global__ __launch_bounds__(256) void k_wmma3(const W w, const __half * __restrict__ x, float * __restrict__ y,
                                              const long long rows, const long long n_sub) {
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
    const int tile = wv / KW;
    const int kp = wv % KW;
    const int hk = lane >> 4;
    const int t = lane & 15;
    const long long row0 = ((long long) blockIdx.x * TILES + tile) * 16;
    const long long row = row0 + (lane & 15) < rows ? row0 + (lane & 15) : rows - 1;
    const long long blocks = n_sub / 8;
    const long long rb = row * blocks;
    const long long k = n_sub * 32;
    const long long per = (blocks + KW - 1) / KW;
    const long long b0 = kp * per;
    const long long b1 = b0 + per < blocks ? b0 + per : blocks;
    wf8 acc = {};
    for (long long blk = b0; blk < b1; ++blk) {
        const uint4 * qv = reinterpret_cast<const uint4 *>(w.qs + (rb + blk) * 64);
        const uint4 q4[4] = {qv[0], qv[1], qv[2], qv[3]};
        const uint2 qh8 = *reinterpret_cast<const uint2 *>(w.qh + (rb + blk) * 8);
        const uint4 * sv = reinterpret_cast<const uint4 *>(w.signs + (rb + blk) * 32);
        const uint4 s4[2] = {sv[0], sv[1]};
        const uint32_t sc4 = *reinterpret_cast<const uint32_t *>(w.scales + (rb + blk) * 4);
        const float dv = __half2float(*reinterpret_cast<const __half *>(w.d + (rb + blk) * 2));
        const uint32_t * qw = reinterpret_cast<const uint32_t *>(q4);
        const uint32_t * sw = reinterpret_cast<const uint32_t *>(s4);
        const uint8_t * qhb8 = reinterpret_cast<const uint8_t *>(&qh8);
#pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            const long long s = blk * 8 + sb;
            const uint32_t qx = qw[2 * sb], qy = qw[2 * sb + 1];
            const uint32_t qhb = qhb8[sb];
            const uint32_t sg = sw[sb];
            const uint32_t sc = (sc4 >> (8 * (sb / 2))) & 0xFFu;
            const float dbr = dv * (1.0f + 2.0f * (float) ((sb & 1) ? (sc >> 4) : (sc & 0xF)));
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int g = 2 * j + hk;
                const uint32_t word = ((g < 2 ? qx : qy) >> (16 * (g & 1))) & 0xFFFFu;
                const uint32_t i1 = (word & 0xFFu) | (((qhb >> (2 * g)) & 1u) << 8);
                const uint32_t i2 = (word >> 8) | (((qhb >> (2 * g + 1)) & 1u) << 8);
                const uint2 gg1 = grid[i1];
                const uint2 gg2 = grid[i2];
                constexpr uint32_t kSign2 = 0x80008000u;
                const uint32_t wv4[4] = {((sg << (4 * g)) & kSign2) | gg1.x, ((sg << (4 * g + 1)) & kSign2) | gg1.y,
                                         ((sg << (4 * g + 2)) & kSign2) | gg2.x, ((sg << (4 * g + 3)) & kSign2) | gg2.y};
                // the scale in f16, packed multiplies
                const _Float16 dh = (_Float16) dbr;
                typedef _Float16 h2v __attribute__((ext_vector_type(2)));
                const h2v d2 = {dh, dh};
                wh8 a;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const h2v p = __builtin_bit_cast(h2v, wv4[i]) * d2;
                    a[2 * i] = p[0];
                    a[2 * i + 1] = p[1];
                }
                wh8 b = {};
                if (t < NT) b = *reinterpret_cast<const wh8 *>(x + t * k + s * 32 + g * 8);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
            }
        }
    }
#pragma unroll
    for (int v = 0; v < 8; ++v) red[wv][hk * 8 + v][t] = acc[v];
    __syncthreads();
    if (tid < TILES * 16 * NT) {
        const int tl = tid / (16 * NT);
        const int m = (tid / NT) % 16;
        const int tt = tid % NT;
        float sum = red[tl * KW][m][tt];
#pragma unroll
        for (int i = 1; i < KW; ++i) sum += red[tl * KW + i][m][tt];
        const long long r0 = ((long long) blockIdx.x * TILES + tl) * 16;
        if (r0 + m < rows) y[tt * rows + r0 + m] = sum;
    }
}

template <int NT, int KW>
void launch_wmma3(const Tensor & t, const __half * x, float * y) {
    constexpr int TILES = 8 / KW;
    k_wmma3<NT, KW><<<(unsigned) ((t.rows + 16 * TILES - 1) / (16 * TILES)), 256>>>(t.w, x, y, t.rows, t.k / 32);
}

template <int NT, int KW>
float wmma3_ms(const std::vector<const Tensor *> & ts, const __half * x, float * y, int reps) {
    hipEvent_t a, b;
    (void) hipEventCreate(&a);
    (void) hipEventCreate(&b);
    std::vector<float> v;
    for (int i = 0; i < reps; ++i) {
        (void) hipEventRecord(a, nullptr);
        for (const Tensor * t : ts) launch_wmma3<NT, KW>(*t, x, y);
        (void) hipEventRecord(b, nullptr);
        (void) hipEventSynchronize(b);
        float ms = 0;
        (void) hipEventElapsedTime(&ms, a, b);
        v.push_back(ms);
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <int NT, int KW>
void launch_wmma2(const Tensor & t, const __half * x, float * y) {
    constexpr int TILES = 8 / KW;
    k_wmma2<NT, KW><<<(unsigned) ((t.rows + 16 * TILES - 1) / (16 * TILES)), 256>>>(t.w, x, y, t.rows, t.k / 32);
}

template <int NT, int KW>
float wmma2_ms(const std::vector<const Tensor *> & ts, const __half * x, float * y, int reps) {
    hipEvent_t a, b;
    (void) hipEventCreate(&a);
    (void) hipEventCreate(&b);
    std::vector<float> v;
    for (int i = 0; i < reps; ++i) {
        (void) hipEventRecord(a, nullptr);
        for (const Tensor * t : ts) launch_wmma2<NT, KW>(*t, x, y);
        (void) hipEventRecord(b, nullptr);
        (void) hipEventSynchronize(b);
        float ms = 0;
        (void) hipEventElapsedTime(&ms, a, b);
        v.push_back(ms);
    }
    check(hipGetLastError(), "wmma2 launch");
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
    std::vector<Tensor> tensors;
    long long max_rows = 0;
    for (const auto & s : shapes) {
        for (int i = 0; i < s.n; ++i) {
            const long long nb = s.rows * (s.k / 256);
            const size_t sizes[5] = {(size_t) nb * 64, (size_t) nb * 8, (size_t) nb * 32, (size_t) nb * 4, (size_t) nb * 2};
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
            }
            tensors.push_back({s.rows, s.k, s.name, {p[0], p[1], p[2], p[3], p[4]}});
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
    for (const Tensor & t : tensors) all.push_back(&t);
    for (int i = 0; i < 100; ++i) (void) pass_ms<kFull>(all, 1, x, y, false, 1);  // warm-up

    // accuracy vs the engine's kernel, and the NT identity
    double worst = 0.0;
    int differ = 0;
    for (const auto * t : {&tensors[0], &tensors[10], &tensors[60], &tensors[143]}) {
        std::vector<float> ref(t->rows), got(t->rows), g4((size_t) 4 * t->rows);
        launch<kFull>(*t, 1, x, y, nullptr, false);
        check(hipMemcpy(ref.data(), y, t->rows * 4, hipMemcpyDeviceToHost), "copy");
        launch_wmma<1, 8>(*t, x, y2);
        check(hipMemcpy(got.data(), y2, t->rows * 4, hipMemcpyDeviceToHost), "copy");
        launch_wmma3<1, 8>(*t, x, y2);
        check(hipMemcpy(got.data(), y2, t->rows * 4, hipMemcpyDeviceToHost), "copy");
        launch_wmma3<4, 8>(*t, x, y2);
        check(hipMemcpy(g4.data(), y2, 4 * t->rows * 4, hipMemcpyDeviceToHost), "copy");
        double mx = 0.0, md = 0.0;
        for (long long r = 0; r < t->rows; ++r) {
            mx = std::max(mx, (double) std::fabs(ref[r]));
            md = std::max(md, (double) std::fabs(ref[r] - got[r]));
            if (std::memcmp(&got[r], &g4[r], 4) != 0) ++differ;
        }
        worst = std::max(worst, md / mx);
    }
    std::printf("WMMA vs the engine's kernel: max|d| / max|ref| = %.2e; NT = 4 row 0 vs NT = 1: %d rows differ\n",
                worst, differ);
    const int reps = 9;
    std::printf("all 144 IQ3_S tensors, ms per pass (median of %d)\n", reps);
    std::printf("  engine  NT = 1 %.3f   NT = 4 %.3f\n", pass_ms<kFull>(all, 1, x, y, false, reps),
                pass_ms<kFull>(all, 4, x, y, false, reps));
    std::printf("  wmma3 KW=4 NT = 1 %.3f   NT = 4 %.3f\n", wmma3_ms<1, 4>(all, x, y, reps), wmma3_ms<4, 4>(all, x, y, reps));
    std::printf("  wmma3 KW=8 NT = 1 %.3f   NT = 4 %.3f   NT = 8 %.3f\n", wmma3_ms<1, 8>(all, x, y, reps),
                wmma3_ms<4, 8>(all, x, y, reps), wmma3_ms<8, 8>(all, x, y, reps));
    std::printf("  wmma2 KW=8 NT = 1 %.3f   NT = 4 %.3f   NT = 8 %.3f\n", wmma2_ms<1, 8>(all, x, y, reps),
                wmma2_ms<4, 8>(all, x, y, reps), wmma2_ms<8, 8>(all, x, y, reps));
    return 0;
}
