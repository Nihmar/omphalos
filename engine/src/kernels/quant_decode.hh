// Device-side decoding of the repacked quant layouts (PLAN.md §8.3), shared by
// the GEMVs, the f16 dequant of the GEMM path and the fused dequant + WMMA GEMM
// (#141). Include from .hip files only.
#pragma once

#include <cstdint>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "format/repack.hh"
#include "kernels/iq_tables.hh"

namespace omph::kernels::qd {

typedef _Float16 half2_v __attribute__((ext_vector_type(2)));

// Two unsigned byte codebook values (the bytes Sel picks from g) as a packed
// half2: the f16 with exponent field 25 and mantissa b is 1024 + b, so one byte
// permute builds 1024 + b and one packed add removes the 1024 (exact, b < 256).
template <uint32_t Sel>
__device__ inline uint32_t grid_pair_to_half2(const uint32_t g) {
    const uint32_t biased = __builtin_amdgcn_perm(0x64646464u, g, Sel);
    const half2_v v =
        __builtin_bit_cast(half2_v, biased) - half2_v{(_Float16) 1024, (_Float16) 1024};
    return __builtin_bit_cast(uint32_t, v);
}

// IQ3_XXS: the pair-ordered sign word (weight 2i at bit 15 - i, 2i + 1 at
// 31 - i, as for IQ3_S) of group l from the repacked aux word (repack.hh):
// even-weight bits land with one shift, odd ones with another, and weight 7's
// sign is the parity of the group's seven stored bits.
__device__ inline uint32_t iq3xxs_group_signs(const uint32_t a, const int l) {
    const uint32_t even = (a >> (4 * l)) & 0xFu;
    const uint32_t odd = (a >> (16 + 3 * l)) & 0x7u;
    const uint32_t par = (uint32_t) __builtin_popcount(even | (odd << 4)) & 1u;
    return (even << 12) | (odd << 29) | (par << 28);
}

// Q4_K: d * scale and dmin * min of sub-block j (0..7) from a block's 16-byte
// meta record: the GGUF 12-byte 6-bit scale/min area (ggml's
// get_scale_min_k4), then d and dmin (#94).
__device__ inline void q4k_meta(const uint4 m, const int j, float & ds, float & ms) {
    const int sh = 8 * (j & 3);
    const uint32_t b0 = (m.x >> sh) & 0xFFu;
    const uint32_t b1 = (m.y >> sh) & 0xFFu;
    const uint32_t b2 = (m.z >> sh) & 0xFFu;
    const uint32_t sc = j < 4 ? (b0 & 63u) : ((b2 & 0xFu) | ((b0 >> 6) << 4));
    const uint32_t mn = j < 4 ? (b1 & 63u) : ((b2 >> 4) | ((b1 >> 6) << 4));
    ds = __half2float(__ushort_as_half((unsigned short) (m.w & 0xFFFFu))) * (float) sc;
    ms = __half2float(__ushort_as_half((unsigned short) (m.w >> 16))) * (float) mn;
}

// IQ4_XS: d * (scale - 32) of sub-block ib (0..7) from a block's 8-byte meta
// record: scales_l (4 bits at 4*ib), scales_h (2 bits at 2*ib), d (#94).
__device__ inline float iq4_meta(const uint2 m, const int ib) {
    const uint32_t sc = ((m.x >> (4 * ib)) & 0xFu) | (((m.y >> (2 * ib)) & 3u) << 4);
    return __half2float(__ushort_as_half((unsigned short) (m.y >> 16))) * (float) ((int) sc - 32);
}

// The IQ4_NL codebook (omph::quant::kKvaluesIq4nl) + 128 as bytes, in four
// words: entries 0-3, 4-7, 8-11, 12-15.
constexpr uint32_t kIq4B0 = 0x3F2D1801u;
constexpr uint32_t kIq4B1 = 0x766A5D4Fu;
constexpr uint32_t kIq4B2 = 0xA6998D81u;
constexpr uint32_t kIq4B3 = 0xF1D9C5B5u;
constexpr uint32_t iq4_biased_word(const int w) {
    uint32_t v = 0;
    for (int j = 0; j < 4; ++j) {
        v |= (uint32_t) (uint8_t) (omph::quant::kKvaluesIq4nl[4 * w + j] + 128) << (8 * j);
    }
    return v;
}
static_assert(kIq4B0 == iq4_biased_word(0) && kIq4B1 == iq4_biased_word(1) &&
                  kIq4B2 == iq4_biased_word(2) && kIq4B3 == iq4_biased_word(3),
              "kIq4B* must be the IQ4_NL codebook + 128");

// Four 4-bit codes (one per byte) to their codebook values + 128, one per byte:
// two byte permutes look the low three bits up in each half of the table and a
// third picks the half by bit 3.
__device__ inline uint32_t iq4_lookup(const uint32_t codes) {
    const uint32_t idx = codes & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(kIq4B1, kIq4B0, idx);
    const uint32_t hi = __builtin_amdgcn_perm(kIq4B3, kIq4B2, idx);
    return __builtin_amdgcn_perm(hi, lo, 0x03020100u | ((codes & 0x08080808u) >> 1));
}

// Two biased codebook bytes (the bytes Sel picks) as a half2 of the values:
// 1024 + (v + 128) from the byte permute, minus 1152 (exact, |v| <= 127).
template <uint32_t Sel>
__device__ inline uint32_t iq4_pair_to_half2(const uint32_t g) {
    const uint32_t biased = __builtin_amdgcn_perm(0x64646464u, g, Sel);
    const half2_v v =
        __builtin_bit_cast(half2_v, biased) - half2_v{(_Float16) 1152, (_Float16) 1152};
    return __builtin_bit_cast(uint32_t, v);
}

// --- one 32-weight sub-block to f32 (#141) ----------------------------------
//
// One decoder per repacked type: load() reads the sub-block's words (row, s:
// the sub-block index within the row), decode() turns them into its 32
// weights in order, as the raw dequant kernels compute them (dequant.hip: same
// float expressions, same order). The grid values come out of the decode
// signed (exact in f16), and db * (+-g) == +-(db * g) exactly, so the f16 of
// every weight is bit-identical to dequantizing the GGUF bytes. Splitting
// load from decode lets a GEMM fetch the next tile's words while it
// multiplies the current one.

__device__ inline float half_lo(const uint32_t w) {
    return __half2float(__ushort_as_half((unsigned short) (w & 0xFFFFu)));
}
__device__ inline float half_hi(const uint32_t w) {
    return __half2float(__ushort_as_half((unsigned short) (w >> 16)));
}

// the 8 weights of a group l from four signed half2 words, times db
__device__ inline void put8(float (&v)[32], const int l, const float db, const uint32_t w0,
                            const uint32_t w1, const uint32_t w2, const uint32_t w3) {
    v[l * 8 + 0] = db * half_lo(w0);
    v[l * 8 + 1] = db * half_hi(w0);
    v[l * 8 + 2] = db * half_lo(w1);
    v[l * 8 + 3] = db * half_hi(w1);
    v[l * 8 + 4] = db * half_lo(w2);
    v[l * 8 + 5] = db * half_hi(w2);
    v[l * 8 + 6] = db * half_lo(w3);
    v[l * 8 + 7] = db * half_hi(w3);
}

constexpr uint32_t kSignBits = 0x80008000u;

__device__ inline float load_half(const uint8_t * p) {
    return __half2float(*reinterpret_cast<const __half *>(p));
}

struct DecIq3s {  // GGUF type 21
    const uint8_t * qs;
    const uint8_t * qh;
    const uint8_t * signs;
    const uint8_t * scales;
    const uint8_t * d;
    long long blocks;  // 256-weight blocks per row
    struct Raw {
        uint2 q;
        uint32_t qhb;
        uint32_t sg;
        uint32_t sc;
        float dv;
    };
    static DecIq3s make(const uint8_t * base, const int64_t n_blocks, const long long blocks) {
        const auto l = omph::format::iq3s_layout(n_blocks);
        return {base + l.qs_off, base + l.qh_off, base + l.signs_off, base + l.scales_off,
                base + l.d_off, blocks};
    }
    __device__ Raw load(const long long row, const long long s) const {
        Raw r;
        r.q = *reinterpret_cast<const uint2 *>(qs + row * blocks * 64 + s * 8);
        r.qhb = qh[row * blocks * 8 + s];
        r.sg = *reinterpret_cast<const uint32_t *>(signs + row * blocks * 32 + s * 4);
        r.sc = scales[row * blocks * 4 + s / 2];
        r.dv = load_half(d + row * blocks * 2 + (s / 8) * 2);
        return r;
    }
    __device__ void decode(const Raw & r, const long long s, float (&v)[32]) const {
        const float db = r.dv * (1.0f + 2.0f * (float) ((s & 1) ? (r.sc >> 4) : (r.sc & 0xF)));
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t lo = (r.q.x >> (16 * (l & 1))) & 0xFFFFu;
            const uint32_t word = l < 2 ? lo : (r.q.y >> (16 * (l & 1))) & 0xFFFFu;
            const uint32_t i1 = (word & 0xFFu) | (((r.qhb >> (2 * l)) & 1u) << 8);
            const uint32_t i2 = (word >> 8) | (((r.qhb >> (2 * l + 1)) & 1u) << 8);
            const uint32_t b1 = omph::quant::kIq3sGrid[i1];
            const uint32_t b2 = omph::quant::kIq3sGrid[i2];
            put8(v, l, db,
                 ((r.sg << (4 * l)) & kSignBits) | grid_pair_to_half2<0x04010400u>(b1),
                 ((r.sg << (4 * l + 1)) & kSignBits) | grid_pair_to_half2<0x04030402u>(b1),
                 ((r.sg << (4 * l + 2)) & kSignBits) | grid_pair_to_half2<0x04010400u>(b2),
                 ((r.sg << (4 * l + 3)) & kSignBits) | grid_pair_to_half2<0x04030402u>(b2));
        }
    }
};

// IQ3_S in the WMMA tiles of format/repack.hh (#178): the same Raw and the
// same values as DecIq3s, read from the tile of 16 rows and the block that
// hold (row, s): the qs words, sign bytes and qh nibbles of the row's two lanes.
struct DecIq3sTile {  // GGUF type 21, .omph layout 2
    const uint8_t * base;
    long long blocks;  // 256-weight blocks per row
    using Raw = DecIq3s::Raw;
    static DecIq3sTile make(const uint8_t * base, const int64_t /*n_blocks*/, const long long blocks) {
        return {base, blocks};
    }
    __device__ Raw load(const long long row, const long long s) const {
        const uint8_t * o = base + ((row >> 4) * blocks + (s >> 3)) * omph::format::kIq3sTileBytes;
        const int r = (int) (row & 15);
        const int sb = (int) (s & 7);
        Raw w;
        w.q.x = *reinterpret_cast<const uint32_t *>(o + r * 32 + sb * 4);
        w.q.y = *reinterpret_cast<const uint32_t *>(o + (r + 16) * 32 + sb * 4);
        const uint32_t q0 = *reinterpret_cast<const uint32_t *>(o + 1536 + r * 4);
        const uint32_t q1 = *reinterpret_cast<const uint32_t *>(o + 1536 + (r + 16) * 4);
        w.qhb = ((q0 >> (4 * sb)) & 0xFu) | (((q1 >> (4 * sb)) & 0xFu) << 4);
        const uint8_t * s0 = o + 1024 + r * 16 + sb * 2;         // half 0: bytes 1, 3
        const uint8_t * s1 = o + 1024 + (r + 16) * 16 + sb * 2;  // half 1: bytes 0, 2
        w.sg = (uint32_t) s1[0] | (uint32_t) s0[0] << 8 | (uint32_t) s1[1] << 16 | (uint32_t) s0[1] << 24;
        w.sc = o[1664 + r * 4 + (sb >> 1)];
        w.dv = load_half(o + 1728 + r * 2);
        return w;
    }
    __device__ void decode(const Raw & r, const long long s, float (&v)[32]) const {
        DecIq3s{}.decode(r, s, v);
    }
};

struct DecIq3xxs {  // GGUF type 18
    const uint8_t * qs;
    const uint8_t * aux;
    const uint8_t * d;
    long long blocks;
    struct Raw {
        uint2 q;
        uint32_t a;
        float dv;
    };
    static DecIq3xxs make(const uint8_t * base, const int64_t n_blocks, const long long blocks) {
        const auto l = omph::format::iq3_xxs_layout(n_blocks);
        return {base + l.qs_off, base + l.aux_off, base + l.d_off, blocks};
    }
    __device__ Raw load(const long long row, const long long s) const {
        Raw r;
        r.q = *reinterpret_cast<const uint2 *>(qs + row * blocks * 64 + s * 8);
        r.a = *reinterpret_cast<const uint32_t *>(aux + row * blocks * 32 + s * 4);
        r.dv = load_half(d + row * blocks * 2 + (s / 8) * 2);
        return r;
    }
    __device__ void decode(const Raw & r, const long long, float (&v)[32]) const {
        const float db = (r.dv * (0.5f + (float) (r.a >> 28))) * 0.5f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t word = (l < 2 ? r.q.x : r.q.y) >> (16 * (l & 1));
            const uint32_t b1 = omph::quant::kIq3xxsGrid[word & 0xFFu];
            const uint32_t b2 = omph::quant::kIq3xxsGrid[(word >> 8) & 0xFFu];
            const uint32_t sgn = iq3xxs_group_signs(r.a, l);
            put8(v, l, db, (sgn & kSignBits) | grid_pair_to_half2<0x04010400u>(b1),
                 ((sgn << 1) & kSignBits) | grid_pair_to_half2<0x04030402u>(b1),
                 ((sgn << 2) & kSignBits) | grid_pair_to_half2<0x04010400u>(b2),
                 ((sgn << 3) & kSignBits) | grid_pair_to_half2<0x04030402u>(b2));
        }
    }
};

struct DecIq3xxsTile {  // GGUF type 18, .omph layout 2 (#219)
    const uint8_t * base;
    long long blocks;  // 256-weight blocks per row
    using Raw = DecIq3xxs::Raw;
    static DecIq3xxsTile make(const uint8_t * base, const int64_t /*n_blocks*/, const long long blocks) {
        return {base, blocks};
    }
    __device__ Raw load(const long long row, const long long s) const {
        const uint8_t * o = base + ((row >> 4) * blocks + (s >> 3)) * omph::format::kIq3xxsTileBytes;
        const int r = (int) (row & 15);
        const int sb = (int) (s & 7);
        Raw w;
        w.q.x = *reinterpret_cast<const uint32_t *>(o + r * 32 + sb * 4);         // half 0: index bytes 0-3
        w.q.y = *reinterpret_cast<const uint32_t *>(o + (r + 16) * 32 + sb * 4);  // half 1: bytes 4-7
        w.a = *reinterpret_cast<const uint32_t *>(o + 1024 + r * 32 + sb * 4);
        w.dv = load_half(o + 1536 + r * 2);
        return w;
    }
    __device__ void decode(const Raw & r, const long long s, float (&v)[32]) const {
        DecIq3xxs{}.decode(r, s, v);
    }
};

struct DecIq2xxs {  // GGUF type 16
    const uint8_t * qs;
    const uint8_t * d;
    long long blocks;
    struct Raw {
        uint32_t aux0;
        uint32_t aux1;
        float dv;
    };
    static DecIq2xxs make(const uint8_t * base, const int64_t n_blocks, const long long blocks) {
        const auto l = omph::format::iq2_xxs_layout(n_blocks);
        return {base + l.qs_off, base + l.d_off, blocks};
    }
    __device__ Raw load(const long long row, const long long s) const {
        Raw r;
        const uint2 w = *reinterpret_cast<const uint2 *>(qs + row * blocks * 64 + s * 8);
        r.aux0 = w.x;
        r.aux1 = w.y;
        r.dv = load_half(d + row * blocks * 2 + (s / 8) * 2);
        return r;
    }
    __device__ void decode(const Raw & r, const long long, float (&v)[32]) const {
        const float db = (r.dv * (0.5f + (float) (r.aux1 >> 28))) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint64_t g = omph::quant::kIq2xxsGrid[(r.aux0 >> (8 * l)) & 0xFFu];
            const uint32_t g1 = (uint32_t) g;
            const uint32_t g2 = (uint32_t) (g >> 32);
            const uint32_t sgn = iq3xxs_group_signs(r.aux1, l);
            put8(v, l, db, (sgn & kSignBits) | grid_pair_to_half2<0x04010400u>(g1),
                 ((sgn << 1) & kSignBits) | grid_pair_to_half2<0x04030402u>(g1),
                 ((sgn << 2) & kSignBits) | grid_pair_to_half2<0x04010400u>(g2),
                 ((sgn << 3) & kSignBits) | grid_pair_to_half2<0x04030402u>(g2));
        }
    }
};

struct DecIq2xs {  // GGUF type 17
    const uint8_t * qs;
    const uint8_t * sc;
    const uint8_t * d;
    long long blocks;
    struct Raw {
        uint32_t ent[2];
        uint32_t scb;
        float dv;
    };
    static DecIq2xs make(const uint8_t * base, const int64_t n_blocks, const long long blocks) {
        const auto l = omph::format::iq2_xs_layout(n_blocks);
        return {base + l.qs_off, base + l.sc_off, base + l.d_off, blocks};
    }
    __device__ Raw load(const long long row, const long long s) const {
        Raw r;
        const long long blk = s / 8;
        const uint2 e = *reinterpret_cast<const uint2 *>(qs + row * blocks * 64 + s * 8);
        r.ent[0] = e.x;
        r.ent[1] = e.y;
        r.scb = sc[row * blocks * 8 + blk * 8 + (s % 8)];
        r.dv = load_half(d + row * blocks * 2 + blk * 2);
        return r;
    }
    __device__ void decode(const Raw & r, const long long, float (&v)[32]) const {
        const float dbl[2] = {(r.dv * (0.5f + (float) (r.scb & 0xF))) * 0.25f,
                              (r.dv * (0.5f + (float) (r.scb >> 4))) * 0.25f};
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t en = (r.ent[l >> 1] >> (16 * (l & 1))) & 0xFFFFu;
            const uint64_t g = omph::quant::kIq2xsGrid[en & 511u];
            const uint32_t g1 = (uint32_t) g;
            const uint32_t g2 = (uint32_t) (g >> 32);
            const uint32_t f = en >> 9;
            const uint32_t par = (uint32_t) __builtin_popcount(f) & 1u;
            const uint32_t sgn = ((f & 0xFu) << 12) | ((f >> 4) << 29) | (par << 28);
            put8(v, l, dbl[l >> 1], (sgn & kSignBits) | grid_pair_to_half2<0x04010400u>(g1),
                 ((sgn << 1) & kSignBits) | grid_pair_to_half2<0x04030402u>(g1),
                 ((sgn << 2) & kSignBits) | grid_pair_to_half2<0x04010400u>(g2),
                 ((sgn << 3) & kSignBits) | grid_pair_to_half2<0x04030402u>(g2));
        }
    }
};

struct DecIq2s {  // GGUF type 22
    const uint8_t * qs;
    const uint8_t * sg;
    const uint8_t * qh;
    const uint8_t * sc;
    const uint8_t * d;
    long long blocks;
    struct Raw {
        uint32_t qsw;
        uint32_t sgw;
        uint32_t qhb;
        uint32_t scb;
        float dv;
    };
    static DecIq2s make(const uint8_t * base, const int64_t n_blocks, const long long blocks) {
        const auto l = omph::format::iq2s_layout(n_blocks);
        return {base + l.qs_off, base + l.signs_off, base + l.qh_off, base + l.sc_off,
                base + l.d_off, blocks};
    }
    __device__ Raw load(const long long row, const long long s) const {
        Raw r;
        const long long blk = s / 8;
        r.qsw = *reinterpret_cast<const uint32_t *>(qs + row * blocks * 32 + s * 4);
        r.sgw = *reinterpret_cast<const uint32_t *>(sg + row * blocks * 32 + s * 4);
        r.qhb = qh[row * blocks * 8 + s];
        r.scb = sc[row * blocks * 8 + blk * 8 + (s % 8)];
        r.dv = load_half(d + row * blocks * 2 + blk * 2);
        return r;
    }
    __device__ void decode(const Raw & r, const long long, float (&v)[32]) const {
        const float dbl[2] = {(r.dv * (0.5f + (float) (r.scb & 0xF))) * 0.25f,
                              (r.dv * (0.5f + (float) (r.scb >> 4))) * 0.25f};
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t lo = (r.qsw >> (8 * l)) & 0xFFu;
            const uint32_t hi = ((r.qhb >> (2 * l)) & 3u) << 8;
            const uint64_t g = omph::quant::kIq2sGrid[lo | hi];
            const uint32_t g1 = (uint32_t) g;
            const uint32_t g2 = (uint32_t) (g >> 32);
            put8(v, l, dbl[l >> 1],
                 ((r.sgw << (4 * l)) & kSignBits) | grid_pair_to_half2<0x04010400u>(g1),
                 ((r.sgw << (4 * l + 1)) & kSignBits) | grid_pair_to_half2<0x04030402u>(g1),
                 ((r.sgw << (4 * l + 2)) & kSignBits) | grid_pair_to_half2<0x04010400u>(g2),
                 ((r.sgw << (4 * l + 3)) & kSignBits) | grid_pair_to_half2<0x04030402u>(g2));
        }
    }
};

struct DecIq4xs {  // GGUF type 23
    const uint8_t * qs;
    const uint8_t * meta;
    long long blocks;
    struct Raw {
        uint4 q;
        uint2 mt;
    };
    static DecIq4xs make(const uint8_t * base, const int64_t n_blocks, const long long blocks) {
        const auto l = omph::format::iq4_layout(n_blocks);
        return {base + l.qs_off, base + l.meta_off, blocks};
    }
    __device__ Raw load(const long long row, const long long s) const {
        Raw r;
        r.q = *reinterpret_cast<const uint4 *>(qs + row * blocks * 128 + s * 16);
        r.mt = *reinterpret_cast<const uint2 *>(meta + (row * blocks + s / 8) * 8);
        return r;
    }
    __device__ void decode(const Raw & r, const long long s, float (&v)[32]) const {
        const float dl = iq4_meta(r.mt, (int) (s % 8));
        const uint32_t w[4] = {r.q.x, r.q.y, r.q.z, r.q.w};
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint32_t byte = (w[i / 4] >> (8 * (i % 4))) & 0xFFu;
            v[i] = dl * (float) omph::quant::kKvaluesIq4nl[byte & 0xFu];
            v[16 + i] = dl * (float) omph::quant::kKvaluesIq4nl[byte >> 4];
        }
    }
};

struct DecQ4k {  // GGUF type 12
    const uint8_t * qs;
    const uint8_t * meta;
    long long blocks;
    struct Raw {
        uint4 q;
        uint4 mt;
    };
    static DecQ4k make(const uint8_t * base, const int64_t n_blocks, const long long blocks) {
        const auto l = omph::format::q4k_layout(n_blocks);
        return {base + l.qs_off, base + l.meta_off, blocks};
    }
    __device__ Raw load(const long long row, const long long s) const {
        Raw r;
        r.q = *reinterpret_cast<const uint4 *>(qs + row * blocks * 128 + s * 16);
        r.mt = *reinterpret_cast<const uint4 *>(meta + (row * blocks + s / 8) * 16);
        return r;
    }
    __device__ void decode(const Raw & r, const long long s, float (&v)[32]) const {
        float dl;
        float ml;
        q4k_meta(r.mt, (int) (s % 8), dl, ml);
        const uint32_t w[4] = {r.q.x, r.q.y, r.q.z, r.q.w};
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint32_t byte = (w[i / 4] >> (8 * (i % 4))) & 0xFFu;
            v[2 * i] = dl * (float) (byte & 0xFu) - ml;
            v[2 * i + 1] = dl * (float) (byte >> 4) - ml;
        }
    }
};

struct DecQ2k {  // GGUF type 10
    const uint8_t * sc;
    const uint8_t * qs;
    const uint8_t * d;
    const uint8_t * dmin;
    long long blocks;
    struct Raw {
        uint2 scw;
        float dv;
        float mv;
        uint4 qa;
        uint4 qb;
    };
    static DecQ2k make(const uint8_t * base, const int64_t n_blocks, const long long blocks) {
        const auto l = omph::format::q2k_layout(n_blocks);
        return {base + l.sc_off, base + l.qs_off, base + l.d_off, base + l.dmin_off, blocks};
    }
    __device__ Raw load(const long long row, const long long s) const {
        Raw r;
        const long long g = s / 4;  // the 128-weight group
        const long long blk = g >> 1;
        const int half = (int) (g & 1);
        const uint8_t * qrow = qs + row * blocks * 64 + g * 32;
        r.scw = *reinterpret_cast<const uint2 *>(sc + row * blocks * 16 + blk * 16 + half * 8);
        r.dv = load_half(d + row * blocks * 2 + blk * 2);
        r.mv = load_half(dmin + row * blocks * 2 + blk * 2);
        r.qa = *reinterpret_cast<const uint4 *>(qrow);
        r.qb = *reinterpret_cast<const uint4 *>(qrow + 16);
        return r;
    }
    __device__ void decode(const Raw & r, const long long s, float (&v)[32]) const {
        const uint64_t scw = (uint64_t) r.scw.x | ((uint64_t) r.scw.y << 32);
        const uint32_t wa[4] = {r.qa.x, r.qa.y, r.qa.z, r.qa.w};
        const uint32_t wb[4] = {r.qb.x, r.qb.y, r.qb.z, r.qb.w};
#pragma unroll
        for (int uu = 0; uu < 2; ++uu) {
            const int u = (int) (s % 4) * 2 + uu;  // 16-weight unit of the group
            const int shift = (u >> 1) * 2;
            const uint32_t scb = (uint32_t) (scw >> (8 * u)) & 0xFFu;
            const float dl = r.dv * (float) (scb & 15);
            const float ml = r.mv * (float) (scb >> 4);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint32_t word = (u & 1) ? wb[q] : wa[q];
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    v[uu * 16 + q * 4 + b] = dl * (float) ((word >> (8 * b + shift)) & 3u) - ml;
                }
            }
        }
    }
};

struct DecQ6k {  // GGUF type 14
    const uint8_t * ql;
    const uint8_t * qh;
    const uint8_t * sc;
    const uint8_t * d;
    long long blocks;
    struct Raw {
        uint32_t lo[8];
        uint32_t hi[8];
        uint32_t hb[8];
        float s0;
        float s1;
        float dv;
    };
    static DecQ6k make(const uint8_t * base, const int64_t n_blocks, const long long blocks) {
        const auto l = omph::format::q6k_layout(n_blocks);
        return {base + l.ql_off, base + l.qh_off, base + l.sc_off, base + l.d_off, blocks};
    }
    __device__ Raw load(const long long row, const long long s) const {
        Raw r;
        const long long g = s / 4;   // the 128-weight half-block
        const int j = (int) (s % 4); // this sub-block's quarter of it
        const long long blk = g >> 1;
        const int half = (int) (g & 1);
        const uint32_t * pql =
            reinterpret_cast<const uint32_t *>(ql + row * blocks * 128 + blk * 128 + half * 64);
        const uint32_t * pqh =
            reinterpret_cast<const uint32_t *>(qh + row * blocks * 64 + blk * 64 + half * 32);
        const int8_t * psc =
            reinterpret_cast<const int8_t *>(sc + row * blocks * 16 + blk * 16 + half * 8);
#pragma unroll
        for (int wi = 0; wi < 8; ++wi) {
            r.lo[wi] = pql[wi];
            r.hi[wi] = pql[wi + 8];
            r.hb[wi] = pqh[wi];
        }
        r.s0 = (float) psc[2 * j];
        r.s1 = (float) psc[2 * j + 1];
        r.dv = load_half(d + row * blocks * 2 + blk * 2);
        return r;
    }
    __device__ void decode(const Raw & r, const long long s, float (&v)[32]) const {
        const int j = (int) (s % 4);
#pragma unroll
        for (int wi = 0; wi < 8; ++wi) {
            const uint32_t lo = r.lo[wi];
            const uint32_t hi = r.hi[wi];
            const uint32_t hb = r.hb[wi];
            const uint32_t c = j == 0   ? (lo & 0x0F0F0F0Fu) | ((hb & 0x03030303u) << 4)
                               : j == 1 ? (hi & 0x0F0F0F0Fu) | ((hb & 0x0C0C0C0Cu) << 2)
                               : j == 2 ? ((lo >> 4) & 0x0F0F0F0Fu) | (hb & 0x30303030u)
                                        : ((hi >> 4) & 0x0F0F0F0Fu) | ((hb >> 2) & 0x30303030u);
            const float sv = (wi >> 2) ? r.s1 : r.s0;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                v[wi * 4 + b] = r.dv * sv * (float) ((int) ((c >> (8 * b)) & 0xFFu) - 32);
            }
        }
    }
};

struct DecIq1m {  // GGUF type 29: the "repacked" layout is the GGUF blocks (56 bytes)
    const uint8_t * base;
    long long blocks;
    struct Raw {
        uint32_t qs4;  // qs[4 ib .. 4 ib + 3]
        uint32_t qh2;  // qh[2 ib], qh[2 ib + 1]
        uint2 sc;      // the 8 scale bytes (sc16[0..3])
    };
    static DecIq1m make(const uint8_t * b, const int64_t, const long long blocks) {
        return {b, blocks};
    }
    __device__ Raw load(const long long row, const long long s) const {
        const uint8_t * blk = base + (row * blocks + s / 8) * 56;
        const int ib = (int) (s % 8);
        Raw r;
        r.qs4 = *reinterpret_cast<const uint32_t *>(blk + 4 * ib);
        r.qh2 = *reinterpret_cast<const uint16_t *>(blk + 32 + 2 * ib);
        r.sc = *reinterpret_cast<const uint2 *>(blk + 48);
        return r;
    }
    // dequant_iq1_m_kernel's expressions, per weight, in its order
    __device__ void decode(const Raw & r, const long long s, float (&v)[32]) const {
        const int ib = (int) (s % 8);
        const uint16_t sc16[4] = {(uint16_t) (r.sc.x & 0xFFFFu), (uint16_t) (r.sc.x >> 16),
                                  (uint16_t) (r.sc.y & 0xFFFFu), (uint16_t) (r.sc.y >> 16)};
        const uint16_t scale_u16 = (uint16_t) ((sc16[0] >> 12) | ((sc16[1] >> 8) & 0x00F0) |
                                               ((sc16[2] >> 4) & 0x0F00) | (sc16[3] & 0xF000));
        const float d = __half2float(__ushort_as_half(scale_u16));
        const uint8_t qh_a = (uint8_t) (r.qh2 & 0xFFu);
        const uint8_t qh_b = (uint8_t) (r.qh2 >> 8);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int sh = 6 * (ib % 2) + 3 * (l >> 1);
            const float dl = d * (2.0f * (float) ((sc16[ib >> 1] >> sh) & 7) + 1.0f);
            const int shift = ((l & 1) == 0) ? 8 : 4;
            const uint8_t qh_sel = ((l >> 1) == 0) ? qh_a : qh_b;
            const uint16_t idx =
                (uint16_t) (((r.qs4 >> (8 * l)) & 0xFFu) | ((qh_sel << shift) & 0x700));
            const bool neg = l == 0   ? (qh_a & 0x08) != 0
                             : l == 1 ? (qh_a & 0x80) != 0
                             : l == 2 ? (qh_b & 0x08) != 0
                                      : (qh_b & 0x80) != 0;
            const float delta = neg ? -0.125f : 0.125f;
            const uint64_t grid = omph::quant::kIq1sGrid[idx];
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const int8_t g = (int8_t) ((grid >> (8 * jj)) & 0xFF);
                v[l * 8 + jj] = dl * ((float) g + delta);
            }
        }
    }
};

// 32 weights as 16 packed half2 words (RNE, as __float2half), in order.
__device__ inline void pack32(const float (&v)[32], uint4 (&o)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __half lo = __float2half(v[i * 8 + 2 * j]);
            const __half hi = __float2half(v[i * 8 + 2 * j + 1]);
            w[j] = (uint32_t) __half_as_ushort(lo) | ((uint32_t) __half_as_ushort(hi) << 16);
        }
        o[i] = make_uint4(w[0], w[1], w[2], w[3]);
    }
}

// --- WMMA tile decoders (#178, #219) ----------------------------------------
//
// One per tile layout (format/repack.hh), shared by the tile GEMVs (gemv.hip)
// and the tile GEMM (gemm.hip). A wave's lane l (row l % 16, half h = l / 16)
// loads its words of a 256-weight block (load), then builds its WMMA A operand
// for sub-block sb and group half gl: weights 16 h + 8 gl .. + 7 as f16 (the
// sub-block scale times the grid value, in half2 arithmetic; frag); the
// activations take the same k order. init fills the kLds bytes of LDS the
// decode reads (the grid as half2 pairs or words).
typedef _Float16 tile_h8 __attribute__((ext_vector_type(8)));

struct TileIq3s {  // GGUF type 21
    static constexpr long long kBytes = omph::format::kIq3sTileBytes;
    static constexpr int kLds = 4096;
    __device__ static void init(uint8_t * lds) {
        uint2 * grid = reinterpret_cast<uint2 *>(lds);
        for (int i = (int) threadIdx.x; i < 512; i += blockDim.x) {
            const uint32_t g = omph::quant::kIq3sGrid[i];
            grid[i] = make_uint2(grid_pair_to_half2<0x04010400u>(g), grid_pair_to_half2<0x04030402u>(g));
        }
    }
    struct Blk {
        uint4 q0, q1, s4;
        uint32_t qhw, scw;
        float dv;
    };
    __device__ static Blk load(const uint8_t * o, const int lane) {
        const int r = lane & 15;
        Blk k;
        k.q0 = *reinterpret_cast<const uint4 *>(o + lane * 32);
        k.q1 = *reinterpret_cast<const uint4 *>(o + lane * 32 + 16);
        k.s4 = *reinterpret_cast<const uint4 *>(o + 1024 + lane * 16);
        k.qhw = *reinterpret_cast<const uint32_t *>(o + 1536 + lane * 4);
        k.scw = *reinterpret_cast<const uint32_t *>(o + 1664 + r * 4);
        k.dv = __half2float(*reinterpret_cast<const __half *>(o + 1728 + r * 2));
        return k;
    }
    __device__ static tile_h8 frag(const uint8_t * lds, const Blk & k, const int sb, const int gl, const int h) {
        const uint2 * grid = reinterpret_cast<const uint2 *>(lds);
        const uint32_t qw[8] = {k.q0.x, k.q0.y, k.q0.z, k.q0.w, k.q1.x, k.q1.y, k.q1.z, k.q1.w};
        const uint32_t sw[4] = {k.s4.x, k.s4.y, k.s4.z, k.s4.w};
        const uint32_t q = qw[sb];
        const uint32_t qhn = (k.qhw >> (4 * sb)) & 0xFu;
        const uint32_t s16 = (sw[sb >> 1] >> (16 * (sb & 1))) & 0xFFFFu;
        // the pair-ordered sign word with this half's bytes back in place
        const uint32_t sg = h == 0 ? (((s16 & 0xFFu) << 8) | ((s16 >> 8) << 24)) : ((s16 & 0xFFu) | ((s16 >> 8) << 16));
        const uint32_t sc = (k.scw >> (8 * (sb >> 1))) & 0xFFu;
        const float dbr = k.dv * (1.0f + 2.0f * (float) ((sb & 1) ? (sc >> 4) : (sc & 0xF)));
        const half2_v d2 = {(_Float16) dbr, (_Float16) dbr};
        const int g = 2 * h + gl;  // the 8-weight group of the sub-block
        const uint32_t word = (q >> (16 * gl)) & 0xFFFFu;
        const uint32_t i1 = (word & 0xFFu) | (((qhn >> (2 * gl)) & 1u) << 8);
        const uint32_t i2 = (word >> 8) | (((qhn >> (2 * gl + 1)) & 1u) << 8);
        const uint2 g1 = grid[i1];
        const uint2 g2 = grid[i2];
        constexpr uint32_t kSign2 = 0x80008000u;
        const uint32_t wv4[4] = {((sg << (4 * g)) & kSign2) | g1.x, ((sg << (4 * g + 1)) & kSign2) | g1.y,
                                 ((sg << (4 * g + 2)) & kSign2) | g2.x, ((sg << (4 * g + 3)) & kSign2) | g2.y};
        tile_h8 a;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const half2_v p = __builtin_bit_cast(half2_v, wv4[i]) * d2;
            a[2 * i] = p[0];
            a[2 * i + 1] = p[1];
        }
        return a;
    }
};

struct TileIq3xxs {  // GGUF type 18 (#219)
    static constexpr long long kBytes = omph::format::kIq3xxsTileBytes;
    static constexpr int kLds = 1024;
    __device__ static void init(uint8_t * lds) {
        uint32_t * grid = reinterpret_cast<uint32_t *>(lds);
        for (int i = (int) threadIdx.x; i < 256; i += blockDim.x) {
            grid[i] = omph::quant::kIq3xxsGrid[i];
        }
    }
    struct Blk {
        uint4 q0, q1, a0, a1;
        float dv;
    };
    __device__ static Blk load(const uint8_t * o, const int lane) {
        const int r = lane & 15;
        Blk k;
        k.q0 = *reinterpret_cast<const uint4 *>(o + lane * 32);
        k.q1 = *reinterpret_cast<const uint4 *>(o + lane * 32 + 16);
        k.a0 = *reinterpret_cast<const uint4 *>(o + 1024 + r * 32);
        k.a1 = *reinterpret_cast<const uint4 *>(o + 1024 + r * 32 + 16);
        k.dv = __half2float(*reinterpret_cast<const __half *>(o + 1536 + r * 2));
        return k;
    }
    __device__ static tile_h8 frag(const uint8_t * lds, const Blk & k, const int sb, const int gl, const int h) {
        const uint32_t * grid = reinterpret_cast<const uint32_t *>(lds);
        const uint32_t qw[8] = {k.q0.x, k.q0.y, k.q0.z, k.q0.w, k.q1.x, k.q1.y, k.q1.z, k.q1.w};
        const uint32_t aw[8] = {k.a0.x, k.a0.y, k.a0.z, k.a0.w, k.a1.x, k.a1.y, k.a1.z, k.a1.w};
        const uint32_t a = aw[sb];
        const float db = (k.dv * (0.5f + (float) (a >> 28))) * 0.5f;
        const half2_v d2 = {(_Float16) db, (_Float16) db};
        const uint32_t word = (qw[sb] >> (16 * gl)) & 0xFFFFu;  // index bytes 4 h + 2 gl, + 1
        const uint32_t b1 = grid[word & 0xFFu];
        const uint32_t b2 = grid[word >> 8];
        const uint32_t sgn = iq3xxs_group_signs(a, 2 * h + gl);
        constexpr uint32_t kSign2 = 0x80008000u;
        const uint32_t wv4[4] = {(sgn & kSign2) | grid_pair_to_half2<0x04010400u>(b1),
                                 ((sgn << 1) & kSign2) | grid_pair_to_half2<0x04030402u>(b1),
                                 ((sgn << 2) & kSign2) | grid_pair_to_half2<0x04010400u>(b2),
                                 ((sgn << 3) & kSign2) | grid_pair_to_half2<0x04030402u>(b2)};
        tile_h8 v;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const half2_v p = __builtin_bit_cast(half2_v, wv4[i]) * d2;
            v[2 * i] = p[0];
            v[2 * i + 1] = p[1];
        }
        return v;
    }
};

// IQ4_XS tiles (#244): lane words = the codes of its half, two per byte (even
// weight low); the frag looks the codes up (iq4_lookup), pairs (even, odd) with
// two byte permutes and scales by f16(d (scale - 32)).
struct TileIq4xs {  // GGUF type 23
    static constexpr long long kBytes = omph::format::kIq4xsTileBytes;
    static constexpr int kLds = 16;
    __device__ static void init(uint8_t *) {}
    struct Blk {
        uint4 q0, q1, q2, q3;
        uint2 mt;
    };
    __device__ static Blk load(const uint8_t * o, const int lane) {
        Blk k;
        const uint4 * q = reinterpret_cast<const uint4 *>(o + lane * 64);
        k.q0 = q[0];
        k.q1 = q[1];
        k.q2 = q[2];
        k.q3 = q[3];
        k.mt = *reinterpret_cast<const uint2 *>(o + 2048 + (lane & 15) * 8);
        return k;
    }
    __device__ static tile_h8 frag(const uint8_t *, const Blk & k, const int sb, const int gl, const int) {
        const uint32_t qw[16] = {k.q0.x, k.q0.y, k.q0.z, k.q0.w, k.q1.x, k.q1.y, k.q1.z, k.q1.w,
                                 k.q2.x, k.q2.y, k.q2.z, k.q2.w, k.q3.x, k.q3.y, k.q3.z, k.q3.w};
        const uint32_t w = qw[2 * sb + gl];
        const uint32_t ve = iq4_lookup(w & 0x0F0F0F0Fu);         // weights 0, 2, 4, 6 (+ 128)
        const uint32_t vo = iq4_lookup((w >> 4) & 0x0F0F0F0Fu);  // 1, 3, 5, 7
        const uint32_t p01 = __builtin_amdgcn_perm(vo, ve, 0x05010400u);  // e0 o0 e1 o1
        const uint32_t p23 = __builtin_amdgcn_perm(vo, ve, 0x07030602u);  // e2 o2 e3 o3
        const float dl = iq4_meta(k.mt, sb);
        const half2_v d2 = {(_Float16) dl, (_Float16) dl};
        const uint32_t hv[4] = {iq4_pair_to_half2<0x04010400u>(p01), iq4_pair_to_half2<0x04030402u>(p01),
                                iq4_pair_to_half2<0x04010400u>(p23), iq4_pair_to_half2<0x04030402u>(p23)};
        tile_h8 a;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const half2_v p = __builtin_bit_cast(half2_v, hv[i]) * d2;
            a[2 * i] = p[0];
            a[2 * i + 1] = p[1];
        }
        return a;
    }
};

// Q4_K tiles (#244): lane words = its half's repacked bytes; each weight
// d sc q - dmin m in f32, rounded to f16 once (the GEMM's dequant values).
struct TileQ4k {  // GGUF type 12
    static constexpr long long kBytes = omph::format::kQ4kTileBytes;
    static constexpr int kLds = 16;
    __device__ static void init(uint8_t *) {}
    struct Blk {
        uint4 q0, q1, q2, q3;
        uint4 mt;
    };
    __device__ static Blk load(const uint8_t * o, const int lane) {
        Blk k;
        const uint4 * q = reinterpret_cast<const uint4 *>(o + lane * 64);
        k.q0 = q[0];
        k.q1 = q[1];
        k.q2 = q[2];
        k.q3 = q[3];
        k.mt = *reinterpret_cast<const uint4 *>(o + 2048 + (lane & 15) * 16);
        return k;
    }
    __device__ static tile_h8 frag(const uint8_t *, const Blk & k, const int sb, const int gl, const int) {
        // the word by selects: an indexed 16-word array went to scratch in the GEMM's sb loop
        const int idx = 2 * sb + gl;
        const uint4 v = idx < 4 ? k.q0 : idx < 8 ? k.q1 : idx < 12 ? k.q2 : k.q3;
        const int e = idx & 3;
        const uint32_t w = e == 0 ? v.x : e == 1 ? v.y : e == 2 ? v.z : v.w;
        float ds;
        float ms;
        q4k_meta(k.mt, sb, ds, ms);
        tile_h8 a;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t byte = (w >> (8 * m)) & 0xFFu;
            a[2 * m] = (_Float16) (ds * (float) (byte & 0xFu) - ms);
            a[2 * m + 1] = (_Float16) (ds * (float) (byte >> 4) - ms);
        }
        return a;
    }
};

// The row-wise decoders over these tiles (the GEMM for rows % 128 != 0, the f16
// dequant): DecIq4xs's / DecQ4k's Raw rebuilt from the row's two lanes.
struct DecIq4xsTile {  // GGUF type 23, .omph layout 2 (#244)
    const uint8_t * base;
    long long blocks;
    using Raw = DecIq4xs::Raw;
    static DecIq4xsTile make(const uint8_t * base, const int64_t /*n_blocks*/, const long long blocks) {
        return {base, blocks};
    }
    __device__ Raw load(const long long row, const long long s) const {
        const uint8_t * o = base + ((row >> 4) * blocks + (s >> 3)) * omph::format::kIq4xsTileBytes;
        const int r = (int) (row & 15);
        const int sb = (int) (s & 7);
        const uint2 h0 = *reinterpret_cast<const uint2 *>(o + r * 64 + sb * 8);
        const uint2 h1 = *reinterpret_cast<const uint2 *>(o + (r + 16) * 64 + sb * 8);
        const uint32_t a[2] = {h0.x, h0.y};
        const uint32_t b[2] = {h1.x, h1.y};
        uint32_t q[4] = {0, 0, 0, 0};
#pragma unroll
        for (int j = 0; j < 16; ++j) {  // byte j: weight j (low), j + 16 (high)
            const uint32_t lo = (a[j / 8] >> (4 * (j % 8))) & 0xFu;
            const uint32_t hi = (b[j / 8] >> (4 * (j % 8))) & 0xFu;
            q[j / 4] |= (lo | (hi << 4)) << (8 * (j % 4));
        }
        Raw w;
        w.q = make_uint4(q[0], q[1], q[2], q[3]);
        w.mt = *reinterpret_cast<const uint2 *>(o + 2048 + r * 8);
        return w;
    }
    __device__ void decode(const Raw & r, const long long s, float (&v)[32]) const { DecIq4xs{}.decode(r, s, v); }
};

struct DecQ4kTile {  // GGUF type 12, .omph layout 2 (#244)
    const uint8_t * base;
    long long blocks;
    using Raw = DecQ4k::Raw;
    static DecQ4kTile make(const uint8_t * base, const int64_t /*n_blocks*/, const long long blocks) {
        return {base, blocks};
    }
    __device__ Raw load(const long long row, const long long s) const {
        const uint8_t * o = base + ((row >> 4) * blocks + (s >> 3)) * omph::format::kQ4kTileBytes;
        const int r = (int) (row & 15);
        const int sb = (int) (s & 7);
        const uint2 h0 = *reinterpret_cast<const uint2 *>(o + r * 64 + sb * 8);
        const uint2 h1 = *reinterpret_cast<const uint2 *>(o + (r + 16) * 64 + sb * 8);
        Raw w;
        w.q = make_uint4(h0.x, h0.y, h1.x, h1.y);
        w.mt = *reinterpret_cast<const uint4 *>(o + 2048 + r * 16);
        return w;
    }
    __device__ void decode(const Raw & r, const long long s, float (&v)[32]) const { DecQ4k{}.decode(r, s, v); }
};

// IQ2_S tiles (#253): the grid's 1024 eight-byte entries in LDS; a lane's
// group g = 2 h + gl takes its index byte (+ 2 qh bits), the pair-ordered
// sign bits of its half (as TileIq3s) and the scale nibble h, the block scale
// applied in f16.
struct TileIq2s {  // GGUF type 22
    static constexpr long long kBytes = omph::format::kIq2sTileBytes;
    static constexpr int kLds = 8192;
    __device__ static void init(uint8_t * lds) {
        uint2 * grid = reinterpret_cast<uint2 *>(lds);
        for (int i = (int) threadIdx.x; i < 1024; i += blockDim.x) {
            const uint64_t g = omph::quant::kIq2sGrid[i];
            grid[i] = make_uint2((uint32_t) g, (uint32_t) (g >> 32));
        }
    }
    struct Blk {
        uint4 q, s4;
        uint2 scw;
        uint32_t qhw;
        float dv;
    };
    __device__ static Blk load(const uint8_t * o, const int lane) {
        const int r = lane & 15;
        Blk k;
        k.q = *reinterpret_cast<const uint4 *>(o + lane * 16);
        k.s4 = *reinterpret_cast<const uint4 *>(o + 512 + lane * 16);
        k.qhw = *reinterpret_cast<const uint32_t *>(o + 1024 + lane * 4);
        k.scw = *reinterpret_cast<const uint2 *>(o + 1152 + r * 8);
        k.dv = __half2float(*reinterpret_cast<const __half *>(o + 1280 + r * 2));
        return k;
    }
    __device__ static tile_h8 frag(const uint8_t * lds, const Blk & k, const int sb, const int gl, const int h) {
        const uint2 * grid = reinterpret_cast<const uint2 *>(lds);
        const uint32_t qw[4] = {k.q.x, k.q.y, k.q.z, k.q.w};
        const uint32_t sw[4] = {k.s4.x, k.s4.y, k.s4.z, k.s4.w};
        const uint32_t lo = (qw[sb >> 1] >> (16 * (sb & 1) + 8 * gl)) & 0xFFu;
        const uint32_t hi = ((k.qhw >> (4 * sb + 2 * gl)) & 3u) << 8;
        const uint2 gr = grid[lo | hi];
        const uint32_t s16 = (sw[sb >> 1] >> (16 * (sb & 1))) & 0xFFFFu;
        const uint32_t sg = h == 0 ? (((s16 & 0xFFu) << 8) | ((s16 >> 8) << 24)) : ((s16 & 0xFFu) | ((s16 >> 8) << 16));
        const uint32_t sc = ((sb < 4 ? k.scw.x : k.scw.y) >> (8 * (sb & 3))) & 0xFFu;
        const float db = (k.dv * (0.5f + (float) (h ? (sc >> 4) : (sc & 0xF)))) * 0.25f;
        const half2_v d2 = {(_Float16) db, (_Float16) db};
        const int g = 2 * h + gl;
        constexpr uint32_t kSign2 = 0x80008000u;
        const uint32_t wv4[4] = {((sg << (4 * g)) & kSign2) | grid_pair_to_half2<0x04010400u>(gr.x),
                                 ((sg << (4 * g + 1)) & kSign2) | grid_pair_to_half2<0x04030402u>(gr.x),
                                 ((sg << (4 * g + 2)) & kSign2) | grid_pair_to_half2<0x04010400u>(gr.y),
                                 ((sg << (4 * g + 3)) & kSign2) | grid_pair_to_half2<0x04030402u>(gr.y)};
        tile_h8 a;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const half2_v p = __builtin_bit_cast(half2_v, wv4[i]) * d2;
            a[2 * i] = p[0];
            a[2 * i + 1] = p[1];
        }
        return a;
    }
};

struct DecIq2sTile {  // GGUF type 22, .omph layout 2 (#253)
    const uint8_t * base;
    long long blocks;
    using Raw = DecIq2s::Raw;
    static DecIq2sTile make(const uint8_t * base, const int64_t /*n_blocks*/, const long long blocks) {
        return {base, blocks};
    }
    __device__ Raw load(const long long row, const long long s) const {
        const uint8_t * o = base + ((row >> 4) * blocks + (s >> 3)) * omph::format::kIq2sTileBytes;
        const int r = (int) (row & 15);
        const int sb = (int) (s & 7);
        const uint32_t q0 = *reinterpret_cast<const uint16_t *>(o + r * 16 + sb * 2);
        const uint32_t q1 = *reinterpret_cast<const uint16_t *>(o + (r + 16) * 16 + sb * 2);
        const uint32_t s0 = *reinterpret_cast<const uint16_t *>(o + 512 + r * 16 + sb * 2);  // bytes 1, 3
        const uint32_t s1 = *reinterpret_cast<const uint16_t *>(o + 512 + (r + 16) * 16 + sb * 2);  // 0, 2
        const uint32_t h0 = *reinterpret_cast<const uint32_t *>(o + 1024 + r * 4);
        const uint32_t h1 = *reinterpret_cast<const uint32_t *>(o + 1024 + (r + 16) * 4);
        Raw w;
        w.qsw = q0 | (q1 << 16);
        w.sgw = (s1 & 0xFFu) | ((s0 & 0xFFu) << 8) | ((s1 >> 8) << 16) | ((s0 >> 8) << 24);
        w.qhb = ((h0 >> (4 * sb)) & 0xFu) | (((h1 >> (4 * sb)) & 0xFu) << 4);
        w.scb = o[1152 + r * 8 + sb];
        w.dv = __half2float(*reinterpret_cast<const __half *>(o + 1280 + r * 2));
        return w;
    }
    __device__ void decode(const Raw & r, const long long s, float (&v)[32]) const { DecIq2s{}.decode(r, s, v); }
};

// Q2_K tiles (#253): sub-block sb's 2-bit codes at bits 2 (sb % 4) of the
// lane's bytes of half sb / 4; each weight d sc q - dmin m in f32, rounded to
// f16 once (DecQ2k's values).
struct TileQ2k {  // GGUF type 10
    static constexpr long long kBytes = omph::format::kQ2kTileBytes;
    static constexpr int kLds = 16;
    __device__ static void init(uint8_t *) {}
    struct Blk {
        uint4 q0, q1;
        uint2 scw;
        float d, m;
    };
    __device__ static Blk load(const uint8_t * o, const int lane) {
        Blk k;
        const uint4 * q = reinterpret_cast<const uint4 *>(o + lane * 32);
        k.q0 = q[0];
        k.q1 = q[1];
        k.scw = *reinterpret_cast<const uint2 *>(o + 1024 + lane * 8);
        const uint32_t dm = *reinterpret_cast<const uint32_t *>(o + 1280 + (lane & 15) * 4);
        k.d = half_lo(dm);
        k.m = half_hi(dm);
        return k;
    }
    __device__ static tile_h8 frag(const uint8_t *, const Blk & k, const int sb, const int gl, const int) {
        const uint4 v = sb < 4 ? k.q0 : k.q1;
        const uint32_t w0 = gl == 0 ? v.x : v.z;
        const uint32_t w1 = gl == 0 ? v.y : v.w;
        const int shift = 2 * (sb & 3);
        const uint32_t c0 = (w0 >> shift) & 0x03030303u;
        const uint32_t c1 = (w1 >> shift) & 0x03030303u;
        const uint32_t sc = ((sb < 4 ? k.scw.x : k.scw.y) >> (8 * (sb & 3))) & 0xFFu;
        const float dl = k.d * (float) (sc & 15);
        const float ml = k.m * (float) (sc >> 4);
        tile_h8 a;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            a[m] = (_Float16) (dl * (float) ((c0 >> (8 * m)) & 0xFFu) - ml);
            a[4 + m] = (_Float16) (dl * (float) ((c1 >> (8 * m)) & 0xFFu) - ml);
        }
        return a;
    }
};

// IQ2_XS tiles (#253): entry = 9-bit grid index + 7 pair-ordered sign bits
// (the 8th the parity), scale nibble h; grid (512 x 8 bytes) in LDS.
struct TileIq2xs {  // GGUF type 17
    static constexpr long long kBytes = omph::format::kIq2xsTileBytes;
    static constexpr int kLds = 4096;
    __device__ static void init(uint8_t * lds) {
        uint2 * grid = reinterpret_cast<uint2 *>(lds);
        for (int i = (int) threadIdx.x; i < 512; i += blockDim.x) {
            const uint64_t g = omph::quant::kIq2xsGrid[i];
            grid[i] = make_uint2((uint32_t) g, (uint32_t) (g >> 32));
        }
    }
    struct Blk {
        uint4 q0, q1;
        uint2 scw;
        float dv;
    };
    __device__ static Blk load(const uint8_t * o, const int lane) {
        const int r = lane & 15;
        Blk k;
        const uint4 * q = reinterpret_cast<const uint4 *>(o + lane * 32);
        k.q0 = q[0];
        k.q1 = q[1];
        k.scw = *reinterpret_cast<const uint2 *>(o + 1024 + r * 8);
        k.dv = __half2float(*reinterpret_cast<const __half *>(o + 1152 + r * 2));
        return k;
    }
    __device__ static tile_h8 frag(const uint8_t * lds, const Blk & k, const int sb, const int gl, const int h) {
        const uint2 * grid = reinterpret_cast<const uint2 *>(lds);
        const uint4 v = sb < 4 ? k.q0 : k.q1;
        const int e = sb & 3;
        const uint32_t w = e == 0 ? v.x : e == 1 ? v.y : e == 2 ? v.z : v.w;
        const uint32_t en = (w >> (16 * gl)) & 0xFFFFu;
        const uint2 gr = grid[en & 511u];
        const uint32_t f = en >> 9;
        const uint32_t par = (uint32_t) __builtin_popcount(f) & 1u;
        const uint32_t sgn = ((f & 0xFu) << 12) | ((f >> 4) << 29) | (par << 28);
        const uint32_t sc = ((sb < 4 ? k.scw.x : k.scw.y) >> (8 * (sb & 3))) & 0xFFu;
        const float db = (k.dv * (0.5f + (float) (h ? (sc >> 4) : (sc & 0xF)))) * 0.25f;
        const half2_v d2 = {(_Float16) db, (_Float16) db};
        const uint32_t wv4[4] = {(sgn & kSignBits) | grid_pair_to_half2<0x04010400u>(gr.x),
                                 ((sgn << 1) & kSignBits) | grid_pair_to_half2<0x04030402u>(gr.x),
                                 ((sgn << 2) & kSignBits) | grid_pair_to_half2<0x04010400u>(gr.y),
                                 ((sgn << 3) & kSignBits) | grid_pair_to_half2<0x04030402u>(gr.y)};
        tile_h8 a;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const half2_v p = __builtin_bit_cast(half2_v, wv4[i]) * d2;
            a[2 * i] = p[0];
            a[2 * i + 1] = p[1];
        }
        return a;
    }
};

// IQ2_XXS tiles (#253): index byte per group, the row's scale + sign word per
// sub-block (IQ3_XXS's sign order); grid (256 x 8 bytes) in LDS.
struct TileIq2xxs {  // GGUF type 16
    static constexpr long long kBytes = omph::format::kIq2xxsTileBytes;
    static constexpr int kLds = 2048;
    __device__ static void init(uint8_t * lds) {
        uint2 * grid = reinterpret_cast<uint2 *>(lds);
        for (int i = (int) threadIdx.x; i < 256; i += blockDim.x) {
            const uint64_t g = omph::quant::kIq2xxsGrid[i];
            grid[i] = make_uint2((uint32_t) g, (uint32_t) (g >> 32));
        }
    }
    struct Blk {
        uint4 q, a0, a1;
        float dv;
    };
    __device__ static Blk load(const uint8_t * o, const int lane) {
        const int r = lane & 15;
        Blk k;
        k.q = *reinterpret_cast<const uint4 *>(o + lane * 16);
        k.a0 = *reinterpret_cast<const uint4 *>(o + 512 + r * 32);
        k.a1 = *reinterpret_cast<const uint4 *>(o + 512 + r * 32 + 16);
        k.dv = __half2float(*reinterpret_cast<const __half *>(o + 1024 + r * 2));
        return k;
    }
    __device__ static tile_h8 frag(const uint8_t * lds, const Blk & k, const int sb, const int gl, const int h) {
        const uint2 * grid = reinterpret_cast<const uint2 *>(lds);
        const uint32_t qw[4] = {k.q.x, k.q.y, k.q.z, k.q.w};
        const uint4 av = sb < 4 ? k.a0 : k.a1;
        const int e = sb & 3;
        const uint32_t a = e == 0 ? av.x : e == 1 ? av.y : e == 2 ? av.z : av.w;
        const uint32_t idx = (qw[sb >> 1] >> (16 * (sb & 1) + 8 * gl)) & 0xFFu;
        const uint2 gr = grid[idx];
        const float db = (k.dv * (0.5f + (float) (a >> 28))) * 0.25f;
        const half2_v d2 = {(_Float16) db, (_Float16) db};
        const uint32_t sgn = iq3xxs_group_signs(a, 2 * h + gl);
        const uint32_t wv4[4] = {(sgn & kSignBits) | grid_pair_to_half2<0x04010400u>(gr.x),
                                 ((sgn << 1) & kSignBits) | grid_pair_to_half2<0x04030402u>(gr.x),
                                 ((sgn << 2) & kSignBits) | grid_pair_to_half2<0x04010400u>(gr.y),
                                 ((sgn << 3) & kSignBits) | grid_pair_to_half2<0x04030402u>(gr.y)};
        tile_h8 v;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const half2_v p = __builtin_bit_cast(half2_v, wv4[i]) * d2;
            v[2 * i] = p[0];
            v[2 * i + 1] = p[1];
        }
        return v;
    }
};

// Row-wise decoders over the Q2_K / IQ2_XS / IQ2_XXS tiles (the GEMM for rows
// % 128 != 0, the f16 dequant): the row-wise Raw rebuilt from the row's lanes.
struct DecQ2kTile {  // GGUF type 10, .omph layout 2 (#253)
    const uint8_t * base;
    long long blocks;
    using Raw = DecQ2k::Raw;
    static DecQ2kTile make(const uint8_t * base, const int64_t, const long long blocks) { return {base, blocks}; }
    __device__ Raw load(const long long row, const long long s) const {
        const uint8_t * o = base + ((row >> 4) * blocks + (s >> 3)) * omph::format::kQ2kTileBytes;
        const int r = (int) (row & 15);
        const int g = (int) ((s & 7) >> 2);  // the 128-weight half
        Raw w;
        w.qa = *reinterpret_cast<const uint4 *>(o + r * 32 + g * 16);         // bytes 0..15
        w.qb = *reinterpret_cast<const uint4 *>(o + (r + 16) * 32 + g * 16);  // 16..31
        const uint32_t s0 = *reinterpret_cast<const uint32_t *>(o + 1024 + r * 8 + g * 4);
        const uint32_t s1 = *reinterpret_cast<const uint32_t *>(o + 1024 + (r + 16) * 8 + g * 4);
        // scale bytes 8 g + 2 j (+ 1 for half 1): interleave the two lanes' bytes
        w.scw.x = __builtin_amdgcn_perm(s1, s0, 0x05010400u);
        w.scw.y = __builtin_amdgcn_perm(s1, s0, 0x07030602u);
        const uint32_t dm = *reinterpret_cast<const uint32_t *>(o + 1280 + r * 4);
        w.dv = half_lo(dm);
        w.mv = half_hi(dm);
        return w;
    }
    __device__ void decode(const Raw & r, const long long s, float (&v)[32]) const { DecQ2k{}.decode(r, s, v); }
};

struct DecIq2xsTile {  // GGUF type 17, .omph layout 2 (#253)
    const uint8_t * base;
    long long blocks;
    using Raw = DecIq2xs::Raw;
    static DecIq2xsTile make(const uint8_t * base, const int64_t, const long long blocks) { return {base, blocks}; }
    __device__ Raw load(const long long row, const long long s) const {
        const uint8_t * o = base + ((row >> 4) * blocks + (s >> 3)) * omph::format::kIq2xsTileBytes;
        const int r = (int) (row & 15);
        const int sb = (int) (s & 7);
        Raw w;
        w.ent[0] = *reinterpret_cast<const uint32_t *>(o + r * 32 + sb * 4);
        w.ent[1] = *reinterpret_cast<const uint32_t *>(o + (r + 16) * 32 + sb * 4);
        w.scb = o[1024 + r * 8 + sb];
        w.dv = __half2float(*reinterpret_cast<const __half *>(o + 1152 + r * 2));
        return w;
    }
    __device__ void decode(const Raw & r, const long long s, float (&v)[32]) const { DecIq2xs{}.decode(r, s, v); }
};

struct DecIq2xxsTile {  // GGUF type 16, .omph layout 2 (#253)
    const uint8_t * base;
    long long blocks;
    using Raw = DecIq2xxs::Raw;
    static DecIq2xxsTile make(const uint8_t * base, const int64_t, const long long blocks) { return {base, blocks}; }
    __device__ Raw load(const long long row, const long long s) const {
        const uint8_t * o = base + ((row >> 4) * blocks + (s >> 3)) * omph::format::kIq2xxsTileBytes;
        const int r = (int) (row & 15);
        const int sb = (int) (s & 7);
        Raw w;
        const uint32_t i0 = *reinterpret_cast<const uint16_t *>(o + r * 16 + sb * 2);
        const uint32_t i1 = *reinterpret_cast<const uint16_t *>(o + (r + 16) * 16 + sb * 2);
        w.aux0 = i0 | (i1 << 16);
        w.aux1 = *reinterpret_cast<const uint32_t *>(o + 512 + r * 32 + sb * 4);
        w.dv = __half2float(*reinterpret_cast<const __half *>(o + 1024 + r * 2));
        return w;
    }
    __device__ void decode(const Raw & r, const long long s, float (&v)[32]) const { DecIq2xxs{}.decode(r, s, v); }
};

// Q6_K tiles (#258): sub-block sb = 4 g + j takes the low (j < 2) or high
// nibbles of the lane's ql bytes of half g (the first 16 for even j, the second
// for odd), qh bits 2 j, and the scale byte at 4 g + j; each weight
// d sc (q - 32) in f32 (DecQ6k's expression), rounded to f16 once.
struct TileQ6k {  // GGUF type 14
    static constexpr long long kBytes = omph::format::kQ6kTileBytes;
    static constexpr int kLds = 16;
    __device__ static void init(uint8_t *) {}
    struct Blk {
        uint4 q0, q1, q2, q3, h0, h1;
        uint2 scw;
        float dv;
    };
    __device__ static Blk load(const uint8_t * o, const int lane) {
        Blk k;
        const uint4 * q = reinterpret_cast<const uint4 *>(o + lane * 64);
        k.q0 = q[0];
        k.q1 = q[1];
        k.q2 = q[2];
        k.q3 = q[3];
        const uint4 * qh = reinterpret_cast<const uint4 *>(o + 2048 + lane * 32);
        k.h0 = qh[0];
        k.h1 = qh[1];
        k.scw = *reinterpret_cast<const uint2 *>(o + 3072 + lane * 8);
        k.dv = __half2float(*reinterpret_cast<const __half *>(o + 3328 + (lane & 15) * 2));
        return k;
    }
    __device__ static tile_h8 frag(const uint8_t *, const Blk & k, const int sb, const int gl, const int) {
        const int g = sb >> 2;
        const int j = sb & 3;
        const uint4 ql = g == 0 ? ((j & 1) ? k.q1 : k.q0) : ((j & 1) ? k.q3 : k.q2);
        const uint4 qh = g == 0 ? k.h0 : k.h1;
        const uint32_t l0 = gl == 0 ? ql.x : ql.z;
        const uint32_t l1 = gl == 0 ? ql.y : ql.w;
        const uint32_t m0 = gl == 0 ? qh.x : qh.z;
        const uint32_t m1 = gl == 0 ? qh.y : qh.w;
        const int ns = j >= 2 ? 4 : 0;
        const uint32_t c0 = ((l0 >> ns) & 0x0F0F0F0Fu) | (((m0 >> (2 * j)) & 0x03030303u) << 4);
        const uint32_t c1 = ((l1 >> ns) & 0x0F0F0F0Fu) | (((m1 >> (2 * j)) & 0x03030303u) << 4);
        const uint32_t scb = ((g == 0 ? k.scw.x : k.scw.y) >> (8 * j)) & 0xFFu;
        const float ds = k.dv * (float) (int8_t) scb;
        tile_h8 a;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            a[m] = (_Float16) (ds * (float) ((int) ((c0 >> (8 * m)) & 0xFFu) - 32));
            a[4 + m] = (_Float16) (ds * (float) ((int) ((c1 >> (8 * m)) & 0xFFu) - 32));
        }
        return a;
    }
};

struct DecQ6kTile {  // GGUF type 14, .omph layout 2 (#258)
    const uint8_t * base;
    long long blocks;
    using Raw = DecQ6k::Raw;
    static DecQ6kTile make(const uint8_t * base, const int64_t, const long long blocks) { return {base, blocks}; }
    __device__ Raw load(const long long row, const long long s) const {
        const uint8_t * o = base + ((row >> 4) * blocks + (s >> 3)) * omph::format::kQ6kTileBytes;
        const int r = (int) (row & 15);
        const int g = (int) ((s & 7) >> 2);
        const int j = (int) (s & 3);
        const uint32_t * a = reinterpret_cast<const uint32_t *>(o + r * 64 + g * 32);         // bytes 0..15 | 32..47
        const uint32_t * b = reinterpret_cast<const uint32_t *>(o + (r + 16) * 64 + g * 32);  // 16..31 | 48..63
        const uint32_t * ha = reinterpret_cast<const uint32_t *>(o + 2048 + r * 32 + g * 16);
        const uint32_t * hb = reinterpret_cast<const uint32_t *>(o + 2048 + (r + 16) * 32 + g * 16);
        Raw w;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            w.lo[i] = a[i];
            w.lo[4 + i] = b[i];
            w.hi[i] = a[4 + i];
            w.hi[4 + i] = b[4 + i];
            w.hb[i] = ha[i];
            w.hb[4 + i] = hb[i];
        }
        w.s0 = (float) (int8_t) o[3072 + r * 8 + 4 * g + j];
        w.s1 = (float) (int8_t) o[3072 + (r + 16) * 8 + 4 * g + j];
        w.dv = __half2float(*reinterpret_cast<const __half *>(o + 3328 + r * 2));
        return w;
    }
    __device__ void decode(const Raw & r, const long long s, float (&v)[32]) const { DecQ6k{}.decode(r, s, v); }
};

} // namespace omph::kernels::qd
