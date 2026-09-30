// Lossless Q4_K repacking at load time (PLAN.md §8.3).
//
// GGUF stores 256-weight blocks as an array of structs (144 B each: d, dmin,
// a 12-byte 6-bit scale/min area and 128 B of packed nibbles); 144 is not
// 16-byte aligned, so a lane cannot issue aligned 128-bit loads.
//
// The repacked layout is struct-of-arrays with the nibbles reordered so that
// each 32-weight sub-block occupies 16 contiguous, 16-byte aligned bytes:
//
//   [ qs  ] n_blocks * 128 B   sub-block s at s*16, value v in nibble v&1 of byte v/2
//   [ sc  ] n_blocks * 8 B     6-bit scales, decoded to bytes
//   [ mn  ] n_blocks * 8 B     6-bit mins,   decoded to bytes
//   [ d   ] n_blocks * 2 B     f16 super-block scale
//   [ dmin] n_blocks * 2 B     f16 super-block min-scale
//
// Same size as the source. Every value keeps its exact 4-bit code and every
// scale/min keeps its exact 6-bit value, so dequantization is bit-identical.
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace omph::format {

struct Q4kLayout {
    int64_t n_blocks = 0;   // 256-weight blocks
    int64_t qs_off = 0;
    int64_t sc_off = 0;
    int64_t mn_off = 0;
    int64_t d_off = 0;
    int64_t dmin_off = 0;
    int64_t total = 0;      // bytes (sections padded to 128 B)
};

// Section offsets for a tensor with `n_blocks` 256-weight blocks.
Q4kLayout q4k_layout(int64_t n_blocks);

// Repacks `n_blocks` GGUF Q4_K blocks (144 B each) into `dst`.
void repack_q4k(const void * src, int64_t n_blocks, void * dst);

// Rebuilds the original GGUF block stream from the repacked layout; used to
// prove the repacking is lossless (byte-for-byte comparison with the source).
void unrepack_q4k(const void * src, int64_t n_blocks, void * dst);

// ------------------------------------------------------------------ IQ4_XS
//
// GGUF IQ4_XS: 136-byte blocks (d f16, a 16-bit `scales_h`, 4 bytes of
// `scales_l`, 128 bytes of 4-bit codes into the 16-entry non-linear codebook
// kvalues_iq4nl). The codes already form one contiguous 16-byte run per
// 32-weight sub-block, so only the scales move:
//
//   [ qs ] n_blocks * 128 B   16 B per sub-block, codes as in the source
//   [ sc ] n_blocks * 8 B     the 8 six-bit scales, decoded to bytes
//   [ d  ] n_blocks * 2 B     f16 per-block scale
struct Iq4Layout {
    int64_t n_blocks = 0;
    int64_t qs_off = 0;
    int64_t sc_off = 0;
    int64_t d_off = 0;
    int64_t total = 0;
};

Iq4Layout iq4_layout(int64_t n_blocks);
void repack_iq4_xs(const void * src, int64_t n_blocks, void * dst);
void unrepack_iq4_xs(const void * src, int64_t n_blocks, void * dst);

// The 16 int8 codebook values of IQ4_NL / IQ4_XS (ggml-common.h).
extern const int8_t kIq4Codebook[16];

// ------------------------------------------------------------------ IQ3_XXS
//
// GGUF IQ3_XXS: 98-byte blocks — f16 `d`, 64 bytes of 4-byte grid indices
// (iq3xxs_grid) and 32 bytes of packed words holding a 4-bit scale plus four
// 7-bit sign indices each. The repack only splits the fields into aligned
// streams; the kernel resolves the sign indices with the 128-byte table from
// kernels/iq_tables.hh, so the block keeps its 98 bytes:
//
//   [ qs  ] 64 B/block : grid indices, as in the source
//   [ aux ] 32 B/block : the scale/sign words, as in the source
//   [ d   ]  2 B/block : f16
struct Iq3XxsLayout {
    int64_t n_blocks = 0;
    int64_t qs_off = 0;
    int64_t aux_off = 0;
    int64_t d_off = 0;
    int64_t total = 0;
};

Iq3XxsLayout iq3_xxs_layout(int64_t n_blocks);
void repack_iq3_xxs(const void * src, int64_t n_blocks, void * dst);
void unrepack_iq3_xxs(const void * src, int64_t n_blocks, void * dst);

// ------------------------------------------------------------------ IQ3_S
//
// GGUF IQ3_S: 110-byte blocks — f16 `d`, 64 bytes of low grid-index bits,
// 8 bytes holding the 9th index bit for each 32-weight sub-block, 32 bytes of
// sign masks, 4 bytes of packed 4-bit scales. Same size after repacking; the
// kernel decodes the scale nibbles itself.
//
//   [ qs     ] 64 B/block
//   [ qh     ]  8 B/block
//   [ signs  ] 32 B/block
//   [ scales ]  4 B/block
//   [ d      ]  2 B/block
struct Iq3sLayout {
    int64_t n_blocks = 0;
    int64_t qs_off = 0;
    int64_t qh_off = 0;
    int64_t signs_off = 0;
    int64_t scales_off = 0;
    int64_t d_off = 0;
    int64_t total = 0;
};

Iq3sLayout iq3s_layout(int64_t n_blocks);
void repack_iq3_s(const void * src, int64_t n_blocks, void * dst);
void unrepack_iq3_s(const void * src, int64_t n_blocks, void * dst);

// ------------------------------------------------------------------ Q2_K
//
// GGUF Q2_K: 84-byte blocks — 16 bytes of packed 4-bit scales (low nibble
// scale, high nibble min), 64 bytes of 2-bit quants, then f16 `d` and `dmin`.
// The repack only splits the fields into aligned streams; the size is unchanged
// and the kernel decodes the scale nibbles itself.
//
//   [ sc   ] 16 B/block
//   [ qs   ] 64 B/block
//   [ d    ]  2 B/block
//   [ dmin ]  2 B/block
struct Q2kLayout {
    int64_t n_blocks = 0;
    int64_t sc_off = 0;
    int64_t qs_off = 0;
    int64_t d_off = 0;
    int64_t dmin_off = 0;
    int64_t total = 0;
};

Q2kLayout q2k_layout(int64_t n_blocks);
void repack_q2k(const void * src, int64_t n_blocks, void * dst);
void unrepack_q2k(const void * src, int64_t n_blocks, void * dst);

// Byte size of one quantized block of `type` (0 when unsupported).
int64_t quant_block_bytes(uint32_t type);

// Dispatch helpers over the types with a repacked layout + GEMV kernel.
bool repack_any(uint32_t type, const void * src, int64_t n_blocks, std::vector<uint8_t> & dst);
bool unrepack_any(uint32_t type, const void * src, int64_t n_blocks, std::vector<uint8_t> & dst);

// ------------------------------------------------------------------ Q6_K
//
// GGUF Q6_K: 210-byte blocks — 128 bytes of low nibbles, 64 bytes carrying the
// upper 2 bits, 16 signed 8-bit scales (one per 16 weights) and f16 `d`.
// Streams are split for alignment; size unchanged.
//   [ ql ] 128 B   [ qh ] 64 B   [ sc ] 16 B   [ d ] 2 B
struct Q6kLayout {
    int64_t n_blocks = 0;
    int64_t ql_off = 0;
    int64_t qh_off = 0;
    int64_t sc_off = 0;
    int64_t d_off = 0;
    int64_t total = 0;
};
Q6kLayout q6k_layout(int64_t n_blocks);
void repack_q6k(const void * src, int64_t n_blocks, void * dst);
void unrepack_q6k(const void * src, int64_t n_blocks, void * dst);

// ------------------------------------------------------------------ IQ2_XS / IQ2_XXS
//
// GGUF IQ2_XS: 74-byte blocks — f16 `d`, 32 uint16 entries (9-bit grid index +
// 7-bit sign index) and 8 bytes of 4-bit scales. IQ2_XXS: 66-byte blocks — f16
// `d` and 32 uint16 words each holding a grid index, a 7-bit sign index and a
// 4-bit scale. Both only need their fields split into aligned streams; sizes are
// unchanged and the kernels resolve signs with the 128-byte table.
//   IQ2_XS : [ qs 64 ][ sc 8 ][ d 2 ]
//   IQ2_XXS: [ qs 64 ][ d 2 ]
struct Iq2Layout {
    int64_t n_blocks = 0;
    int64_t qs_off = 0;
    int64_t sc_off = 0;
    int64_t d_off = 0;
    int64_t total = 0;
};

Iq2Layout iq2_xs_layout(int64_t n_blocks);
void repack_iq2_xs(const void * src, int64_t n_blocks, void * dst);
void unrepack_iq2_xs(const void * src, int64_t n_blocks, void * dst);

Iq2Layout iq2_xxs_layout(int64_t n_blocks);
void repack_iq2_xxs(const void * src, int64_t n_blocks, void * dst);
void unrepack_iq2_xxs(const void * src, int64_t n_blocks, void * dst);

} // namespace omph::format
