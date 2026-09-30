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

} // namespace omph::format
