// DFlash2 drafter kernels (#245): the block-diffusion draft model of
// z-lab/Qwen3.8-27B-DFlash2, as llama.cpp's src/models/dflash.cpp runs it.
#pragma once

#include <cstdint>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace omph::kernels {

// T rows of x (f32, row stride ld_x) into feat (f16, row stride ld_f) at
// column col: one target layer's hidden states into the drafter's features.
bool dflash_capture(const float * x, int64_t T, int64_t n, int64_t ld_x, __half * feat, int64_t ld_f, int64_t col,
                    hipStream_t stream);

// In place on x (T, heads, 128) f32: RMSNorm over each head's 128 values with
// weight w (null: none) and eps, then NeoX RoPE over all 128 dims at position
// pos0 + t with base theta.
bool dflash_norm_rope(float * x, int64_t T, int64_t heads, const float * w, float eps, int64_t pos0, float theta,
                      hipStream_t stream);

// T rows of src (f32, `row` floats each) into the ring (f16, `slots` rows of
// `row`) at slot (pos0 + t) % slots, and the slot's position tag.
bool dflash_ring_put(const float * src, int64_t T, int64_t row, int64_t pos0, __half * ring, int32_t * tags,
                     int64_t slots, hipStream_t stream);

// Non-causal block attention: queries q (T, 32, 128) at positions pos0 + t see
// the ring's keys at positions max(0, pos0 + t - n_swa + 1) .. pos0 - 1 whose
// tag holds that position, and all T block keys bk / bv (T, 8, 128); out (T,
// 32 * 128) f32. GQA: query head h reads KV head h / 4. T <= 8; `ws` holds
// dflash_attention_ws_floats(slots) floats (flash-decoding partials).
bool dflash_attention(const float * q, const __half * ring_k, const __half * ring_v, const int32_t * tags,
                      int64_t slots, const float * bk, const float * bv, int64_t T, int64_t pos0, int64_t n_swa,
                      float scale, float * ws, float * out, hipStream_t stream);
int64_t dflash_attention_ws_floats(int64_t slots);

// DFlash2's two-tap dynamic convolution inside the block: out[t][c] =
// sum_tap (dyn[t][g + groups * (tap + 2 * side)] + base[c + n * (tap + 2 * side)])
// * x[t - tap][c], g = c / group, x[-1] = 0. out must not alias x.
bool dflash_conv(const float * x, const float * dyn, const float * base, int64_t T, int64_t n, int64_t group,
                 int side, float * out, hipStream_t stream);

// The k = 16 largest of each of `rows` rows of n values: ids and values,
// largest first.
bool dflash_topk(const float * x, int64_t rows, int64_t n, int32_t * ids, float * vals, hipStream_t stream);

// DFlash2's selector lattice for block positions 1 .. P: position i's scores
// score[a][b] = vals_i[b] + sum_r prev(a)[r] * gate_i[r] * next(cand_i[b])[r],
// prev(a) the predecessor table's row of cand_{i-1}[a] (the anchor at i = 1),
// both tables repacked Q4_K with one 256-weight block per row (rank 256).
// cand / vals: P x 16 (positions 1 .. P); gate: (P + 1) x 256 (position 0
// first); scores: P x 16 x 16 ([pred][succ]).
bool dflash_selector(const int32_t * cand, const float * vals, const float * gate, int32_t anchor,
                     const void * prev_tab, const void * next_tab, int64_t vocab, int64_t P, float * scores,
                     hipStream_t stream);

} // namespace omph::kernels
