// Attention block kernels for the naive path (PLAN.md §10.4).
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

namespace omph::kernels {

// Splits the fused Q projection (ggml: 2*head_dim per head = [q | gate]) into
// contiguous q and gate tensors of shape (tokens, heads, head_dim).
bool split_qg(const float * q_full, float * q, float * gate, int64_t tokens, int64_t heads,
              int64_t head_dim, hipStream_t stream);

// Partial NeoX RoPE in place: rotates pairs (i, i + n_rot/2) of the first n_rot
// head dims; theta_i = position / freq_base^(2i/n_rot).
bool rope_neox(float * x, int64_t tokens, int64_t heads, int64_t head_dim, int64_t n_rot,
               float freq_base, int64_t pos_offset, hipStream_t stream);

// Naive causal attention with a KV cache (f32), GQA and the sigmoid output gate.
//   q: (tokens, nh, hd) post-RoPE; k_cache/v_cache: (seq, nkv, hd);
//   gate: (tokens, nh, hd) or nullptr; out: (tokens, nh, hd).
// Query token t corresponds to cache row (seq - tokens + t).
bool attention(const float * q, const float * k_cache, const float * v_cache, const float * gate,
               float * out, int64_t tokens, int64_t seq, int64_t nh, int64_t nkv, int64_t hd,
               float scale, hipStream_t stream);

// --- quantized KV (PLAN.md §13) -------------------------------------------
//
// K is stored as Q8 and V as Q4, in 32-element blocks along head_dim with the
// scales in a separate stream. head_dim is 256, so there are 8 blocks per head
// per token: K is 272 B per head (256 + 16 of scales) and V is 144 B, against
// 1024 B each in f32.

// Fast Walsh-Hadamard along n (a power of two), applied to q and k after RoPE.
// H is orthogonal, so (Hq)·(Hk) = q·k and the scores are unchanged while the
// outliers get spread across the channels, which is what makes Q4 viable.
bool hadamard_f32(float * x, const int64_t rows, const int64_t n, hipStream_t stream);

// Quantizes the T token rows of k/v (f32, T * nkv * hd each) into the cache at
// `pos`, computing one scale per 32-element block.
bool kv_quant(const float * k_src, const float * v_src, uint8_t * k_q8, void * k_scales,
              uint8_t * v_q4, void * v_scales, const int64_t pos, const int64_t tokens,
              const int64_t nkv, const int64_t hd, hipStream_t stream);

// Flash attention over the quantized cache: same contract as `attention`, with K
// and V dequantized as they are read.
bool attention_flash_q8q4(const float * q, const uint8_t * k_q8, const void * k_scales,
                          const uint8_t * v_q4, const void * v_scales, const float * gate,
                          float * out, const int64_t tokens, const int64_t seq,
                          const int64_t nh, const int64_t nkv, const int64_t hd,
                          const float scale, const int64_t head_ratio, hipStream_t stream);

} // namespace omph::kernels
