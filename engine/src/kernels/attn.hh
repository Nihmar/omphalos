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

} // namespace omph::kernels
