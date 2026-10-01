// Attention block kernels (PLAN.md §10.4, §13).
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
// `k16`/`v16` (may be null) hold an exact f16 copy of the most recent `window`
// tokens in a ring: the FP16 window of PLAN §13.4, in the same rotated basis.
bool kv_quant(const float * k_src, const float * v_src, uint8_t * k_q8, void * k_scales,
              uint8_t * v_q4, void * v_scales, void * k16, void * v16, const int64_t pos,
              const int64_t tokens, const int64_t nkv, const int64_t hd, const int64_t window,
              hipStream_t stream);

// --- GQA-grouped flash attention (issue #59) -------------------------------
//
// One layer's KV cache, in either storage: f32 (k_f32/v_f32, (seq, nkv, hd)) or
// K Q8 / V Q4 with the optional FP16 window (the fields of kv_quant).
struct KvCache {
    const float * k_f32 = nullptr;
    const float * v_f32 = nullptr;
    const uint8_t * k_q8 = nullptr;
    const void * k_scales = nullptr;
    const uint8_t * v_q4 = nullptr;
    const void * v_scales = nullptr;
    const void * k16 = nullptr;
    const void * v16 = nullptr;
    int64_t window = 0;
};

// Causal attention with GQA and the sigmoid output gate. q, gate, out:
// (tokens, nh, hd), q post-RoPE (and rotated for a Q8/Q4 cache); query t sits
// at cache row seq - tokens + t. A
// workgroup serves a tile of query tokens times the nh/nkv query heads of one KV
// head, so each K/V block is read and dequantized once for all of them; the key
// range is split so a decode step still fills the GPU, and a second kernel
// merges the splits in a fixed order (deterministic), un-rotates V when
// `v_rotated` (Q8/Q4 cache) and applies the gate. Written for this model:
// head_dim 256 and 6 query heads per KV head. `work` must hold
// attention_gqa_work_bytes(tokens, ...).
bool attention_gqa(const float * q, const KvCache & kv, const float * gate, float * out,
                   int64_t tokens, int64_t seq, int64_t nh, int64_t nkv, int64_t hd, float scale,
                   bool v_rotated, void * work, size_t work_bytes, hipStream_t stream);

// Workspace for any call with up to `max_tokens` query tokens.
size_t attention_gqa_work_bytes(int64_t max_tokens, int64_t nh, int64_t nkv, int64_t hd);

} // namespace omph::kernels
