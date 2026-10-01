// Runner: the full-attention and gated-delta-net blocks and the KV cache.
#include "model/runner.hh"

#include "format/repack.hh"
#include "kernels/attn.hh"
#include "kernels/dequant.hh"
#include "kernels/elementwise.hh"
#include "kernels/gdn.hh"
#include "kernels/gemv.hh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace omph::model {

// Which cache an attention layer reads and writes: the stack's (by kv_index_)
// in whatever mode it runs, or the MTP block's (always Q8/Q4, no FP16 ring:
// draft rows must never overwrite slots of older positions, #124).
Runner::KvView Runner::kv_view(const int64_t il) const {
    KvView v;
    if (il == h_.n_layer) {
        v.quant = true;
        v.q = {static_cast<uint8_t *>(mtp_kq_), mtp_ks_, static_cast<uint8_t *>(mtp_vq_), mtp_vs_,
               nullptr, nullptr};
        return v;
    }
    const int64_t kv_out = h_.n_head_kv * h_.head_dim;
    v.quant = kv_q8q4_;
    if (kv_q8q4_) {
        v.q = quant_kv(il);
        v.window = kv_window_;
        v.k_q4 = kv_k4_;
    } else {
        // Null in the quantized-KV mode, where the f32 cache is never allocated.
        v.k_f32 = static_cast<float *>(kv_k_) + kv_index_[il] * max_seq_ * kv_out;
        v.v_f32 = static_cast<float *>(kv_v_) + kv_index_[il] * max_seq_ * kv_out;
    }
    return v;
}

bool Runner::attn_layer(const int64_t il, const LayerWeights & L, const int64_t T,
                        const int64_t pos0, const bool kv_only) {
    const int64_t ne = h_.n_embd;
    const int64_t q_out = h_.n_head * 2 * h_.head_dim;
    const int64_t kv_out = h_.n_head_kv * h_.head_dim;
    const KvView kv = kv_view(il);

    // kv_only (the MTP KV fill, #124): K and V into the cache, nothing else.
    // attn_prep still runs its q heads on whatever fused_ holds; nobody reads
    // their output.
    const bool proj_ok = fork_join(
        T,
        [&] {
            return matmul(L.attn_k, h16_, static_cast<float *>(k_), kv_out, ne, T) &&
                   matmul(L.attn_v, h16_, static_cast<float *>(v_), kv_out, ne, T);
        },
        [&] {
            return kv_only ||
                   matmul(L.attn_q, h16_, static_cast<float *>(fused_), q_out, ne, T);
        });
    // split, QK-norm, RoPE, the Hadamard rotation and (quantized cache)
    // the KV write, in one launch (#79).
    omph::kernels::AttnPrep prep;
    prep.qg = static_cast<const float *>(fused_);
    prep.q = static_cast<float *>(q_);
    prep.gate = static_cast<float *>(gate_);
    prep.k = static_cast<float *>(k_);
    prep.v = static_cast<float *>(v_);
    prep.q_norm = L.q_norm;
    prep.k_norm = L.k_norm;
    prep.tokens = T;
    prep.pos0 = pos0;
    prep.nh = h_.n_head;
    prep.nkv = h_.n_head_kv;
    prep.hd = h_.head_dim;
    prep.n_rot = (int) h_.n_rot;
    prep.freq_base = (float) h_.freq_base;
    prep.eps = (float) h_.eps;
    prep.rotate = kv.quant;
    if (kv.quant) {
        prep.k_q8 = kv.q.kq;
        prep.k_scales = reinterpret_cast<__half *>(kv.q.ksc);
        prep.v_q4 = kv.q.vq;
        prep.v_scales = reinterpret_cast<__half *>(kv.q.vsc);
        prep.k16 = kv.window > 0 ? reinterpret_cast<__half *>(kv.q.k16) : nullptr;
        prep.v16 = kv.window > 0 ? reinterpret_cast<__half *>(kv.q.v16) : nullptr;
        prep.window = kv.window;
        prep.k_q4 = kv.k_q4;
    }
    if (!proj_ok || !omph::kernels::attn_prep(prep, nullptr)) {
        return fail("attention prep failed");
    }
    if (kv_only) {
        if (!kv.quant &&
            (hipMemcpy(kv.k_f32 + pos0 * kv_out, k_, (size_t) T * kv_out * 4, hipMemcpyDefault) !=
                 hipSuccess ||
             hipMemcpy(kv.v_f32 + pos0 * kv_out, v_, (size_t) T * kv_out * 4, hipMemcpyDefault) !=
                 hipSuccess)) {
            return fail("kv write failed");
        }
        return true;
    }
    if (!attn_impl(kv, pos0, T) ||
        !matmul(L.attn_output, ffn16_, static_cast<float *>(blk_), ne,
                h_.n_head * h_.head_dim, T)) {
        return fail("attention layer failed");
    }
    return true;
}

bool Runner::gdn_layer(const int64_t il, const LayerWeights & L, const int64_t T) {
    const int64_t ne = h_.n_embd;
    const int64_t n_kh = h_.ssm_n_kh;
    const int64_t n_vh = h_.ssm_n_vh;
    const int64_t s = h_.ssm_s;
    const int64_t v_dim = h_.ssm_inner / n_vh;
    const int64_t q_dims = n_kh * s;
    const int64_t k_dims = n_kh * s;
    const int64_t v_dims = n_vh * v_dim;
    const int64_t channels = q_dims + k_dims + v_dims;
    const float l2_scale = 1.0f / std::sqrt((float) s);

    const float * dt_bias = L.dt_bias;
    const float * ssm_a = L.ssm_a;
    const float * ssm_norm = L.ssm_norm;
    const float * conv_w = L.conv_w;
    uint8_t * st = static_cast<uint8_t *>(states_[il]);
    const int64_t n_conv_f = (h_.ssm_conv_k - 1) * channels;
    float * conv_a = reinterpret_cast<float *>(st);
    float * conv_b = conv_a + n_conv_f;
    float * conv_cur = conv_flip_[il] ? conv_b : conv_a;
    float * conv_new = conv_flip_[il] ? conv_a : conv_b;
    float * seq_state = state_cur_[(size_t) il];
    (void) st;

    const bool proj_ok = fork_join(
        T,
        [&] {
            // beta / alpha (BF16, 48 rows each) are dotted inside gdn_step (#88)
            return matmul(L.attn_gate, h16_, static_cast<float *>(z_), v_dims, ne, T);
        },
        [&] {
            return matmul(L.attn_qkv, h16_, static_cast<float *>(fused_), channels, ne, T);
        });
    if (!proj_ok) {
        return fail("gdn projection failed");
    }
    // conv, L2 norms, gates, delta rule and gated norm: one launch per token
    // (#78, #83). The L2 norm is rms_norm(x, eps/s) / sqrt(s).
    omph::kernels::GdnStep step;
    step.state = seq_state;
    step.qkv = static_cast<const float *>(fused_);
    step.conv_w = conv_w;
    step.conv_cur = conv_cur;
    step.conv_new = conv_new;
    step.w_beta = L.w_beta;
    step.w_alpha = L.w_alpha;
    step.k_in = ne;
    step.dt_bias = dt_bias;
    step.ssm_a = ssm_a;
    step.norm_w = ssm_norm;
    step.tokens = T;
    step.channels = channels;
    step.q_dims = q_dims;
    step.kv_dims = k_dims;
    step.conv_k = h_.ssm_conv_k;
    step.n_kh = n_kh;
    step.eps_l2 = (float) (h_.eps / (double) s);
    step.l2_scale = l2_scale;
    step.eps_norm = (float) h_.eps;
    // A speculative verification (#122) reads the state from the current buffer
    // and leaves it intact: the first token writes the alternate one, the rest
    // update that; every token records its rank-1 factors, and the first the
    // conv input history, so commit() can roll back to any prefix.
    float * alt = verifying_ ? state_alt_[(size_t) il] : nullptr;
    if (verifying_) {
        step.replay = static_cast<float *>(replay_pool_) +
                      rec_index_[(size_t) il] * spec_max_ *
                          omph::kernels::gdn_replay_floats(n_vh, n_kh);
        step.conv_hist = static_cast<float *>(conv_hist_pool_) +
                         rec_index_[(size_t) il] * (h_.ssm_conv_k - 1 + spec_max_) * channels;
    }
    for (int64_t t = 0; t < T; ++t) {
        step.t = t;
        if (alt != nullptr) {
            step.state = t == 0 ? seq_state : alt;
            step.state_out = alt;
        }
        step.x16 = reinterpret_cast<const __half *>(static_cast<const uint8_t *>(h16_) +
                                                    t * ne * 2);
        step.z = static_cast<const float *>(z_) + t * v_dims;
        step.out16 = reinterpret_cast<__half *>(static_cast<uint8_t *>(ffn16_) + t * v_dims * 2);
        if (!omph::kernels::gdn_step(step, n_vh, nullptr)) {
            return fail("delta rule failed");
        }
    }
    if (!matmul(L.ssm_out, ffn16_, static_cast<float *>(blk_), ne, v_dims, T)) {
        return fail("gdn output failed");
    }
    conv_flip_[il] ^= 1;
    return true;
}

Runner::QuantKv Runner::quant_kv(const int64_t il) const {
    const int64_t kv_out = h_.n_head_kv * h_.head_dim;
    const int64_t nblk = h_.head_dim / 32;
    const int64_t kvl = kv_index_[il];
    return {static_cast<uint8_t *>(kv_kq_) + kvl * max_seq_ * kv_out / (kv_k4_ ? 2 : 1),
            static_cast<uint8_t *>(kv_ks_) + kvl * max_seq_ * h_.n_head_kv * nblk * 2,
            static_cast<uint8_t *>(kv_vq_) + kvl * max_seq_ * kv_out / 2,
            static_cast<uint8_t *>(kv_vs_) + kvl * max_seq_ * h_.n_head_kv * nblk * 2,
            static_cast<uint8_t *>(kv_k16_) + kvl * kv_window_ * kv_out * 2,
            static_cast<uint8_t *>(kv_v16_) + kvl * kv_window_ * kv_out * 2};
}

// KV write + attention, on either the f32 cache (memcpy) or the Q8/Q4 one
// (quantize); attention_gqa then reads either, dequantizing on the fly.
// KV write + attention, on either the f32 cache (memcpy) or the Q8/Q4 one
// (written by attn_prep); attention_gqa then reads either, dequantizing on the fly.
bool Runner::attn_impl(const KvView & kv_in, const int64_t pos0, const int64_t T) {
    const int64_t kv_out = h_.n_head_kv * h_.head_dim;
    const float scale = 1.0f / std::sqrt((float) h_.head_dim);
    if (kv_in.quant) {
        omph::kernels::KvCache kv;
        kv.k_q8 = kv_in.q.kq;
        kv.k_scales = kv_in.q.ksc;
        kv.v_q4 = kv_in.q.vq;
        kv.v_scales = kv_in.q.vsc;
        kv.k16 = kv_in.window > 0 ? kv_in.q.k16 : nullptr;
        kv.v16 = kv_in.window > 0 ? kv_in.q.v16 : nullptr;
        kv.window = kv_in.window;
        kv.k_q4 = kv_in.k_q4;
        return omph::kernels::attention_gqa(
            static_cast<const float *>(q_), kv, static_cast<const float *>(gate_),
            nullptr, T, pos0 + T, h_.n_head, h_.n_head_kv, h_.head_dim, scale, true,
            attn_work_, attn_work_bytes_, nullptr, ffn16_);
    }
    float * k_cache = kv_in.k_f32;
    float * v_cache = kv_in.v_f32;
    if (k_cache == nullptr || v_cache == nullptr) {
        return false;
    }
    // hipMemcpyDefault: the cache is device memory, or pinned host memory
    // with OMPH_KV_HOST.
    if (hipMemcpy(k_cache + pos0 * kv_out, k_, (size_t) T * kv_out * 4, hipMemcpyDefault) !=
            hipSuccess ||
        hipMemcpy(v_cache + pos0 * kv_out, v_, (size_t) T * kv_out * 4, hipMemcpyDefault) !=
            hipSuccess) {
        return false;
    }
    if (kv_host_) {
        const size_t bytes = (size_t) (pos0 + T) * kv_out * 4;
        if (hipMemcpy(kv_stage_k_, k_cache, bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(kv_stage_v_, v_cache, bytes, hipMemcpyHostToDevice) != hipSuccess) {
            return false;
        }
        k_cache = static_cast<float *>(kv_stage_k_);
        v_cache = static_cast<float *>(kv_stage_v_);
    }
    omph::kernels::KvCache kv;
    kv.k_f32 = k_cache;
    kv.v_f32 = v_cache;
    return omph::kernels::attention_gqa(static_cast<const float *>(q_), kv,
                                        static_cast<const float *>(gate_),
                                        nullptr, T, pos0 + T, h_.n_head, h_.n_head_kv,
                                        h_.head_dim, scale, false, attn_work_,
                                        attn_work_bytes_, nullptr, ffn16_);
}

} // namespace omph::model
