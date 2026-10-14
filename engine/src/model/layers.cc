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
    v.layer = il;
    v.quant = kv_q8q4_;
    if (kv_q8q4_) {
        v.q = quant_kv(il);
        v.window = kv_window_;
        v.ring = kv_ring_;
        v.k_q4 = kv_k4_[(size_t) kv_index_[il]] != 0;
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

    // kv_only (the MTP KV fill, #124): K and V into the cache, nothing else --
    // attn_prep's q blocks are skipped (skip_q), their output is not read.
    const bool grp = kv_only ? grouped(h16_, ne, T, {{&L.attn_k, k_, kv_out}, {&L.attn_v, v_, kv_out}})
                             : grouped(h16_, ne, T,
                                       {{&L.attn_k, k_, kv_out}, {&L.attn_v, v_, kv_out}, {&L.attn_q, fused_, q_out}});
    const bool proj_ok = grp || fork_join(
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
    prep.mpos = inputs_ != nullptr && !inputs_->mpos.empty() ? static_cast<const int *>(mpos_dev_) : nullptr;
    prep.rope_delta = rope_delta_;
    for (int c = 0; c < 3; ++c) prep.sections[c] = h_.rope_sections[c];
    prep.eps = (float) h_.eps;
    prep.rotate = kv.quant;
    prep.skip_q = kv_only;
    if (kv.quant) {
        prep.k_q8 = kv.q.kq;
        prep.k_scales = reinterpret_cast<__half *>(kv.q.ksc);
        prep.v_q4 = kv.q.vq;
        prep.v_scales = reinterpret_cast<__half *>(kv.q.vsc);
        prep.k16 = kv.window > 0 ? reinterpret_cast<__half *>(kv.q.k16) : nullptr;
        prep.v16 = kv.window > 0 ? reinterpret_cast<__half *>(kv.q.v16) : nullptr;
        prep.ring = kv.ring;
        prep.k_q4 = kv.k_q4;
        // #409 step 3: with the streaming working set, the rows at or beyond
        // the resident prefix go to the write staging (by token index) instead
        // of the cache, and the MTP block keeps its whole window in its cache.
        if (kv_split_ > 0 && il < h_.n_layer) {
            prep.kv_split = kv_split_;
            prep.kst_q8 = static_cast<uint8_t *>(kv_wstg_k_);
            prep.kst_scales = static_cast<__half *>(kv_wstg_ks_);
            prep.vst_q4 = static_cast<uint8_t *>(kv_wstg_vq_);
            prep.vst_scales = static_cast<__half *>(kv_wstg_vs_);
        }
    }
    if (!proj_ok || !omph::kernels::attn_prep(prep, nullptr)) {
        return fail("attention prep failed");
    }
    // #409 step 1: the KV the kernel just wrote goes to the host store too.
    if (kv.quant && !kv_stream_mirror(kv.layer, kv.q, pos0, T)) {
        return false;
    }
    // #409 step 3: a prefill chunk (more than the decode tiles' tokens) has no
    // fixed key splits in the attention, so its key range cannot be staged: the
    // prompt must fit the resident KV. Checked here, where the chunk's position
    // and size are known, and only for the stack (the MTP block's cache is
    // windowed and complete).
    if (kv_split_ > 0 && il < h_.n_layer && T > omph::kernels::kAttentionDecTokens &&
        pos0 + T > kv_split_) {
        return fail("OMPH_KV_STREAM_MIB: the prompt does not fit the resident KV (a prefill "
                    "chunk cannot attend to streamed rows yet; raise the budget)");
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
    __half * seq_state = state_cur_[(size_t) il];
    (void) st;

    const bool proj_ok =
        grouped(h16_, ne, T, {{&L.attn_gate, z_, v_dims}, {&L.attn_qkv, fused_, channels}}) || fork_join(
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
    if (L.ssm_ab_f32) {
        // qwen35moe stores the two projections as f32 (#493)
        step.w_beta_f32 = reinterpret_cast<const float *>(L.w_beta);
        step.w_alpha_f32 = reinterpret_cast<const float *>(L.w_alpha);
        step.ab_f32 = true;
    }
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
    __half * alt = verifying_ ? state_alt_[(size_t) il] : nullptr;
    if (verifying_) {
        step.replay = static_cast<float *>(replay_pool_) +
                      rec_index_[(size_t) il] * spec_max_ *
                          omph::kernels::gdn_replay_floats(n_vh, n_kh);
        step.conv_hist = static_cast<float *>(conv_hist_pool_) +
                         rec_index_[(size_t) il] * (h_.ssm_conv_k - 1 + spec_max_) * channels;
    }
    // Several tokens (prefill chunks, verifications): one launch for all of
    // them, the state in registers throughout (#96); bit-identical to a launch
    // per token. OMPH_GDN_PER_TOKEN=1 keeps the per-token launches (A/B).
    if (T > 1 && !env_.gdn_per_token) {
        step.state = seq_state;
        step.state_out = alt;
        step.x16 = static_cast<const __half *>(h16_);
        step.z = static_cast<const float *>(z_);
        step.out16 = static_cast<__half *>(ffn16_);
        // the token-parallel form (#96), its scratch in ffn1_, which the FFN
        // fills only later; OMPH_GDN_SERIAL=1 keeps the single launch (A/B)
        if (!env_.gdn_serial &&
            omph::kernels::gdn_work_floats(T, n_vh, n_kh) <= T * h_.n_ff) {
            step.work = static_cast<float *>(ffn1_);
            // a prefill chunk: the chunked WY form (#240), its scratch in ffn2_ (also
            // free until the FFN); verifications keep the exact per-token rule
            if (!verifying_ && !env_.gdn_exact &&
                omph::kernels::gdn_wy_floats(T, n_vh, n_kh) <= max_tokens_ * h_.n_ff) {
                step.wy = static_cast<float *>(ffn2_);
            }
        }
        if (!omph::kernels::gdn_chunk(step, n_vh, nullptr)) {
            return fail("delta rule (chunk) failed");
        }
    }
    for (int64_t t = 0; t < T && (T == 1 || env_.gdn_per_token); ++t) {
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

// The device cache's per-layer pointers (#409 step 3: its strides are the
// resident row count, not the context's).
Runner::QuantKv Runner::quant_kv(const int64_t il) const {
    const int64_t kv_out = h_.n_head_kv * h_.head_dim;
    const int64_t nblk = h_.head_dim / 32;
    const int64_t kvl = kv_index_[il];
    return {static_cast<uint8_t *>(kv_kq_) + kv_kq_off_[(size_t) kvl],
            static_cast<uint8_t *>(kv_ks_) + kvl * kv_rows_ * h_.n_head_kv * nblk * 2,
            static_cast<uint8_t *>(kv_vq_) + kvl * kv_rows_ * kv_out / 2,
            static_cast<uint8_t *>(kv_vs_) + kvl * kv_rows_ * h_.n_head_kv * nblk * 2,
            static_cast<uint8_t *>(kv_k16_) + kvl * kv_ring_ * kv_out * 2,
            static_cast<uint8_t *>(kv_v16_) + kvl * kv_ring_ * kv_out * 2};
}

// #409 step 3: the host store's per-layer pointers. It has the same layout as
// the device cache, but spans the whole context (the rows past the resident
// prefix live only here) -- hence its own K offset table.
Runner::HostKv Runner::host_kv(const int64_t layer) const {
    const int64_t kvl = kv_index_[(size_t) layer];
    const int64_t kv_out = h_.n_head_kv * h_.head_dim;
    const size_t sc_row = (size_t) h_.n_head_kv * (h_.head_dim / 32) * 2;
    return {static_cast<const uint8_t *>(kv_sh_kq_) + kv_sh_kq_off_[(size_t) kvl],
            static_cast<const uint8_t *>(kv_sh_ks_) + (size_t) kvl * max_seq_ * sc_row,
            static_cast<const uint8_t *>(kv_sh_vq_) + (size_t) kvl * max_seq_ * kv_out / 2,
            static_cast<const uint8_t *>(kv_sh_vs_) + (size_t) kvl * max_seq_ * sc_row};
}

// #409 step 1/3: publish the rows attn_prep just wrote into the pinned host
// store, the authoritative copy of the whole history. Copying the writes rather
// than counting rows is what makes a rewind free: a rejected draft or a
// re-prefill rewrites the same rows and is published again (#122, #342). With
// the streaming mode the device cache holds the resident prefix only, and the
// rows at or beyond `kv_split_` come from the write staging instead (at their
// token index in this call). OMPH_KV_STREAM_CHECK reads every range back once
// and compares it: it catches a copy that failed and an offset that ran past
// the store; a wrong-but-in-bounds offset is written and read by the same
// arithmetic, so the attention reading the store is the content witness.
// `layer` is the stack's attention layer; the MTP block's cache is not mirrored.
bool Runner::kv_stream_mirror(const int64_t layer, const QuantKv & q, const int64_t pos0,
                              const int64_t T) {
    if (!kv_stream_ || T <= 0 || layer < 0 || layer >= h_.n_layer) {
        return true;
    }
    const int64_t kvl = kv_index_[(size_t) layer];
    const int64_t attn_kv = h_.n_head_kv * h_.head_dim;
    const size_t k_row = (size_t) (kv_k4_[(size_t) kvl] ? attn_kv / 2 : attn_kv);
    const size_t sc_row = (size_t) h_.n_head_kv * (h_.head_dim / 32) * 2;
    const size_t v_row = (size_t) attn_kv / 2;
    const struct Part {
        void * host;      // the store's per-layer base (over the whole context)
        const void * dev;  // the device cache's per-layer base (the resident rows)
        const void * wstg;  // the write staging (rows >= kv_split_)
        size_t row;
        const char * name;
    } parts[] = {
        {static_cast<uint8_t *>(kv_sh_kq_) + kv_sh_kq_off_[(size_t) kvl], q.kq, kv_wstg_k_, k_row,
         "K"},
        {static_cast<uint8_t *>(kv_sh_ks_) + (size_t) kvl * max_seq_ * sc_row, q.ksc, kv_wstg_ks_,
         sc_row, "K scales"},
        {static_cast<uint8_t *>(kv_sh_vq_) + (size_t) kvl * max_seq_ * v_row, q.vq, kv_wstg_vq_,
         v_row, "V"},
        {static_cast<uint8_t *>(kv_sh_vs_) + (size_t) kvl * max_seq_ * sc_row, q.vsc, kv_wstg_vs_,
         sc_row, "V scales"},
    };
    // one contiguous range, device source -> the store, with the check
    const auto publish = [&](const Part & p, const uint8_t * from, const int64_t a, const int64_t b) {
        if (b <= a) {
            return true;
        }
        const size_t bytes = (size_t) (b - a) * p.row;
        uint8_t * to = static_cast<uint8_t *>(p.host) + (size_t) a * p.row;
        if (hipMemcpyAsync(to, from, bytes, hipMemcpyDeviceToHost, nullptr) != hipSuccess) {
            return fail("host KV store copy failed");
        }
        if (kv_sh_scratch_ == nullptr) {
            return true;
        }
        if (bytes > kv_sh_scratch_bytes_) {
            return fail("host KV store check: chunk larger than max_tokens");
        }
        if (hipMemcpy(kv_sh_scratch_, from, bytes, hipMemcpyDeviceToHost) != hipSuccess) {
            return fail("host KV store check read failed");
        }
        if (std::memcmp(kv_sh_scratch_, to, bytes) != 0) {
            char msg[160];
            std::snprintf(msg, sizeof(msg), "host KV store mismatch: layer %lld pos0 %lld, %s",
                          (long long) layer, (long long) pos0, p.name);
            return fail(msg);
        }
        return true;
    };
    for (const Part & p : parts) {
        // rows below the split (all of them, when the streaming mode is off)
        const int64_t cut = kv_split_ > 0 ? std::min(pos0 + T, kv_split_) : pos0 + T;
        if (!publish(p, static_cast<const uint8_t *>(p.dev) + (size_t) pos0 * p.row, pos0, cut)) {
            return false;
        }
        // and the rows this call wrote to the write staging
        if (kv_split_ > 0 && pos0 + T > kv_split_) {
            const int64_t w0 = std::max(pos0, kv_split_);
            if (!publish(p, static_cast<const uint8_t *>(p.wstg) + (size_t) (w0 - pos0) * p.row, w0,
                         pos0 + T)) {
                return false;
            }
        }
    }
    return true;
}

// #409 step 2 (OMPH_ATTN_SPLIT_GROUPS): a decode / verification call's fixed
// key splits, run a group at a time with each group's rows staged into a buffer
// of its own. The partials keep their global split indices and the merge is the
// same one, so the result is bit-identical to the single call (#409,
// bench/results/kv-stream-409-splits.txt); the staging and the staged base
// (`kbase`) are the machinery step 3 points at the host store.
bool Runner::attn_splits_staged(const omph::kernels::KvCache & kv, const int64_t layer,
                                const bool q8, const int64_t t0, const int64_t n,
                                const int64_t seq, const int64_t pos0, const float scale) {
    const int64_t qrow = h_.n_head * h_.head_dim;
    // #409 step 3: a decode / verification keeps the WMMA decode tiles; a
    // prefill chunk (more than kAttentionDecTokens tokens) has no fixed splits
    // in the WMMA flash kernel, whose key loop cannot resume over staged ranges
    // yet, so it runs the scalar kernel (step 5's item).
    const bool wmma = n <= omph::kernels::kAttentionDecTokens;
    const auto call = [&](const omph::kernels::KvCache & c, const int64_t split0, const int64_t ns,
                          const bool merge) {
        return omph::kernels::attention_gqa_range(
            static_cast<const float *>(q_) + t0 * qrow, c,
            static_cast<const float *>(gate_) + t0 * qrow, nullptr, n, seq, h_.n_head, h_.n_head_kv,
            h_.head_dim, scale, q8, attn_work_, attn_work_bytes_, nullptr,
            static_cast<__half *>(ffn16_) + t0 * qrow, !env_.attn_scalar && wmma, key_chunk_,
            !env_.attn_dec_scalar, split0, ns, merge);
    };
    // step 3 runs the scalar kernel over the same fixed chunks for a prefill
    // chunk (the WMMA flash kernel cannot resume over staged ranges yet, step 5)
    const omph::kernels::AttentionSplits sp = omph::kernels::attention_dec_splits(seq, key_chunk_);
    if (kv_groups_ < 2 || sp.splits <= 1 || sp.keys <= 0) {
        return call(kv, 0, 0, true);
    }
    const int64_t kv_out = h_.n_head_kv * h_.head_dim;
    const size_t k_row = (size_t) (kv.k_q4 ? kv_out / 2 : kv_out);
    const size_t sc_row = (size_t) h_.n_head_kv * (h_.head_dim / 32) * 2;
    const size_t v_row = (size_t) kv_out / 2;
    // #409 step 3: every split below kv_split_ is resident, and they are one
    // contiguous range of splits (kv_split_ is a whole number of chunks): one
    // call with no copies at all.
    int64_t split_res = kv_split_ > 0 ? std::min(kv_split_ / sp.keys, sp.splits) : 0;
    if (split_res > 0 && !call(kv, 0, split_res, false)) {
        return false;
    }
    const int64_t remaining = sp.splits - split_res;
    if (remaining > 0) {
        const int64_t groups = std::min(kv_groups_, remaining);
        const int64_t per_group = (remaining + groups - 1) / groups;
        omph::kernels::KvCache staged = kv;
        staged.k_q8 = static_cast<const uint8_t *>(kv_g_k_);
        staged.k_scales = kv_g_ks_;
        staged.v_q4 = static_cast<const uint8_t *>(kv_g_vq_);
        staged.v_scales = kv_g_vs_;
        const HostKv store = kv_split_ > 0 && layer >= 0 ? host_kv(layer) : HostKv{};
        for (int64_t split0 = split_res; split0 < sp.splits; split0 += per_group) {
            const int64_t ns = std::min(per_group, sp.splits - split0);
            const int64_t r0 = split0 * sp.keys;
            const int64_t r1 = std::min(r0 + ns * sp.keys, seq);
            // stage [a, b) of each component into the group buffer, whose row 0
            // is r0: `src` is the device cache, the host store or the write
            // staging, and `base` is the cache position of its row 0.
            const auto stage = [&](const void * sk, const void * sks, const void * svq,
                                   const void * svs, const int64_t a, const int64_t b,
                                   const int64_t base, const bool host) {
                if (b <= a) {
                    return true;
                }
                const size_t rows = (size_t) (b - a);
                const size_t off = (size_t) (a - r0);
                const auto mv = [&](void * dst, const void * src, const size_t row) {
                    return hipMemcpyAsync(static_cast<uint8_t *>(dst) + off * row,
                                          static_cast<const uint8_t *>(src) + (size_t) (a - base) * row,
                                          rows * row,
                                          host ? hipMemcpyHostToDevice : hipMemcpyDeviceToDevice,
                                          nullptr) == hipSuccess;
                };
                // #409 step 4: the telemetry OMPH_TIMING reports per step.
                kv_stage_rows_ += (int64_t) rows;
                kv_stage_bytes_ += (int64_t) rows * (int64_t) (k_row + sc_row + v_row + sc_row);
                return mv(kv_g_k_, sk, k_row) && mv(kv_g_ks_, sks, sc_row) && mv(kv_g_vq_, svq, v_row) &&
                       mv(kv_g_vs_, svs, sc_row);
            };
            if (kv_split_ > 0) {
                // the rows this step has not written yet come from the store,
                // the ones it just wrote from the write staging
                const int64_t cut = std::min<int64_t>(r1, pos0);
                const int64_t w0 = std::max<int64_t>(r0, pos0);
                if (!stage(store.kq, store.ksc, store.vq, store.vsc, r0, cut, 0, true) ||
                    !stage(kv_wstg_k_, kv_wstg_ks_, kv_wstg_vq_, kv_wstg_vs_, w0, r1, pos0, false)) {
                    return fail("KV group staging failed");
                }
            } else if (!stage(kv.k_q8, kv.k_scales, kv.v_q4, kv.v_scales, r0, r1, 0, false)) {
                // step 2's diagnostics mode: every group from the device cache
                return fail("KV group staging failed");
            }
            staged.kbase = r0;
            ++kv_stage_groups_;
            if (!call(staged, split0, ns, false)) {
                return false;
            }
        }
    }
    // the merge over every split: the partials are at their global indices
    return omph::kernels::attention_gqa_merge(
        static_cast<const float *>(attn_work_), static_cast<const float *>(gate_) + t0 * qrow,
        nullptr, static_cast<__half *>(ffn16_) + t0 * qrow, n, h_.n_head, sp.splits, q8, nullptr);
}

// KV write + attention, on either the f32 cache (memcpy) or the Q8/Q4 one
// (written by attn_prep); attention_gqa then reads either, dequantizing on the fly.
bool Runner::attn_impl(const KvView & kv_in, const int64_t pos0, const int64_t T) {
    const int64_t kv_out = h_.n_head_kv * h_.head_dim;
    const float scale = 1.0f / std::sqrt((float) h_.head_dim);
    // A verification runs its queries in groups of up to 8 (#251): past 8 tokens
    // attention_gqa leaves the decode kernel's fixed key chunks (9..15: splits
    // by token count; 16+: the prefill kernel), and its rows would no longer
    // be the decode step's bit for bit (#161).
    // #450: a short prefill chunk at a long context leaves the flash kernel's
    // grid nearly empty (one workgroup per 32 rows, 16 in flight - a 32-token
    // turn at 60k costs 4.6 ms per layer for what the 512-row chunk does at
    // 35 us/token), while the decode path's fixed key splits fill the GPU. It is
    // exact for groups of up to kKvRingExtra rows: the ring holds window +
    // kKvRingExtra slots, so every query of the group still reads its own last
    // window (#161, #342). The split geometry is a function of (seq, key_chunk),
    // so a run stays deterministic.
    const bool fa_short = env_.attn_short_tokens > 0 && !verifying_ && T <= env_.attn_short_tokens &&
                          T <= omph::kernels::kKvRingExtra + 1 && kv_in.quant && kv_window_ > 0 &&
                          kv_in.q.k16 != nullptr && kv_ring_ >= kv_window_ + omph::kernels::kKvRingExtra;
    const int64_t group = (verifying_ || fa_short) ? 8 : T;
    const int64_t qrow = h_.n_head * h_.head_dim;
    const auto attend = [&](const omph::kernels::KvCache & kv, const bool q8) {
        for (int64_t t0 = 0; t0 < T; t0 += group) {
            const int64_t n = std::min(group, T - t0);
            // #409 step 2/3: with the split groups on (or the streaming working
            // set), a call stages each group of key splits and merges at the end.
            // The MTP block (layer < 0) keeps its whole windowed cache and its
            // single call: its rows are not the stack's positions.
            const bool grouped = kv_groups_ > 1 && kv_in.layer >= 0 &&
                                 n <= omph::kernels::kAttentionDecTokens;
            const bool ok = grouped
                                ? attn_splits_staged(kv, kv_in.layer, q8, t0, n, pos0 + t0 + n, pos0,
                                                     scale)
                                : fa_short
                                ? omph::kernels::attention_gqa_range(
                                      static_cast<const float *>(q_) + t0 * qrow, kv,
                                      static_cast<const float *>(gate_) + t0 * qrow, nullptr, n,
                                      pos0 + t0 + n, h_.n_head, h_.n_head_kv, h_.head_dim, scale, q8,
                                      attn_work_, attn_work_bytes_, nullptr,
                                      static_cast<__half *>(ffn16_) + t0 * qrow, false, key_chunk_,
                                      !env_.attn_dec_scalar, 0, 0, true)
                                : omph::kernels::attention_gqa(
                                      static_cast<const float *>(q_) + t0 * qrow, kv,
                                      static_cast<const float *>(gate_) + t0 * qrow, nullptr, n,
                                      pos0 + t0 + n, h_.n_head, h_.n_head_kv, h_.head_dim, scale, q8,
                                      attn_work_, attn_work_bytes_, nullptr,
                                      static_cast<__half *>(ffn16_) + t0 * qrow, !env_.attn_scalar,
                                      key_chunk_, !env_.attn_dec_scalar);
            if (!ok) {
                return false;
            }
        }
        return true;
    };
    if (kv_in.quant) {
        omph::kernels::KvCache kv;
        kv.k_q8 = kv_in.q.kq;
        kv.k_scales = kv_in.q.ksc;
        kv.v_q4 = kv_in.q.vq;
        kv.v_scales = kv_in.q.vsc;
        kv.k16 = kv_in.window > 0 ? kv_in.q.k16 : nullptr;
        kv.v16 = kv_in.window > 0 ? kv_in.q.v16 : nullptr;
        kv.window = kv_in.window;
        kv.ring = kv_in.ring;
        kv.k_q4 = kv_in.k_q4;
        return attend(kv, true);
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
        // The layer's own staging when it has one (#316): only the rows this call
        // adds are uploaded. Between two calls of this layer nothing else writes
        // its rows (the other layers have their own caches), except this call's
        // own [pos0, pos0 + T) -- so a pos0 below what is staged means the
        // sequence was rewound (a rejected draft, a checkpoint) and those rows
        // were just rewritten over the garbage a rollback leaves: re-upload from
        // pos0, not from 0. Everything below pos0 is unchanged.
        if (Runner::HostStage * st = host_stage(kv_in.layer); st != nullptr) {
            const int64_t want = pos0 + T;
            if (pos0 < st->rows) {
                st->rows = pos0;
            }
            if (st->rows < want) {
                const size_t from = (size_t) st->rows * kv_out * 4;
                const size_t bytes = (size_t) (want - st->rows) * kv_out * 4;
                if (hipMemcpy(static_cast<uint8_t *>(static_cast<void *>(st->k)) + from,
                              k_cache + st->rows * kv_out, bytes, hipMemcpyHostToDevice) != hipSuccess ||
                    hipMemcpy(static_cast<uint8_t *>(static_cast<void *>(st->v)) + from,
                              v_cache + st->rows * kv_out, bytes, hipMemcpyHostToDevice) != hipSuccess) {
                    return false;
                }
                st->rows = want;
            }
            k_cache = st->k;
            v_cache = st->v;
        } else {
            // No staging of its own: the whole prefix, on every call, for this
            // layer. The shared buffer cannot be kept across calls, because the
            // other layers overwrite its rows with theirs (a per-layer row
            // counter cannot tell what is already staged: tried and caught by the
            // bit check in #315). That costs 8 KiB of host-to-device traffic per
            // token per layer: measured 55 vs 42 ms per step at 2.6k tokens, i.e.
            // ~12 ms of PCIe against 42 ms of GPU, growing with the context.
            // Final fix when the pairs do not fit: a key offset in attention_gqa,
            // staging the keys in pieces and accumulating the splits (#316).
            const size_t bytes = (size_t) (pos0 + T) * kv_out * 4;
            kv_shared_stage((size_t) max_seq_ * kv_out * 4);  // once, at the first such layer (#342)
            if (hipMemcpy(kv_stage_k_, k_cache, bytes, hipMemcpyHostToDevice) != hipSuccess ||
                hipMemcpy(kv_stage_v_, v_cache, bytes, hipMemcpyHostToDevice) != hipSuccess) {
                return false;
            }
            k_cache = static_cast<float *>(kv_stage_k_);
            v_cache = static_cast<float *>(kv_stage_v_);
        }
    }
    omph::kernels::KvCache kv;
    kv.k_f32 = k_cache;
    kv.v_f32 = v_cache;
    return attend(kv, false);
}

} // namespace omph::model
