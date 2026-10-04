// Runner: the DFlash2 block-diffusion drafter (#245), as llama.cpp runs it
// (src/models/dflash.cpp, common/speculative.cpp draft-dflash).
//
// Context: every kept token's target features (the outputs of the target
// layers the drafter names, concatenated) go through fc + RMSNorm, then each
// draft layer's K (k_norm, RoPE) and V into the drafter's KV ring (sliding
// window). A draft: the block [token, MASK x (B - 1)] at positions pos.. runs
// the 5 layers with non-causal attention over the ring and the block, and
// DFlash2's dynamic two-tap convolutions around attention and FFN; the
// target's lm_head gives each MASK position's logits, and the selector scores
// the top-16 candidates of adjacent positions; the drafts are its greedy path.
#include "model/runner.hh"

#include "kernels/dflash.hh"
#include "kernels/elementwise.hh"
#include "kernels/gemv.hh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace omph::model {

namespace {

constexpr int64_t kHeads = 32;
constexpr int64_t kKvHeads = 8;
constexpr int64_t kHd = 128;
constexpr int64_t kTopK = 16;
constexpr int64_t kRank = 256;

int64_t meta_i(const omph::gguf::File & f, const char * key) {
    const omph::gguf::Value * v = f.find(key);
    uint64_t u = 0;
    if (v == nullptr || !v->as_u64(u)) {
        throw std::runtime_error(std::string("drafter: missing ") + key);
    }
    return (int64_t) u;
}

double meta_f(const omph::gguf::File & f, const char * key) {
    const omph::gguf::Value * v = f.find(key);
    if (v == nullptr) {
        throw std::runtime_error(std::string("drafter: missing ") + key);
    }
    return v->f;
}

} // namespace

void Runner::dflash_load() {
    const omph::gguf::File & f = *dft_file_;
    const auto * arch = f.find("general.architecture");
    std::string_view a;
    if (arch == nullptr || !arch->as_str(a) || a != "dflash" || f.tensor("selector_hidden.weight") == nullptr) {
        throw std::runtime_error("the drafter is not a DFlash2 model");
    }
    dfl_ne_ = meta_i(f, "dflash.embedding_length");
    dfl_ff_ = meta_i(f, "dflash.feed_forward_length");
    dfl_swa_ = meta_i(f, "dflash.attention.sliding_window");
    dfl_group_ = meta_i(f, "dflash.conv_group_size");
    dfl_eps_ = (float) meta_f(f, "dflash.attention.layer_norm_rms_epsilon");
    dfl_theta_ = (float) meta_f(f, "dflash.rope.freq_base");
    dfl_mask_ = (int32_t) meta_i(f, "tokenizer.ggml.mask_token_id");
    const int64_t n_layer = meta_i(f, "dflash.block_count");
    const int64_t block = meta_i(f, "dflash.block_size");
    if (dfl_ne_ != h_.n_embd || meta_i(f, "dflash.attention.head_count") != kHeads ||
        meta_i(f, "dflash.attention.head_count_kv") != kKvHeads || meta_i(f, "dflash.attention.key_length") != kHd ||
        meta_i(f, "dflash.conv_kernel_size") != 2 || meta_i(f, "dflash.selector_rank") != kRank ||
        meta_i(f, "dflash.selector_top_k") != kTopK || block < 2 || block > 16 || dfl_swa_ <= 0 ||
        dfl_swa_ > 2048 || dfl_ne_ % dfl_group_ != 0) {
        throw std::runtime_error("drafter: unsupported shape (this engine runs Qwen3.8-27B-DFlash2)");
    }
    // llama.cpp extracts the *inputs* of these target layers: the outputs of the ones before
    for (const int64_t l : f.int_array("dflash.target_layers")) {
        if (l < 1 || l > h_.n_layer) {
            throw std::runtime_error("drafter: target layer out of range");
        }
        dfl_capture_.push_back(l - 1);
    }
    dfl_.resize((size_t) n_layer);
    for (int64_t il = 0; il < n_layer; ++il) {
        const std::string p = "dflash.blk." + std::to_string(il) + ".";
        DflashLayer & L = dfl_[(size_t) il];
        L.attn_norm = resolve_f32(p + "attn_norm.weight");
        L.ffn_norm = resolve_f32(p + "ffn_norm.weight");
        L.q_norm = resolve_f32(p + "attn_q_norm.weight");
        L.k_norm = resolve_f32(p + "attn_k_norm.weight");
        L.attn_conv_base = resolve_f32(p + "attn_conv_base");
        L.ffn_conv_base = resolve_f32(p + "ffn_conv_base");
        L.q = resolve(p + "attn_q.weight");
        L.k = resolve(p + "attn_k.weight");
        L.v = resolve(p + "attn_v.weight");
        L.o = resolve(p + "attn_output.weight");
        L.attn_conv = resolve(p + "attn_conv_proj.weight");
        L.ffn_conv = resolve(p + "ffn_conv_proj.weight");
        L.gate = resolve(p + "ffn_gate.weight");
        L.up = resolve(p + "ffn_up.weight");
        L.down = resolve(p + "ffn_down.weight");
        if (L.attn_norm == nullptr || L.ffn_norm == nullptr || L.q_norm == nullptr || L.k_norm == nullptr ||
            L.attn_conv_base == nullptr || L.ffn_conv_base == nullptr || L.q.gemv == nullptr ||
            L.k.gemv == nullptr || L.v.gemv == nullptr || L.o.gemv == nullptr || L.attn_conv.gemv == nullptr ||
            L.ffn_conv.gemv == nullptr || L.gate.gemv == nullptr || L.up.gemv == nullptr || L.down.gemv == nullptr) {
            throw std::runtime_error("drafter: incomplete layer " + p);
        }
    }
    dfl_fc_ = resolve("dflash.fc.weight");
    dfl_gate_w_ = resolve("dflash.selector_hidden.weight");
    dfl_enc_norm_ = resolve_f32("dflash.enc.output_norm.weight");
    dfl_out_norm_ = resolve_f32("dflash.output_norm.weight");
    const Mat prev = resolve("dflash.selector_predecessor.weight");
    const Mat next = resolve("dflash.selector_successor.weight");
    if (dfl_fc_.gemv == nullptr || dfl_gate_w_.gemv == nullptr || dfl_enc_norm_ == nullptr ||
        dfl_out_norm_ == nullptr || prev.gemv == nullptr || next.gemv == nullptr || prev.gemv->type != 12 ||
        next.gemv->type != 12 || prev.gemv->k != kRank || prev.gemv->rows != h_.n_vocab ||
        (int64_t) dfl_capture_.size() * h_.n_embd != dfl_fc_.gemv->k) {
        throw std::runtime_error("drafter: fc / selector tensors missing or of another shape");
    }
    dfl_sel_prev_ = prev.dev;
    dfl_sel_next_ = next.dev;
    // buffers: the features of a forward's rows, the ring, the block's scratch
    const int64_t B = block;
    const int64_t kv = kKvHeads * kHd;
    const int64_t feat_rows = std::max<int64_t>(max_tokens_, 16);
    dfl_feat_ = mem_.device((size_t) feat_rows * dfl_fc_.gemv->k * 2, "out of VRAM (drafter features)");
    dfl_ring_bytes_ = (size_t) n_layer * dfl_swa_ * kv * 2;
    dfl_ring_k_ = mem_.device(dfl_ring_bytes_, "out of VRAM (drafter KV)");
    dfl_ring_v_ = mem_.device(dfl_ring_bytes_, "out of VRAM (drafter KV)");
    dfl_tags_ = static_cast<int32_t *>(mem_.device((size_t) dfl_swa_ * 4, "out of VRAM (drafter KV)"));
    (void) hipMemset(dfl_tags_, 0xFF, (size_t) dfl_swa_ * 4);
    dfl_bk_ = mem_.device((size_t) B * kv * 4);
    dfl_bv_ = mem_.device((size_t) B * kv * 4);
    dfl_dyn_ = mem_.device((size_t) B * (dfl_ne_ / dfl_group_) * 4 * 4);
    dfl_sgate_ = mem_.device((size_t) B * kRank * 4);
    dfl_attn_ws_ = mem_.device((size_t) omph::kernels::dflash_attention_ws_floats(dfl_swa_) * 4);
    dfl_logits_ = mem_.device((size_t) (B - 1) * h_.n_vocab * 4, "out of VRAM (drafter logits)");
    dfl_ids_ = static_cast<int32_t *>(mem_.device((size_t) (B - 1) * kTopK * 4));
    dfl_vals_ = static_cast<float *>(mem_.device((size_t) (B - 1) * kTopK * 4));
    dfl_scores_ = static_cast<float *>(mem_.device((size_t) (B - 1) * kTopK * kTopK * 4));
    dfl_keep_.assign((size_t) B, 1.0);  // optimistic start: every position drafted
    dfl_block_ = B;
}

void Runner::dflash_observe(const int64_t kept_drafts) {
    for (int64_t n = 1; n <= dfl_drafted_; ++n) {
        double & e = dfl_keep_[(size_t) n];
        e += ((kept_drafts >= n ? 1.0 : 0.0) - e) / 16.0;
    }
}

bool Runner::dflash_inject(const int64_t pos0, const int64_t T) {
    if (!dflash_on() || T <= 0) {
        return true;
    }
    const int64_t ne = h_.n_embd;
    const int64_t kv = kKvHeads * kHd;
    // g = enc_norm(fc(features)) as f16 into h16_
    if (!matmul(dfl_fc_, dfl_feat_, static_cast<float *>(cur_), dfl_ne_, dfl_fc_.gemv->k, T) ||
        !omph::kernels::add_rms_norm_f16(static_cast<const float *>(cur_), nullptr, nullptr, dfl_enc_norm_, h16_, T,
                                         dfl_ne_, dfl_eps_, nullptr)) {
        return fail("drafter: fc failed");
    }
    (void) ne;
    auto * k32 = static_cast<float *>(q_);
    auto * v32 = static_cast<float *>(gate_);
    for (size_t l = 0; l < dfl_.size(); ++l) {
        const DflashLayer & L = dfl_[l];
        auto * rk = static_cast<__half *>(dfl_ring_k_) + l * dfl_swa_ * kv;
        auto * rv = static_cast<__half *>(dfl_ring_v_) + l * dfl_swa_ * kv;
        if (!matmul(L.k, h16_, k32, kv, dfl_ne_, T) || !matmul(L.v, h16_, v32, kv, dfl_ne_, T) ||
            !omph::kernels::dflash_norm_rope(k32, T, kKvHeads, L.k_norm, dfl_eps_, pos0, dfl_theta_, nullptr) ||
            !omph::kernels::dflash_ring_put(k32, T, kv, pos0, rk, nullptr, dfl_swa_, nullptr) ||
            !omph::kernels::dflash_ring_put(v32, T, kv, pos0, rv, l + 1 == dfl_.size() ? dfl_tags_ : nullptr,
                                            dfl_swa_, nullptr)) {
            return fail("drafter: KV injection failed");
        }
    }
    return true;
}

// One draft layer on the B block rows in x_ (f32), in place.
bool Runner::dflash_layer(const DflashLayer & L, const int64_t l, const int64_t B, const int64_t pos) {
    const int64_t ne = dfl_ne_;
    const int64_t kv = kKvHeads * kHd;
    const int64_t nq = kHeads * kHd;
    const int64_t ndyn = (ne / dfl_group_) * 4;
    auto * x = static_cast<float *>(x_);
    auto * a = static_cast<float *>(cur_);
    auto * b = static_cast<float *>(resid_);
    auto * c = static_cast<float *>(blk_);
    auto * q = static_cast<float *>(q_);
    auto * att = static_cast<float *>(gate_);
    auto * dyn = static_cast<float *>(dfl_dyn_);
    auto * bk = static_cast<float *>(dfl_bk_);
    auto * bv = static_cast<float *>(dfl_bv_);
    const auto * rk = static_cast<const __half *>(dfl_ring_k_) + l * dfl_swa_ * kv;
    const auto * rv = static_cast<const __half *>(dfl_ring_v_) + l * dfl_swa_ * kv;
    using namespace omph::kernels;
    // attention: a = attn_norm(x), dyn = attn_conv_proj(a), b = conv(a, side 0)
    if (!rms_norm(x, L.attn_norm, a, B, ne, dfl_eps_, 1.0f, nullptr) || !cast_f32_to_f16(a, h16_, B * ne, nullptr) ||
        !matmul(L.attn_conv, h16_, dyn, ndyn, ne, B) ||
        !dflash_conv(a, dyn, L.attn_conv_base, B, ne, dfl_group_, 0, b, nullptr) ||
        !cast_f32_to_f16(b, h16_, B * ne, nullptr) || !matmul(L.q, h16_, q, nq, ne, B) ||
        !matmul(L.k, h16_, bk, kv, ne, B) || !matmul(L.v, h16_, bv, kv, ne, B) ||
        !dflash_norm_rope(q, B, kHeads, L.q_norm, dfl_eps_, pos, dfl_theta_, nullptr) ||
        !dflash_norm_rope(bk, B, kKvHeads, L.k_norm, dfl_eps_, pos, dfl_theta_, nullptr) ||
        !dflash_attention(q, rk, rv, dfl_tags_, dfl_swa_, bk, bv, B, pos, dfl_swa_, 1.0f / std::sqrt((float) kHd),
                          static_cast<float *>(dfl_attn_ws_), att, nullptr) ||
        !cast_f32_to_f16(att, ffn16_, B * nq, nullptr) || !matmul(L.o, ffn16_, c, ne, nq, B) ||
        !dflash_conv(c, dyn, L.attn_conv_base, B, ne, dfl_group_, 1, a, nullptr) ||
        !add_out(a, x, b, B * ne, nullptr)) {  // b = ffn_inp
        return fail("drafter: attention failed");
    }
    // FFN: a = ffn_norm(b), dyn = ffn_conv_proj(a), c = conv(a, side 0), swiglu, down, conv(side 1)
    if (!rms_norm(b, L.ffn_norm, a, B, ne, dfl_eps_, 1.0f, nullptr) || !cast_f32_to_f16(a, h16_, B * ne, nullptr) ||
        !matmul(L.ffn_conv, h16_, dyn, ndyn, ne, B) ||
        !dflash_conv(a, dyn, L.ffn_conv_base, B, ne, dfl_group_, 0, c, nullptr) ||
        !cast_f32_to_f16(c, h16_, B * ne, nullptr) ||
        !matmul(L.gate, h16_, static_cast<float *>(ffn1_), dfl_ff_, ne, B) ||
        !matmul(L.up, h16_, static_cast<float *>(ffn2_), dfl_ff_, ne, B) ||
        !swiglu_f16(static_cast<const float *>(ffn1_), static_cast<const float *>(ffn2_), ffn16_, B * dfl_ff_,
                    nullptr) ||
        !matmul(L.down, ffn16_, a, ne, dfl_ff_, B) ||
        !dflash_conv(a, dyn, L.ffn_conv_base, B, ne, dfl_group_, 1, c, nullptr) ||
        !add_out(c, b, x, B * ne, nullptr)) {
        return fail("drafter: FFN failed");
    }
    return true;
}

bool Runner::dflash_draft(const int32_t token, const int64_t pos, const int64_t k, std::vector<int32_t> & drafts) {
    drafts.clear();
    const int64_t B = dfl_block_;
    int64_t n = std::min<int64_t>(k, B - 1);
    if (!dflash_on() || n <= 0) {
        return dflash_on() || fail("drafter: not loaded");
    }
    // adaptive length: the positions still worth their verification
    if (env_.dflash_keep > 0.0f && ++dfl_steps_ % 8 != 0) {
        int64_t m = 1;
        while (m < n && dfl_keep_[(size_t) m + 1] >= env_.dflash_keep) {
            ++m;
        }
        n = m;
    }
    dfl_drafted_ = n;
    const int64_t ne = dfl_ne_;
    std::vector<int32_t> toks((size_t) B, dfl_mask_);
    toks[0] = token;
    if (!embed(toks.data(), B, nullptr, static_cast<float *>(x_))) {
        return false;
    }
    for (size_t l = 0; l < dfl_.size(); ++l) {
        if (!dflash_layer(dfl_[l], (int64_t) l, B, pos)) {
            return false;
        }
    }
    // final norm, the selector gate and the MASK rows' logits through the target head
    using namespace omph::kernels;
    const int64_t P = B - 1;
    // The candidates come from the first draft_vocab token ids while the recent
    // text stays inside them, as MTP's drafts (#217): the head's Q4_K tiles hold
    // those rows first, so one tile GEMV reads that fraction of the 682 MiB.
    const GemvEntry * hg = head_.gemv;
    const int64_t nv = env_.draft_vocab > 0 && env_.draft_vocab < h_.n_vocab && env_.draft_vocab % 16 == 0 &&
                               use_gemv_ && draft_oov_ < kDraftOovMax && hg != nullptr && hg->type == 12 &&
                               hg->rows == h_.n_vocab
                           ? env_.draft_vocab
                           : h_.n_vocab;
    const void * hx = static_cast<const uint8_t *>(h16_) + ne * 2;
    if (!rms_norm(static_cast<const float *>(x_), dfl_out_norm_, static_cast<float *>(cur_), B, ne, dfl_eps_, 1.0f,
                  nullptr) ||
        !cast_f32_to_f16(static_cast<const float *>(cur_), h16_, B * ne, nullptr) ||
        !matmul(dfl_gate_w_, h16_, static_cast<float *>(dfl_sgate_), kRank, ne, B) ||
        !(nv < h_.n_vocab
              ? gemv_tokens(12, head_.dev, hx, static_cast<float *>(dfl_logits_), nv, ne, (int) P, nullptr)
              : matmul(head_, hx, static_cast<float *>(dfl_logits_), h_.n_vocab, ne, P)) ||
        !dflash_topk(static_cast<const float *>(dfl_logits_), P, nv, dfl_ids_, dfl_vals_, nullptr) ||
        !dflash_selector(dfl_ids_, dfl_vals_, static_cast<const float *>(dfl_sgate_), token, dfl_sel_prev_,
                         dfl_sel_next_, h_.n_vocab, P, dfl_scores_, nullptr)) {
        return fail("drafter: head / selector failed");
    }
    std::vector<int32_t> ids((size_t) (P * kTopK));
    std::vector<float> scores((size_t) (P * kTopK * kTopK));
    if (hipMemcpy(ids.data(), dfl_ids_, ids.size() * 4, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(scores.data(), dfl_scores_, scores.size() * 4, hipMemcpyDeviceToHost) != hipSuccess) {
        return fail("drafter: lattice copy failed");
    }
    // the selector's greedy path from the anchor (llama.cpp: predecessor 0 at position 1)
    int64_t pred = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float * sc = scores.data() + (i * kTopK + pred) * kTopK;
        int64_t best = 0;
        for (int64_t j = 1; j < kTopK; ++j) {
            if (sc[j] > sc[best]) {
                best = j;
            }
        }
        if (env_.dflash_pmin > 0.0f) {  // softmax of the scores at the best: 1 / sum exp(s - s_best)
            float sum = 0.0f;
            for (int64_t j = 0; j < kTopK; ++j) {
                sum += std::exp(sc[j] - sc[best]);
            }
            if (1.0f / sum < env_.dflash_pmin) {
                break;
            }
        }
        pred = best;
        drafts.push_back(ids[(size_t) (i * kTopK + pred)]);
    }
    return true;
}

} // namespace omph::model
