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
#include <cstdlib>
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
        meta_i(f, "dflash.selector_top_k") != kTopK || block < 2 || block > omph::kernels::kMaxBlock ||
        dfl_swa_ <= 0 ||
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
    // #417's follow-up: the attention window may be shortened (the ring's
    // allocation and the tags stay the model's); a shorter window reads fewer
    // ring rows per layer per step and measured >= the trained 2048 on every
    // text tried (bench/results/dflash-window-417.txt)
    dfl_swa_eff_ = env_.dflash_window > 0
                       ? std::min<int64_t>(std::max<int64_t>(env_.dflash_window, 16), dfl_swa_)
                       : dfl_swa_;
    // buffers: the features of a forward's rows, the ring, the block's scratch.
    // The scratch holds the longest block the engine may run (kMaxBlock); the
    // block itself is OMPH_DFLASH_MASKS + 1 (the trained block_size - 1 before
    // #414's measurement, 11 by default).
    const int64_t B = omph::kernels::kMaxBlock;
    // #433: a cascade chains whole *trained* blocks (block_size - 1 masks each);
    // a single block may be longer than training (#414)
    dfl_blocks_ = std::min<int64_t>(std::max<int64_t>(env_.dflash_blocks, 1), 2);
    const int64_t masks = dfl_blocks_ > 1
                              ? std::min<int64_t>(std::max<int64_t>(block - 1, 1), (B - 1) / dfl_blocks_)
                              : std::min<int64_t>(std::max<int64_t>(env_.dflash_masks, 1), B - 1);
    dfl_masks_ = masks;
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
    dfl_reach_.assign((size_t) B + 1, 0);  // #414 follow-up: per-position observed counts
    dfl_kept_.assign((size_t) B + 1, 0);
    dfl_block_ = dfl_blocks_ * masks + 1;  // the chain: 1 + every block's masks
}

// The per-position keep rates are EMAs over the observations (#245): the
// weight is how fast a position that stops being reached falls out of the
// drafted block (#416). At 1/16 a tail position needed ~32 steps to drop below
// the threshold, long after a short generation had ended, so on code the block
// stayed at its cap and the extra rows cost more than they accepted; 1/4 cuts
// within ~6 steps, which is what makes the cap worth having.
constexpr double kKeepEmaWeight = 4.0;

void Runner::dflash_observe(const int64_t kept_drafts) {
    for (int64_t n = 1; n <= dfl_drafted_; ++n) {
        double & e = dfl_keep_[(size_t) n];
        e += ((kept_drafts >= n ? 1.0 : 0.0) - e) / kKeepEmaWeight;
        if (n < (int64_t) dfl_reach_.size()) {
            ++dfl_reach_[(size_t) n];
            if (kept_drafts >= n) {
                ++dfl_kept_[(size_t) n];
            }
        }
    }
    if (std::getenv("OMPH_DFLASH_DEBUG") != nullptr) {
        std::fprintf(stderr, "per-position reached/kept:");
        for (size_t n = 1; n < dfl_reach_.size(); ++n) {
            if (dfl_reach_[n] == 0) {
                break;
            }
            std::fprintf(stderr, " %zu:%lld/%lld", n, (long long) dfl_kept_[n], (long long) dfl_reach_[n]);
        }
        std::fprintf(stderr, "\n");
    }
}

// #433: the cascade publishes a block's rows into the ring before the
// verification decides; the rows the verification rejects must not be read
// again. The tags are cleared for the whole verified range and dflash_inject
// re-tags the kept rows right after.
bool Runner::dflash_tags_invalidate(const int64_t pos0, const int64_t T) {
    if (dfl_blocks_ <= 1 || T <= 0) {
        return true;
    }
    return omph::kernels::dflash_tags_clear(dfl_tags_, pos0, T, dfl_swa_, nullptr);
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

// #433: the cascade's context. This block's rows 1..n-1 (the drafts) go into the
// ring at their positions, so the next block's queries see them; the
// verification overwrites the kept ones with the target's features and clears
// the tags of the rejected ones.
bool Runner::publish_block_rows(const int64_t l, const float * bk, const float * bv, const int64_t B,
                                const int64_t pos) {
    if (dfl_publish_ <= 0) {
        return true;
    }
    const int64_t kv = kKvHeads * kHd;
    const int64_t n = std::min(dfl_publish_, B - 1);
    auto * rk = static_cast<__half *>(dfl_ring_k_) + l * dfl_swa_ * kv;
    auto * rv = static_cast<__half *>(dfl_ring_v_) + l * dfl_swa_ * kv;
    if (!omph::kernels::dflash_ring_put(bk + kv, n, kv, pos + 1, rk, nullptr, dfl_swa_, nullptr) ||
        !omph::kernels::dflash_ring_put(bv + kv, n, kv, pos + 1, rv,
                                        l + 1 == (int64_t) dfl_.size() ? dfl_tags_ : nullptr, dfl_swa_, nullptr)) {
        return fail("drafter: cascade ring publish failed");
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
        !dflash_attention(q, rk, rv, dfl_tags_, dfl_swa_, bk, bv, B, pos, dfl_swa_eff_, 1.0f / std::sqrt((float) kHd),
                          static_cast<float *>(dfl_attn_ws_), att, nullptr) ||
        !publish_block_rows(l, bk, bv, B, pos) ||
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
    const int64_t Bmax = dfl_block_;  // the chain: 1 + every block's masks
    int64_t n = std::min<int64_t>(k, Bmax - 1);
    if (!dflash_on() || n <= 0) {
        dfl_drafted_ = 0;  // nothing drafted: the keep rates see no step (#340)
        return dflash_on() || fail("drafter: not loaded");
    }
    // Adaptive length: the *chain* is chosen, not the verification after it, so
    // the positions the drafts are not worth never cost their rows (#416). The
    // per-position keep rates are measured ones (dflash_observe), and every 8th
    // step drafts all of them, so a position that starts being kept is seen.
    if (env_.dflash_keep > 0.0f && ++dfl_steps_ % 8 != 0) {
        int64_t m = 1;
        while (m < n && dfl_keep_[(size_t) m + 1] >= env_.dflash_keep) {
            ++m;
        }
        n = m;
    }
    if (std::getenv("OMPH_DFLASH_DEBUG") != nullptr) {
        std::fprintf(stderr, "draft step %lld: cap %lld, n %lld, keep", (long long) dfl_steps_,
                     (long long) (Bmax - 1), (long long) n);
        for (size_t i = 1; i <= (size_t) (Bmax - 1); ++i) {
            std::fprintf(stderr, " %.2f", dfl_keep_[i]);
        }
        std::fprintf(stderr, "\n");
    }
    const int64_t ne = dfl_ne_;
    // The drafter's rows are at most 16: its matmuls take the NT GEMVs, not the
    // f16 GEMM path, which from gemm_min tokens would stage the 682 MiB head
    // (and the drafter's weights) as f16 (#414).
    struct ScopedDraftGemv {
        bool & flag;
        bool saved;
        explicit ScopedDraftGemv(bool & f) : flag(f), saved(f) { flag = true; }
        ~ScopedDraftGemv() { flag = saved; }
    } scoped_gemv(draft_gemv_);

    // One block: [anchor, MASK x P] through the five layers and the selector
    // (#245), P masks -> P drafts. The cascade runs it once per trained block.
    const auto block_pass = [&](const int32_t anchor, const int64_t p0, const int64_t P,
                                std::vector<int32_t> & out) -> bool {
        const int64_t Bl = P + 1;
        std::vector<int32_t> toks((size_t) Bl, dfl_mask_);
        toks[0] = anchor;
        if (!embed(toks.data(), Bl, nullptr, static_cast<float *>(x_))) {
            return false;
        }
        for (size_t l = 0; l < dfl_.size(); ++l) {
            if (!dflash_layer(dfl_[l], (int64_t) l, Bl, p0)) {
                return false;
            }
        }
        using namespace omph::kernels;
        const GemvEntry * hg = head_.gemv;
        const int64_t dv = draft_vocab_rows();  // OMPH_DRAFT_VOCAB in head tiles (#340)
        const int64_t nv = dv > 0 && dv < h_.n_vocab && use_gemv_ && draft_oov_ < kDraftOovMax && hg != nullptr &&
                                   hg->type == 12 && hg->rows == h_.n_vocab
                               ? dv
                               : h_.n_vocab;
        const void * hx = static_cast<const uint8_t *>(h16_) + ne * 2;  // the MASK rows
        if (!rms_norm(static_cast<const float *>(x_), dfl_out_norm_, static_cast<float *>(cur_), Bl, ne, dfl_eps_, 1.0f,
                      nullptr) ||
            !cast_f32_to_f16(static_cast<const float *>(cur_), h16_, Bl * ne, nullptr) ||
            !matmul(dfl_gate_w_, h16_, static_cast<float *>(dfl_sgate_), kRank, ne, Bl) ||
            !(nv < h_.n_vocab
                  ? gemv_tokens(12, head_.dev, hx, static_cast<float *>(dfl_logits_), nv, ne, (int) P, nullptr)
                  : matmul(head_, hx, static_cast<float *>(dfl_logits_), h_.n_vocab, ne, P)) ||
            !dflash_topk(static_cast<const float *>(dfl_logits_), P, nv, dfl_ids_, dfl_vals_, nullptr) ||
            !dflash_selector(dfl_ids_, dfl_vals_, static_cast<const float *>(dfl_sgate_), anchor, dfl_sel_prev_,
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
        out.clear();
        int64_t pred = 0;
        for (int64_t i = 0; i < P; ++i) {
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
            out.push_back(ids[(size_t) (i * kTopK + pred)]);
        }
        return true;
    };

    if (dfl_blocks_ <= 1) {
        if (!block_pass(token, pos, n, drafts)) {
            return false;
        }
    } else {
        // #433: whole trained blocks, each conditioned on the previous one's rows
        // (published into the ring by dflash_layer).
        int64_t left = n;
        int32_t anchor = token;
        int64_t p0 = pos;
        std::vector<int32_t> part;
        while (left > 0 && (int64_t) drafts.size() < n) {
            const int64_t m = std::min(dfl_masks_, left);
            dfl_publish_ = left > m ? m - 1 : 0;  // its rows feed the next block
            const bool ok = block_pass(anchor, p0, m, part);
            dfl_publish_ = 0;
            if (!ok) {
                return false;
            }
            drafts.insert(drafts.end(), part.begin(), part.end());
            left -= m;
            if (part.empty()) {
                break;
            }
            anchor = part.back();
            p0 += m;
        }
    }
    // What was really drafted, the OMPH_DFLASH_PMIN cut included: the previous
    // step's count or the full n would record those positions as rejections (#340).
    dfl_drafted_ = (int64_t) drafts.size();
    return true;
}

} // namespace omph::model
