// Runner: the MTP draft head (blk.<n_layer>, #124; PLAN.md §12).
//
// As llama.cpp runs it (src/models/qwen35.cpp graph_mtp): at position p the
// block reads the pair (h_{p-1}, t_p), with h the target's output_norm(x) and
// h = 0 at p = 0:
//   x = eh_proj([enorm(embed(t_p)); hnorm(h_{p-1})])
//   x += attention(attn_norm(x))          (its own KV cache)
//   x += ffn(post_attention_norm(x))
//   g = shared_head_norm(x),  logits = output.weight g
// A second draft feeds g back as h. After every forward the runner keeps, the
// MTP KV gets the kept positions' pairs (the fill: K / V only).
#include "model/runner.hh"

#include "kernels/attn.hh"
#include "kernels/dequant.hh"
#include "kernels/elementwise.hh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace omph::model {

bool Runner::mtp_block(const int32_t * toks, const float * h_in, const int64_t pos0,
                       const int64_t T, const bool kv_only, int32_t * argmax,
                       std::vector<float> * logits, const ForwardInputs * in) {
    const int64_t ne = h_.n_embd;
    if (!mtp_ || T <= 0 || T > max_tokens_ || pos0 + T > max_seq_) {
        return fail("mtp: not enabled, or tokens out of range");
    }
    // embeddings into cur_, then [enorm(e); hnorm(h)] per token into ffn16_ (f16)
    auto * cat = static_cast<__half *>(ffn16_);
    if (!embed(toks, T, in, static_cast<float *>(cur_))) {  // an image row: its embedding (#160)
        return false;
    }
    for (int64_t t = 0; t < T; ++t) {
        float * e = static_cast<float *>(cur_) + t * ne;
        if (!omph::kernels::add_rms_norm_f16(e, nullptr, nullptr, mtp_enorm_, cat + t * 2 * ne,
                                             1, ne, (float) h_.eps, nullptr) ||
            !omph::kernels::add_rms_norm_f16(h_in + t * ne, nullptr, nullptr, mtp_hnorm_,
                                             cat + t * 2 * ne + ne, 1, ne, (float) h_.eps,
                                             nullptr)) {
            return fail("mtp: input norms failed");
        }
    }
    if (!matmul(mtp_eh_, cat, static_cast<float *>(x_), ne, 2 * ne, T) ||
        !omph::kernels::add_rms_norm_f16(static_cast<const float *>(x_), nullptr, nullptr,
                                         mtp_L_.attn_norm, h16_, T, ne, (float) h_.eps,
                                         nullptr) ||
        !attn_layer(h_.n_layer, mtp_L_, T, pos0, kv_only)) {
        return fail("mtp: attention failed");
    }
    if (kv_only) {
        return true;
    }
    // x = ffn(post_norm(x + attn)) + (x + attn), as a stack layer
    const LayerWeights & L = mtp_L_;
    if (!omph::kernels::add_rms_norm_f16(static_cast<const float *>(blk_),
                                         static_cast<const float *>(x_),
                                         static_cast<float *>(resid_), L.post_norm, h16_, T, ne,
                                         (float) h_.eps, nullptr) ||
        !(grouped(h16_, ne, T, {{&L.ffn_up, ffn2_, h_.n_ff}, {&L.ffn_gate, ffn1_, h_.n_ff}}) ||
          fork_join(
            T,
            [&] { return matmul(L.ffn_up, h16_, static_cast<float *>(ffn2_), h_.n_ff, ne, T); },
            [&] {
                return matmul(L.ffn_gate, h16_, static_cast<float *>(ffn1_), h_.n_ff, ne, T);
            })) ||
        !omph::kernels::swiglu_f16(static_cast<const float *>(ffn1_),
                                   static_cast<const float *>(ffn2_), ffn16_, T * h_.n_ff,
                                   nullptr) ||
        !matmul(L.ffn_down, ffn16_, static_cast<float *>(cur_), ne, h_.n_ff, T) ||
        !omph::kernels::add_out(static_cast<const float *>(cur_),
                                static_cast<const float *>(resid_), static_cast<float *>(x_),
                                T * ne, nullptr)) {
        return fail("mtp: ffn failed");
    }
    if (argmax == nullptr && logits == nullptr) {
        return true;
    }
    // g = shared_head_norm(x) of the last row: the draft's logits, and the h of
    // a chained draft
    const float * xl = static_cast<const float *>(x_) + (T - 1) * ne;
    if (!omph::kernels::rms_norm(xl, mtp_head_norm_, static_cast<float *>(mtp_g_), 1, ne,
                                 (float) h_.eps, 1.0f, nullptr) ||
        !omph::kernels::cast_f32_to_f16(static_cast<const float *>(mtp_g_), h16_, ne, nullptr) ||
        !matmul(head_, h16_, static_cast<float *>(logits_), h_.n_vocab, ne, 1)) {
        return fail("mtp: head failed");
    }
    if (logits != nullptr) {
        const size_t at = logits->size();
        logits->resize(at + (size_t) h_.n_vocab);
        if (hipMemcpy(logits->data() + at, logits_, (size_t) h_.n_vocab * 4,
                      hipMemcpyDeviceToHost) != hipSuccess) {
            return fail("mtp: logits copy failed");
        }
    }
    if (argmax != nullptr) {
        unsigned long long key = 0;
        if (!omph::kernels::argmax_f32(static_cast<const float *>(logits_), h_.n_vocab,
                                       static_cast<unsigned long long *>(argmax_key_), nullptr) ||
            hipMemcpy(&key, argmax_key_, sizeof(key), hipMemcpyDeviceToHost) != hipSuccess) {
            return fail("mtp: argmax failed");
        }
        *argmax = omph::kernels::argmax_key_index(key);
    }
    return true;
}

bool Runner::mtp_fill(const int64_t keep) {
    const int64_t ne = h_.n_embd;
    if (keep < 1 || keep > (int64_t) last_toks_.size()) {
        return fail("mtp fill: nothing to keep");
    }
    // h rows of the pairs: [h_{pos0-1}; h_{pos0} .. h_{pos0+keep-2}]
    const size_t row = (size_t) ne * 4;
    auto * hin = static_cast<uint8_t *>(mtp_hin_);
    if (hipMemcpy(hin, mtp_pending_, row, hipMemcpyDeviceToDevice) != hipSuccess ||
        (keep > 1 && hipMemcpy(hin + row, mtp_hlast_, row * (size_t) (keep - 1),
                               hipMemcpyDeviceToDevice) != hipSuccess) ||
        hipMemcpy(mtp_pending_, static_cast<const uint8_t *>(mtp_hlast_) + row * (keep - 1),
                  row, hipMemcpyDeviceToDevice) != hipSuccess) {
        return fail("mtp fill: h copy failed");
    }
    // inside a forward: its image rows and M-RoPE positions (the rows of
    // [0, keep) are the forward's)
    return mtp_block(last_toks_.data(), static_cast<const float *>(mtp_hin_), last_pos0_, keep,
                     true, nullptr, nullptr, inputs_);
}

bool Runner::mtp_draft(const int32_t token, const int64_t pos, const int64_t k,
                       std::vector<int32_t> & drafts, std::vector<float> * logits) {
    drafts.clear();
    int32_t tok = token;
    const float * h = static_cast<const float *>(mtp_pending_);
    for (int64_t i = 0; i < k; ++i) {
        int32_t d = -1;
        if (!mtp_block(&tok, h, pos + i, 1, false, &d, logits)) {
            return false;
        }
        drafts.push_back(d);
        tok = d;
        h = static_cast<const float *>(mtp_g_);
    }
    return true;
}

} // namespace omph::model
