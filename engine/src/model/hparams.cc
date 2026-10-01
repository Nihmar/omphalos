#include "model/hparams.hh"

#include <stdexcept>
#include <string>

namespace omph::model {

namespace {

bool meta_int(const omph::gguf::File & f, const char * key, int64_t & out) {
    const omph::gguf::Value * v = f.find(key);
    uint64_t u = 0;
    if (v == nullptr || !v->as_u64(u)) {
        return false;
    }
    out = (int64_t) u;
    return true;
}

double meta_float(const omph::gguf::File & f, const char * key, const double fallback) {
    const omph::gguf::Value * v = f.find(key);
    if (v == nullptr) {
        return fallback;
    }
    if (v->type == omph::gguf::ValueType::FLOAT32 || v->type == omph::gguf::ValueType::FLOAT64) {
        return v->f;
    }
    uint64_t u = 0;
    return v->as_u64(u) ? (double) u : fallback;
}

} // namespace

int64_t numel(const omph::gguf::TensorInfo & t) {
    int64_t n = 1;
    for (const uint64_t d : t.ne) {
        n *= (int64_t) d;
    }
    return n;
}

HParams read_hparams(const omph::gguf::File & f) {
    HParams h;
    meta_int(f, "qwen35.embedding_length", h.n_embd);
    meta_int(f, "qwen35.block_count", h.n_layer);
    meta_int(f, "qwen35.attention.head_count", h.n_head);
    meta_int(f, "qwen35.attention.head_count_kv", h.n_head_kv);
    meta_int(f, "qwen35.attention.key_length", h.head_dim);
    meta_int(f, "qwen35.rope.dimension_count", h.n_rot);
    meta_int(f, "qwen35.feed_forward_length", h.n_ff);
    meta_int(f, "qwen35.ssm.group_count", h.ssm_n_kh);
    meta_int(f, "qwen35.ssm.time_step_rank", h.ssm_n_vh);
    meta_int(f, "qwen35.ssm.state_size", h.ssm_s);
    meta_int(f, "qwen35.ssm.inner_size", h.ssm_inner);
    meta_int(f, "qwen35.ssm.conv_kernel", h.ssm_conv_k);
    h.eps = meta_float(f, "qwen35.attention.layer_norm_rms_epsilon", 1e-6);
    h.freq_base = meta_float(f, "qwen35.rope.freq_base", 10000.0);
    // the vocab comes from the embedding table; trailing MTP blocks (the
    // nextn.* group, ignored by the normal decode path) are not part of the
    // stack we run here
    if (const omph::gguf::TensorInfo * te = f.tensor("token_embd.weight")) {
        h.n_vocab = (int64_t) te->ne[1];
    }
    while (h.n_layer > 0 &&
           f.tensor("blk." + std::to_string(h.n_layer - 1) + ".nextn.eh_proj.weight") !=
               nullptr) {
        --h.n_layer;
    }
    if (h.n_embd <= 0 || h.n_layer <= 0 || h.n_vocab <= 0 || h.n_head <= 0 ||
        h.n_head_kv <= 0 || h.head_dim <= 0 || h.n_rot <= 0 || h.n_ff <= 0 ||
        h.ssm_n_kh <= 0 || h.ssm_n_vh <= 0 || h.ssm_s <= 0 || h.ssm_inner <= 0 ||
        h.ssm_conv_k <= 1) {
        throw std::runtime_error("incomplete hyperparameters");
    }
    // What the kernels hard-code for this model (single-model engine).
    const omph::gguf::TensorInfo * out = f.tensor("output.weight");
    if (h.head_dim != 256 || h.n_rot > h.head_dim || h.n_head % h.n_head_kv != 0 ||
        h.ssm_s != 128 || h.ssm_inner % h.ssm_n_vh != 0 || h.ssm_n_vh % h.ssm_n_kh != 0 ||
        out == nullptr || out->ne.size() < 2 || (int64_t) out->ne[1] != h.n_vocab) {
        throw std::runtime_error("hyperparameters outside what the kernels support");
    }
    return h;
}

} // namespace omph::model
