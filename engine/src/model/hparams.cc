#include "model/hparams.hh"

#include <stdexcept>
#include <string>

#include "kernels/shapes.hh"

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
    h.arch = "qwen35";
    if (const omph::gguf::Value * a = f.find("general.architecture")) {
        std::string_view s;
        if (a->as_str(s)) h.arch = std::string(s);
    }
    // The prefix of every hyperparameter key is the architecture name. This
    // engine runs the qwen35 family; anything else is refused here, before a
    // single tensor is read, rather than by a kernel that cannot say why.
    if (h.arch != "qwen35" && h.arch != "qwen35moe") {
        throw std::runtime_error("unsupported architecture '" + h.arch +
                                 "' (this engine runs qwen35 and qwen35moe, #490)");
    }
    const bool moe = h.arch == "qwen35moe";
    const std::string p = h.arch + ".";
    meta_int(f, (p + "embedding_length").c_str(), h.n_embd);
    meta_int(f, (p + "block_count").c_str(), h.n_layer);
    meta_int(f, (p + "attention.head_count").c_str(), h.n_head);
    meta_int(f, (p + "attention.head_count_kv").c_str(), h.n_head_kv);
    meta_int(f, (p + "attention.key_length").c_str(), h.head_dim);
    meta_int(f, (p + "rope.dimension_count").c_str(), h.n_rot);
    meta_int(f, (p + "feed_forward_length").c_str(), h.n_ff);
    meta_int(f, (p + "ssm.group_count").c_str(), h.ssm_n_kh);
    meta_int(f, (p + "ssm.time_step_rank").c_str(), h.ssm_n_vh);
    meta_int(f, (p + "ssm.state_size").c_str(), h.ssm_s);
    meta_int(f, (p + "ssm.inner_size").c_str(), h.ssm_inner);
    meta_int(f, (p + "ssm.conv_kernel").c_str(), h.ssm_conv_k);
    if (moe) {
        meta_int(f, (p + "expert_count").c_str(), h.n_expert);
        meta_int(f, (p + "expert_used_count").c_str(), h.n_expert_used);
        meta_int(f, (p + "expert_feed_forward_length").c_str(), h.n_ff_exp);
        meta_int(f, (p + "expert_shared_feed_forward_length").c_str(), h.n_ff_shexp);
    }
    h.eps = meta_float(f, (p + "attention.layer_norm_rms_epsilon").c_str(), 1e-6);
    h.freq_base = meta_float(f, (p + "rope.freq_base").c_str(), 10000.0);
    if (f.find(p + "rope.dimension_sections") != nullptr) {
        const std::vector<int64_t> sec = f.int_array(p + "rope.dimension_sections");
        for (size_t i = 0; i < 3 && i < sec.size(); ++i) h.rope_sections[i] = (int) sec[i];
        // The engine maps the first three sections and gives every pair past
        // them position 0: the three must cover n_rot / 2 exactly and the
        // fourth must be 0 (Qwen3.8: 11 + 11 + 10 + 0 = 32 = 64 / 2), or an
        // image-free prompt would rotate differently than llama.cpp (#346).
        const int64_t rest = sec.size() > 3 ? sec[3] : 0;
        if (rest != 0 || (int64_t) h.rope_sections[0] + h.rope_sections[1] + h.rope_sections[2] !=
                             h.n_rot / 2) {
            throw std::runtime_error(
                "hparams: unsupported M-RoPE sections (t + h + w must be n_rot / 2 and the fourth 0)");
        }
    }
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
        h.n_head_kv <= 0 || h.head_dim <= 0 || h.n_rot <= 0 ||
        h.ssm_n_kh <= 0 || h.ssm_n_vh <= 0 || h.ssm_s <= 0 || h.ssm_inner <= 0 ||
        h.ssm_conv_k <= 1 || (moe ? (h.n_expert <= 0 || h.n_expert_used <= 0 ||
                                     h.n_expert_used > h.n_expert || h.n_ff_exp <= 0 ||
                                     h.n_ff_shexp <= 0)
                                 : h.n_ff <= 0)) {
        throw std::runtime_error("incomplete hyperparameters");
    }
    // What the kernels hard-code for this model (single-model engine): the
    // constants in kernels/shapes.hh, which every kernel wrapper checks at
    // launch too. Checking them here names the field that is wrong and refuses
    // before 11 GiB of tensors are read, instead of failing on the first layer
    // that runs with a message that does not say why.
    const omph::gguf::TensorInfo * out = f.tensor("output.weight");
    const auto refuse = [&h](const std::string & what) {
        throw std::runtime_error("unsupported model: " + what +
                                 " (this engine runs qwen35 and qwen35moe, #490)");
    };
    if (h.head_dim != omph::kernels::kHeadDim || h.n_rot > h.head_dim || h.n_rot % 2 != 0) {
        refuse("attention head_dim " + std::to_string(h.head_dim) + " with rope " +
               std::to_string(h.n_rot) + ": the kernels need head_dim " +
               std::to_string(omph::kernels::kHeadDim) + " and an even rope");
    }
    if (h.n_head % h.n_head_kv != 0) {
        refuse("attention " + std::to_string(h.n_head) + " query heads / " +
               std::to_string(h.n_head_kv) + " kv heads: whole KV groups are required");
    }
    h.gqa_group = h.n_head / h.n_head_kv;
    if (h.gqa_group != omph::kernels::kGqaGroup && h.gqa_group != omph::kernels::kGqaGroupMoe) {
        refuse("attention " + std::to_string(h.n_head) + " query heads / " +
               std::to_string(h.n_head_kv) + " kv heads: the kernels have " +
               std::to_string(omph::kernels::kGqaGroup) + " and " +
               std::to_string(omph::kernels::kGqaGroupMoe) + " query heads per KV head");
    }
    if (h.ssm_s != omph::kernels::kDeltaS || h.ssm_inner % h.ssm_n_vh != 0 ||
        h.ssm_n_vh % h.ssm_n_kh != 0) {
        refuse("delta-net state " + std::to_string(h.ssm_s) + ", " +
               std::to_string(h.ssm_inner) + " values in " + std::to_string(h.ssm_n_vh) +
               " heads in " + std::to_string(h.ssm_n_kh) + " groups: the kernels need " +
               std::to_string(omph::kernels::kDeltaS) + " and whole groups");
    }
    if (h.ssm_conv_k > omph::kernels::kMaxConv) {
        refuse("conv window " + std::to_string(h.ssm_conv_k) + ": the kernel holds " +
               std::to_string(omph::kernels::kMaxConv));
    }
    if (out == nullptr || out->ne.size() < 2 || (int64_t) out->ne[1] != h.n_vocab) {
        refuse("output.weight and token_embd.weight disagree on the vocabulary (" +
               std::to_string(h.n_vocab) + ")");
    }
    return h;
}

} // namespace omph::model
