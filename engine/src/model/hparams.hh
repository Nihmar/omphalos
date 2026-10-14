// The model's hyperparameters, read from the GGUF metadata and checked against
// what the kernels hard-code (the qwen35 family: Qwen3.8-27B dense, the
// qwen35moe expert models, #490).
#pragma once

#include "format/gguf.hh"

#include <cstdint>
#include <string>

namespace omph::model {

struct HParams {
    std::string arch;  // general.architecture (qwen35, qwen35moe)
    int64_t n_embd = 0;
    int64_t n_layer = 0;
    int64_t n_vocab = 0;
    int64_t n_head = 0;
    int64_t n_head_kv = 0;
    int64_t gqa_group = 0;  // query heads per KV head, n_head / n_head_kv
    int64_t head_dim = 0;
    int64_t n_rot = 0;
    int64_t n_ff = 0;  // dense FFN width (0 on an expert model: see n_ff_exp)
    int64_t ssm_n_kh = 0;
    int64_t ssm_n_vh = 0;
    int64_t ssm_s = 0;
    int64_t ssm_inner = 0;
    int64_t ssm_conv_k = 0;
    // Mixture of experts (qwen35moe; 0 on a dense model)
    int64_t n_expert = 0;       // routed experts per layer
    int64_t n_expert_used = 0;  // experts one token goes through
    int64_t n_ff_exp = 0;       // an expert's FFN width
    int64_t n_ff_shexp = 0;     // the shared expert's FFN width
    double eps = 1e-6;
    double freq_base = 10000.0;
    int rope_sections[3] = {0, 0, 0};  // M-RoPE t / h / w frequency pairs (#160)

    bool moe() const { return n_expert > 0; }
};

// Throws when a hyperparameter is missing or outside what the kernels support.
HParams read_hparams(const omph::gguf::File & f);

// Elements of a tensor.
int64_t numel(const omph::gguf::TensorInfo & t);

} // namespace omph::model
