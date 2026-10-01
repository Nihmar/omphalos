// The model's hyperparameters, read from the GGUF metadata and checked against
// what the kernels hard-code (single-model engine).
#pragma once

#include "format/gguf.hh"

#include <cstdint>

namespace omph::model {

struct HParams {
    int64_t n_embd = 0;
    int64_t n_layer = 0;
    int64_t n_vocab = 0;
    int64_t n_head = 0;
    int64_t n_head_kv = 0;
    int64_t head_dim = 0;
    int64_t n_rot = 0;
    int64_t n_ff = 0;
    int64_t ssm_n_kh = 0;
    int64_t ssm_n_vh = 0;
    int64_t ssm_s = 0;
    int64_t ssm_inner = 0;
    int64_t ssm_conv_k = 0;
    double eps = 1e-6;
    double freq_base = 10000.0;
};

// Throws when a hyperparameter is missing or outside what the kernels support.
HParams read_hparams(const omph::gguf::File & f);

// Elements of a tensor.
int64_t numel(const omph::gguf::TensorInfo & t);

} // namespace omph::model
