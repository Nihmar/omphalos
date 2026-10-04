// The shapes this engine's kernels hard-code for the one model it runs
// (PLAN.md): the attention's query heads per KV head and head_dim, the
// delta-net state size, the conv kernel's window. The kernels use them as
// compile-time constants and their wrappers reject anything else; the model
// loader (model/hparams.cc) checks the metadata against the same constants, so
// a foreign GGUF is refused at load with a message that names the field,
// instead of failing on the first layer that runs.
#pragma once

namespace omph::kernels {

constexpr int kGqaGroup = 6;   // query heads per KV head (24 / 4), attn.hip
constexpr int kHeadDim = 256;  // attention head_dim (kPrepHd, kGqaHd), attn.hip
constexpr int kDeltaS = 128;   // delta-net state size, gdn.hip
constexpr int kMaxConv = 4;    // the conv kernel's window, gdn.hip

} // namespace omph::kernels
