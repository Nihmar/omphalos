// GGUF -> .omph conversion (#178, PLAN.md §8.4-8.5).
//
// The .omph file is the GGUF container with the magic "OMPH" (format/gguf.hh
// reads both): the source's metadata KV block copied verbatim (hyperparameters,
// tokenizer, chat template), then four keys: omph.format_version,
// omph.source_sha256, omph.tensor_layouts and omph.tensor_bytes. Every tensor
// a fused kernel reads is stored in format/repack.hh's layout, checked bit for
// bit by rebuilding the GGUF bytes from it; the rest keep the GGUF bytes (the
// token embedding too: the engine gathers its rows from host memory as they
// are). Tensor data start on 256-byte boundaries.
#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace omph::format {

struct ConvertStats {
    uint64_t tensors = 0;
    uint64_t repacked = 0;    // tensors stored in a repacked layout
    uint64_t bytes_in = 0;    // the GGUF file
    uint64_t bytes_out = 0;   // the .omph file
    std::string source_sha256;
};

// Writes `out_path` from the GGUF at `in_path`. Throws std::runtime_error on
// any failure (an input that is already .omph, a repack that does not rebuild
// the GGUF bytes exactly, I/O). `progress(done, total)` follows the tensors.
ConvertStats convert_to_omph(const std::string & in_path, const std::string & out_path,
                             const std::function<void(uint64_t, uint64_t)> & progress = {});

} // namespace omph::format
