// Images to embeddings on the CPU (#160, PLAN.md M7): llama.cpp's mtmd with the
// GGUF mmproj, from a CPU-only llama.cpp build (no GPU backend in this
// process: the encoder never touches VRAM). Encoded images are cached by
// content in host RAM. Built when OMPH_LLAMA_DIR is given to CMake.
#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "model/generator.hh"

struct mtmd_context;
struct llama_model;

namespace omph::vision {

class Encoder {
public:
    // mmproj: the vision GGUF; model: the text GGUF (only its vocabulary is
    // read: mtmd tokenizes through it); threads: 0 = all hardware threads.
    Encoder(const std::string & mmproj, const std::string & model, int threads = 0);
    ~Encoder();
    Encoder(const Encoder &) = delete;
    Encoder & operator=(const Encoder &) = delete;

    // An image file's bytes (JPEG, PNG, BMP, GIF, ...) to its embeddings.
    // Throws std::runtime_error on undecodable input.
    std::shared_ptr<const model::Image> encode(const std::string & bytes);

    int64_t n_embd() const { return n_embd_; }

private:
    mtmd_context * ctx_ = nullptr;
    llama_model * vocab_ = nullptr;
    int64_t n_embd_ = 0;
    std::mutex mutex_;
    // the most recently used images, up to kCacheBytes of embeddings
    std::list<std::shared_ptr<const model::Image>> cache_;
    size_t cache_bytes_ = 0;
};

// FNV-1a of a byte string (the image cache key and token identity).
uint64_t content_hash(const std::string & bytes);

} // namespace omph::vision
