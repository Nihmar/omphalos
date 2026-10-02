#include "vision/encoder.hh"

#include "format/gguf.hh"

#include <llama.h>
#include <mtmd-helper.h>
#include <mtmd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace omph::vision {
namespace {

constexpr size_t kCacheBytes = size_t(512) << 20;

void quiet(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
}

} // namespace

uint64_t content_hash(const std::string & bytes) {
    uint64_t h = 1469598103934665603ull;
    for (const char c : bytes) {
        h ^= (uint8_t) c;
        h *= 1099511628211ull;
    }
    return h;
}

Encoder::Encoder(const std::string & mmproj, const std::string & model, const int threads) {
    llama_log_set(quiet, nullptr);
    mtmd_log_set(quiet, nullptr);
    mtmd_helper_log_set(quiet, nullptr);
    llama_backend_init();
    // mtmd needs a vocabulary; vocab_only reads the metadata, no tensors
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.n_gpu_layers = 0;
    vocab_ = llama_model_load_from_file(model.c_str(), mp);
    if (vocab_ == nullptr) {
        throw std::runtime_error("vision: cannot read the vocabulary of " + model);
    }
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = false;  // PLAN.md §3: the mmproj runs on the CPU
    cp.print_timings = false;
    cp.warmup = false;
    cp.n_threads = threads > 0 ? threads : (int) std::max(1u, std::thread::hardware_concurrency());
    ctx_ = mtmd_init_from_file(mmproj.c_str(), vocab_, cp);
    if (ctx_ == nullptr) {
        llama_model_free(vocab_);
        throw std::runtime_error("vision: cannot load " + mmproj);
    }
    if (!mtmd_support_vision(ctx_)) {
        mtmd_free(ctx_);
        llama_model_free(vocab_);
        throw std::runtime_error("vision: " + mmproj + " has no vision encoder");
    }
    // the projector's output width, which must be the text model's (a
    // vocab-only llama_model reports no embedding width)
    const omph::gguf::File proj(mmproj), text(model);
    uint64_t pd = 0, te = 0;
    const omph::gguf::Value * v = proj.find("clip.vision.projection_dim");
    const omph::gguf::Value * w = text.find("qwen35.embedding_length");
    if (v == nullptr || w == nullptr || !v->as_u64(pd) || !w->as_u64(te) || pd != te) {
        mtmd_free(ctx_);
        llama_model_free(vocab_);
        throw std::runtime_error("vision: the projector's width does not match the model's");
    }
    n_embd_ = (int64_t) pd;
}

Encoder::~Encoder() {
    if (ctx_ != nullptr) mtmd_free(ctx_);
    if (vocab_ != nullptr) llama_model_free(vocab_);
}

std::shared_ptr<const model::Image> Encoder::encode(const std::string & bytes) {
    const uint64_t hash = content_hash(bytes);
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
        if ((*it)->hash == hash) {
            cache_.splice(cache_.begin(), cache_, it);
            return cache_.front();
        }
    }
    const mtmd_helper_bitmap_wrapper wrapped = mtmd_helper_bitmap_init_from_buf(
        ctx_, reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size(), false,
        mtmd_helper_init_opt_default());
    mtmd_bitmap * bitmap = wrapped.bitmap;
    if (bitmap == nullptr || mtmd_bitmap_is_audio(bitmap) || wrapped.video_ctx != nullptr) {
        if (bitmap != nullptr) mtmd_bitmap_free(bitmap);
        if (wrapped.video_ctx != nullptr) mtmd_helper_video_free(wrapped.video_ctx);
        throw std::runtime_error("vision: cannot decode the image");
    }
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const char * marker = mtmd_default_marker();
    const mtmd_input_text text{marker, std::strlen(marker), false, true};
    const mtmd_bitmap * bitmaps[] = {bitmap};
    const int32_t rc = mtmd_tokenize(ctx_, chunks, &text, bitmaps, 1);
    mtmd_bitmap_free(bitmap);
    auto image = std::make_shared<model::Image>();
    std::string err;
    if (rc != 0) {
        err = "vision: cannot preprocess the image";
    }
    for (size_t i = 0; err.empty() && i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
        const mtmd_image_tokens * tokens = mtmd_input_chunk_get_tokens_image(chunk);
        const auto n = (int64_t) mtmd_image_tokens_get_n_tokens(tokens);
        // the grid from mtmd's own decoder positions, and the check that the
        // Generator's layout (row i at x = i % nx, y = i / nx) is mtmd's
        for (int64_t i = 0; i < n; ++i) {
            const mtmd_decoder_pos p = mtmd_image_tokens_get_decoder_pos(tokens, 0, (size_t) i);
            image->nx = std::max(image->nx, (int) p.x + 1);
            image->ny = std::max(image->ny, (int) p.y + 1);
        }
        bool layout = n == (int64_t) image->nx * image->ny;
        for (int64_t i = 0; layout && i < n; ++i) {
            const mtmd_decoder_pos p = mtmd_image_tokens_get_decoder_pos(tokens, 0, (size_t) i);
            layout = p.t == 0 && (int64_t) p.x == i % image->nx && (int64_t) p.y == i / image->nx;
        }
        if (!layout) {
            err = "vision: unexpected image token layout";
        } else if (mtmd_encode_chunk(ctx_, chunk) != 0) {
            err = "vision: encoding failed";
        } else {
            const float * e = mtmd_get_output_embd(ctx_);
            image->embd.assign(e, e + n * n_embd_);
        }
    }
    mtmd_input_chunks_free(chunks);
    if (err.empty() && image->embd.empty()) {
        err = "vision: no image in the input";
    }
    if (!err.empty()) {
        throw std::runtime_error(err);
    }
    image->hash = hash;
    cache_.push_front(image);
    cache_bytes_ += image->embd.size() * 4;
    while (cache_bytes_ > kCacheBytes && cache_.size() > 1) {
        cache_bytes_ -= cache_.back()->embd.size() * 4;
        cache_.pop_back();
    }
    return image;
}

} // namespace omph::vision
