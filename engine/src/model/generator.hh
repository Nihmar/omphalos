// The engine's generation loop (#152): the model, its tokenizer, and one
// sequence in the caches. Chunked prefill (continuing the cached sequence when
// a prompt extends it), greedy decode with the device argmax or speculative
// MTP decoding, host sampling otherwise, stopping on the end-of-generation
// tokens. Shared by the C ABI, the server and omph-generate.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "format/gguf.hh"
#include "model/runner.hh"
#include "runtime/options.hh"
#include "text/tokenizer.hh"

namespace omph::model {

// An encoded image (#160): nx x ny embedding rows (n_embd floats each, row
// major), and its content hash (the identity of its positions in the cache).
struct Image {
    std::vector<float> embd;
    int nx = 0;
    int ny = 0;
    uint64_t hash = 0;
    int64_t n_tokens() const { return (int64_t) nx * ny; }
};

struct Sampling {
    float temperature = 0.0f;  // 0: greedy
    int top_k = 0;             // 0: off
    float top_p = 1.0f;        // 1: off
    float min_p = 0.0f;        // 0: off
    uint64_t seed = 0;
};

struct GenerateRequest {
    int64_t max_tokens = 256;
    Sampling sampling;
    bool speculative = true;   // MTP drafts when the model has them: greedy gets plain greedy's tokens,
                               // sampling plain sampling's distribution (speculative sampling, #197)
    std::vector<int32_t> stop;  // extra stop tokens (the end-of-generation ones always stop)
    // The images of the prompt, one per <|image_pad|> token, in order (#160).
    std::vector<std::shared_ptr<const Image>> images;
    std::vector<float> * prefill_logits = nullptr;  // validation: the logits after the prompt
    // validation: decode these tokens instead of choosing (one per step, no
    // speculation), appending every step's logits row (the prompt's first)
    const std::vector<int32_t> * force = nullptr;
    std::vector<float> * forced_logits = nullptr;
};

struct GenerateResult {
    enum class Stop { Length, EndOfGeneration, StopToken, Callback, ContextFull, Error };
    std::vector<int32_t> tokens;  // generated, including a final stop token
    Stop stop = Stop::Length;
    int64_t prompt_tokens = 0;    // with each image's tokens
    int64_t cached_tokens = 0;    // of the prompt, already in the caches
    bool restored = false;        // ... from a checkpoint (#158)
    double checkpoint_ms = 0.0;   // saving / restoring checkpoints (in prefill_ms)
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    int64_t drafted = 0;
    int64_t accepted = 0;
    int64_t ngram_steps = 0;      // verifications whose drafts came from n-gram lookup (#199)
};

class Generator {
public:
    struct Config {
        std::string model;
        int64_t context = 8192;  // KV capacity
        int64_t chunk = 512;     // prefill chunk (activation buffers)
        bool mtp = true;         // load the MTP block for speculative decoding
        int64_t draft_k = 3;     // drafts per speculative step (#126: 3 is the best)
        std::string dflash;      // a DFlash2 drafter .omph (#245): its drafts (block - 1 per step)
                                 // instead of the MTP block's, which is then not loaded
        int64_t cache_mib = 2048;  // host RAM for sequence checkpoints (#158); 0: none
    };
    Generator(const Config & config, const omph::runtime::EnvOptions & env);

    const omph::text::Tokenizer & tokenizer() const { return *tokenizer_; }
    int64_t context() const { return config_.context; }

    // Generates after `prompt`. on_token(id) gets every generated token as it
    // is decided; returning false stops (Stop::Callback). The prompt continues
    // the cached sequence when it starts with it, else the latest checkpoint
    // within their common prefix (#158), else the caches restart.
    GenerateResult generate(const std::vector<int32_t> & prompt, const GenerateRequest & request,
                            const std::function<bool(int32_t)> & on_token);
    // The request being generated, its counters so far (drafted, accepted):
    // valid inside on_token only (progress reports, #231).
    const GenerateResult & running() const { return *running_; }

private:
    const GenerateResult * running_ = nullptr;
    bool is_eog(int32_t id) const;
    int32_t sample(const std::vector<float> & logits, const Sampling & s);
    // The sampling distribution of one logits row (temperature, top-k, min-p,
    // top-p applied): the kept tokens with unnormalized weights, most likely
    // first, and their total. sample() draws from it.
    struct Dist {
        std::vector<int32_t> ids;
        std::vector<double> w;
        double total = 0.0;
    };
    void distribution(const float * row, int64_t nv, const Sampling & s, Dist & d) const;
    // A draw from d without token `skip` (-1: none).
    int32_t draw(const Dist & d, int32_t skip);
    // A prompt with its images expanded: an image's positions hold ids
    // derived from its hash (negative: never a vocabulary token), so prefix
    // reuse sees which image is where; M-RoPE positions when there are images.
    struct Expanded {
        std::vector<int32_t> tokens;
        std::vector<const Image *> image_of;  // per position: its image, or null
        std::vector<int64_t> image_row;       // per position: the row in its image
        std::vector<int32_t> mpos;            // 3 per position, or empty (text only)
        int64_t rope_end = 0;                 // the RoPE position after the prompt
    };
    bool expand(const std::vector<int32_t> & prompt, const GenerateRequest & req, Expanded & out) const;
    bool feed(const Expanded & p, int64_t from, std::vector<float> & last_logits,
              const std::vector<int64_t> & cuts, GenerateResult & res);
    // Sequence checkpoints (#158): the state after tokens[0, pos), in pinned
    // host RAM; valid while seq_ starts with `tokens`.
    struct Checkpoint {
        std::vector<int32_t> tokens;
        void * host = nullptr;
        uint64_t used = 0;  // LRU clock
    };
    bool valid(const Checkpoint & c) const;
    int64_t resume(const std::vector<int32_t> & prompt, GenerateResult & res);
    std::vector<int64_t> checkpoint_positions(const std::vector<int32_t> & prompt, int64_t from) const;
    bool save_checkpoint(const std::vector<int32_t> & prompt, int64_t pos);
    void drop_checkpoints(bool all);

    Config config_;
    omph::runtime::EnvOptions env_;
    std::unique_ptr<omph::gguf::File> file_;
    std::unique_ptr<omph::text::Tokenizer> tokenizer_;
    std::unique_ptr<Runner> runner_;
    std::vector<int32_t> eog_;
    std::vector<int32_t> seq_;  // the tokens whose KV / state are in the caches
    std::vector<Checkpoint> checkpoints_;
    std::vector<void *> spare_;  // host buffers of evicted checkpoints
    size_t max_checkpoints_ = 0;
    size_t n_buffers_ = 0;
    uint64_t clock_ = 0;
    int32_t im_start_ = -1;
    int32_t image_pad_ = -1;
    std::mt19937_64 rng_;
};

} // namespace omph::model
