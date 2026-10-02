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
    bool speculative = true;   // MTP drafts when greedy and the model has them
    std::vector<int32_t> stop;  // extra stop tokens (the end-of-generation ones always stop)
};

struct GenerateResult {
    enum class Stop { Length, EndOfGeneration, StopToken, Callback, ContextFull, Error };
    std::vector<int32_t> tokens;  // generated, including a final stop token
    Stop stop = Stop::Length;
    int64_t prompt_tokens = 0;
    int64_t cached_tokens = 0;    // of the prompt, already in the caches
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    int64_t drafted = 0;
    int64_t accepted = 0;
};

class Generator {
public:
    struct Config {
        std::string model;
        int64_t context = 8192;  // KV capacity
        int64_t chunk = 512;     // prefill chunk (activation buffers)
        bool mtp = true;         // load the MTP block for speculative decoding
        int64_t draft_k = 3;     // drafts per speculative step (#126: 3 is the best)
    };
    Generator(const Config & config, const omph::runtime::EnvOptions & env);

    const omph::text::Tokenizer & tokenizer() const { return *tokenizer_; }
    int64_t context() const { return config_.context; }

    // Generates after `prompt`. on_token(id) gets every generated token as it
    // is decided; returning false stops (Stop::Callback). The prompt continues
    // the cached sequence when it starts with it, else the caches restart.
    GenerateResult generate(const std::vector<int32_t> & prompt, const GenerateRequest & request,
                            const std::function<bool(int32_t)> & on_token);

private:
    bool is_eog(int32_t id) const;
    int32_t sample(const std::vector<float> & logits, const Sampling & s);
    bool feed(const std::vector<int32_t> & toks, int64_t from, std::vector<float> & last_logits);

    Config config_;
    omph::runtime::EnvOptions env_;
    std::unique_ptr<omph::gguf::File> file_;
    std::unique_ptr<omph::text::Tokenizer> tokenizer_;
    std::unique_ptr<Runner> runner_;
    std::vector<int32_t> eog_;
    std::vector<int32_t> seq_;  // the tokens whose KV / state are in the caches
    std::mt19937_64 rng_;
};

} // namespace omph::model
