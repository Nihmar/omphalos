// The host sampler (#298): the sampling distribution of one logits row -- the
// penalties of llama.cpp's sampler chain, then temperature, top-k, min-p and
// top-p -- and the draw from it. Pure: no GPU, no state beyond the caller's
// RNG, so the truncations are unit-tested (engine/tests/test_sampler.cc)
// instead of only through a generated sequence.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace omph::model {

struct Sampling {
    float temperature = 0.0f;  // 0: greedy
    int top_k = 0;             // 0: off
    float top_p = 1.0f;        // 1: off
    float min_p = 0.0f;        // 0: off
    uint64_t seed = 0;
    // OpenAI's penalties over the last `penalty_last_n` tokens of the
    // sequence, prompt included, plus the tokens already accepted in the step
    // being sampled (#298): a token that appeared is pushed down by
    // `presence_penalty`, one that appeared n times by n * `frequency_penalty`.
    // `penalty_last_n == 0` (or both penalties 0) is off; the penalty is
    // applied to the raw logits, before temperature and the truncations, as
    // llama.cpp's sampler chain does. Greedy (temperature 0) ignores them: its
    // argmax runs on the device.
    // llama.cpp's penalties sampler, all three over the same window: a token
    // in it is scaled by `repeat_penalty` (sign-aware: see apply_penalties)
    // and pushed down by `presence_penalty + n * frequency_penalty`.
    float repeat_penalty = 1.0f;   // 1.0: off
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    int penalty_last_n = 64;       // 0: off (llama.cpp's --repeat-last-n)
    bool penalizes() const {
        return penalty_last_n > 0 &&
               (repeat_penalty != 1.0f || presence_penalty != 0.0f || frequency_penalty != 0.0f);
    }
};

// The sampling distribution of one logits row: the kept tokens with
// unnormalized weights (in descending order when top-p applied, else in id
// order, as distribution() documents) and their total.
struct Dist {
    std::vector<int32_t> ids;
    std::vector<double> w;
    double total = 0.0;
};

// llama.cpp's llama_sampler_penalties_apply, token for token: the last
// `penalty_last_n` tokens of `seq` (the `n_extra` accepted drafts of the step
// being sampled are the freshest ones, so the window of the sequence shrinks
// accordingly), each with its count. `row` is modified in place.
void apply_penalties(float * row, int64_t nv, const Sampling & s, const std::vector<int32_t> & seq,
                     const int32_t * extra = nullptr, int64_t n_extra = 0);

// The kept set without sorting the vocabulary (#197: the full sort of ~10^5
// candidates cost ~25 ms per row): top-k by selection, min-p by a threshold,
// top-p over a descending prefix sorted only as far as its mass needs. The
// kept tokens are in descending order when top-p applies, else in id order.
// `row` is modified in place when the penalties are on (the callers own their
// logits buffer).
void distribution(float * row, int64_t nv, const Sampling & s, const std::vector<int32_t> & seq, Dist & d,
                  const int32_t * extra = nullptr, int64_t n_extra = 0);

// A draw from d without token `skip` (-1: none).
int32_t draw(const Dist & d, int32_t skip, std::mt19937_64 & rng);

// The decision over one logits row: the penalties, then the argmax
// (temperature 0, llama.cpp applies them before the greedy pick too) or a
// sampled draw.
int32_t sample_row(std::vector<float> & logits, const Sampling & s, const std::vector<int32_t> & seq,
                   std::mt19937_64 & rng);

} // namespace omph::model
