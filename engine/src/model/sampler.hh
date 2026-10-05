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

// The penalty window of a step (#298): the ids of the last `penalty_last_n`
// tokens of the sequence with their counts, kept sorted by id. reset() follows
// a sequence that only grew for O(1) per new token; the rows of a verification
// (whose windows differ by one token at each end) then cost one binary search
// each, instead of copying and sorting the window per row (#315).
class PenaltyWindow {
public:
    // The window after `seq`. A rebuild costs O(n log n) in the window's
    // length, so the plain decode path (one call per step) is what it was;
    // a new length or a sequence that did not only grow rebuilds.
    void reset(const std::vector<int32_t> & seq, int64_t penalty_last_n);
    // Moves to the next row of the same step: the sequence's part of the window
    // loses its oldest token, and the accepted draft `extra` joins it.
    void next(const std::vector<int32_t> & seq, int32_t extra);
    // llama.cpp's llama_sampler_penalties_apply over the window, token for
    // token; `row` is modified in place.
    void apply(float * row, int64_t nv, const Sampling & s) const;

private:
    struct Cnt {
        int32_t id;
        int32_t n;
    };
    std::vector<Cnt> v_;   // ascending by id, counts positive
    std::vector<int32_t> scratch_;
    std::vector<int32_t> added_;  // the drafts next() has added since reset()
    int64_t first_ = 0;    // seq index of the oldest token in the window
    int64_t end_ = 0;      // ... and one past its newest
    int64_t last_n_ = 0;
    // Drafts accepted so far in the step: they are the freshest tokens, and
    // while the sequence is shorter than penalty_last_n - extras they add to
    // the window instead of evicting a sequence token (#298).
    int64_t extras_ = 0;
    void add(int32_t id, int32_t d);
};

// The kept set without sorting the vocabulary (#197: the full sort of ~10^5
// candidates cost ~25 ms per row): top-k by selection, min-p by a threshold,
// top-p over a descending prefix sorted only as far as its mass needs. The
// kept tokens are in descending order when top-p applies, else in id order.
// `row` is modified in place when the penalties are on (the callers own their
// logits buffer).
void distribution(float * row, int64_t nv, const Sampling & s, const PenaltyWindow & pen, Dist & d);

// A draw from d without token `skip` (-1: none).
int32_t draw(const Dist & d, int32_t skip, std::mt19937_64 & rng);

// The greedy pick over one logits row: the largest finite value, the lowest id
// on a tie, and id 0 when the row has no finite value at all. Non-finite
// values are ignored the same way the device's packed argmax ignores them
// (kernels/elementwise.hip's argmax_key), so a corrupted row gives the two
// paths the same token instead of two different ones (#318; OMPH_CHECK_FINITE
// is what reports the corruption).
int32_t argmax_finite(const float * row, int64_t nv);

// The decision over one logits row: the penalties, then the argmax
// (temperature 0, llama.cpp applies them before the greedy pick too) or a
// sampled draw.
int32_t sample_row(std::vector<float> & logits, const Sampling & s, const std::vector<int32_t> & seq,
                   std::mt19937_64 & rng);

} // namespace omph::model
