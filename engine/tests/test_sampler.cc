// The host sampler (#298, #315) without a model: the penalties against a
// brute-force window, each truncation alone, min-p with top-p together -- which
// had no test, and did not compose: top-p read the mass min-p had removed --
// and the draw.
#include "check.hh"
#include "model/sampler.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using omph::model::Dist;
using omph::model::Sampling;

namespace {

// A row of `nv` logits whose first w.size() entries have the given
// unnormalized weights (logit = log w) and the rest are far below the cut.
std::vector<float> row_of(const std::vector<double> & w, const int64_t nv = 16) {
    std::vector<float> row((size_t) nv, -60.0f);
    for (size_t i = 0; i < w.size(); ++i) {
        row[i] = (float) std::log(w[i]);
    }
    return row;
}

std::vector<int32_t> sorted_ids(const Dist & d) {
    std::vector<int32_t> v(d.ids.begin(), d.ids.end());
    std::sort(v.begin(), v.end());
    return v;
}

bool same_ids(const Dist & d, const std::vector<int32_t> & want) { return sorted_ids(d) == want; }

// llama.cpp's llama_sampler_penalties_apply over a window given explicitly:
// what the incremental implementation must reproduce by construction.
void penalties_ref(float * row, const int64_t nv, const Sampling & s, std::vector<int32_t> win) {
    std::sort(win.begin(), win.end());
    for (size_t i = 0; i < win.size();) {
        size_t j = i;
        while (j < win.size() && win[j] == win[i]) {
            ++j;
        }
        const int32_t id = win[i];
        if (id >= 0 && id < nv) {
            if (s.repeat_penalty != 1.0f) {
                row[id] = row[id] <= 0.0f ? row[id] * s.repeat_penalty : row[id] / s.repeat_penalty;
            }
            row[id] -= s.presence_penalty + (float) (j - i) * s.frequency_penalty;
        }
        i = j;
    }
}

// The window: the last `penalty_last_n - n_extra` tokens of the sequence plus
// the accepted drafts, a repeated token counting twice.
void test_penalties() {
    Sampling s;
    s.repeat_penalty = 1.2f;
    s.presence_penalty = 0.25f;
    s.frequency_penalty = 0.1f;
    s.penalty_last_n = 8;
    const std::vector<int32_t> seq = {3, 1, 3, 5, 3, 7, 1, 9, 11, 3, 13};
    const int32_t extra[2] = {14, 3};
    const int64_t nv = 16;
    const std::vector<double> w = {1.0, 0.5, 2.0, 0.8, 0.1, 0.2, 0.3, 0.4,
                                   0.5, 0.6, 0.7, 1.1, 0.9, 1.3, 0.25, 1.5};
    std::vector<float> got = row_of(w, nv);
    std::vector<float> want = got;
    omph::model::apply_penalties(got.data(), nv, s, seq, extra, 2);
    std::vector<int32_t> win(seq.end() - 6, seq.end());  // 8 - 2
    win.push_back(extra[0]);
    win.push_back(extra[1]);
    penalties_ref(want.data(), nv, s, win);
    bool equal = true;
    for (int64_t i = 0; i < nv; ++i) {
        equal = equal && got[i] == want[i];
    }
    CHECK(equal, "the penalties differ from the reference window");
    // token 14 is only in the extras (once), token 3 once in the window and
    // once in the extras: the sign-aware rule divides positives, multiplies
    // the negative logit of token 9, and an untouched token stays untouched
    CHECK(std::fabs(got[14] - (float) (std::log(0.25) * 1.2 - 0.25 - 0.1)) < 1e-5f, "token 14: %f", got[14]);
    CHECK(std::fabs(got[3] - (float) (std::log(0.8) * 1.2 - 0.25 - 2 * 0.1)) < 1e-5f, "token 3: %f", got[3]);
    CHECK(std::fabs(got[9] - (float) (std::log(0.6) * 1.2 - 0.25 - 0.1)) < 1e-5f, "token 9: %f", got[9]);
    CHECK(got[2] == (float) std::log(2.0), "token 2 is not in the window");
    // the greedy pick applies them too (llama.cpp's chain): 10 / 2 = 5, while
    // the 6 of token 1 is untouched, so the argmax moves
    std::vector<float> row(2, -60.0f);
    row[0] = 10.0f;
    row[1] = 6.0f;
    Sampling g;
    g.repeat_penalty = 2.0f;
    g.penalty_last_n = 4;
    const std::vector<int32_t> gseq = {0, 0, 0, 0};
    std::mt19937_64 grng(7);
    CHECK(omph::model::sample_row(row, g, gseq, grng) == 1, "greedy ignores the penalties");
}

void test_top_k() {
    Sampling s;
    s.temperature = 1.0f;
    s.top_k = 3;
    std::vector<float> row = row_of({1.0, 0.5, 0.8, 0.2, 0.9, 0.1});
    Dist d;
    omph::model::distribution(row.data(), (int64_t) row.size(), s, {}, d);
    CHECK(d.ids.size() == 3, "top-k kept %zu candidates", d.ids.size());
    CHECK(same_ids(d, {0, 2, 4}), "top-k kept the wrong ids");
    // without top-p the kept tokens stay in id order: 1.0, 0.8, 0.9
    CHECK(std::fabs(d.w[0] - 1.0) < 1e-6 && std::fabs(d.w[1] - 0.8) < 1e-6 && std::fabs(d.w[2] - 0.9) < 1e-6,
          "top-k weights: %f %f %f", d.w[0], d.w[1], d.w[2]);
}

void test_min_p() {
    Sampling s;
    s.temperature = 1.0f;
    s.min_p = 0.3f;
    std::vector<float> row = row_of({1.0, 0.6, 0.25, 0.1});
    Dist d;
    omph::model::distribution(row.data(), (int64_t) row.size(), s, {}, d);
    CHECK(same_ids(d, {0, 1}), "min-p kept the wrong ids");
    CHECK(std::fabs(d.total - 1.6) < 1e-6, "min-p total: %f", d.total);
}

void test_top_p() {
    Sampling s;
    s.temperature = 1.0f;
    s.top_p = 0.7f;
    std::vector<float> row = row_of({1.0, 0.5, 0.25, 0.125});
    Dist d;
    omph::model::distribution(row.data(), (int64_t) row.size(), s, {}, d);
    CHECK(d.ids.size() == 2 && d.ids[0] == 0 && d.ids[1] == 1, "top-p kept the wrong ids (descending)");
    CHECK(std::fabs(d.total - 1.5) < 1e-6, "top-p total: %f", d.total);
}

// min-p removes 0.35 and 0.25 (2.7 -> 2.1 of mass): top-p 0.6 must then keep
// two candidates, not three (#315: with the pre-min-p denominator it kept
// everything the min-p had left).
void test_min_p_with_top_p() {
    Sampling s;
    s.temperature = 1.0f;
    s.min_p = 0.4f;
    s.top_p = 0.6f;
    std::vector<float> row = row_of({1.0, 0.6, 0.5, 0.35, 0.25});
    Dist d;
    omph::model::distribution(row.data(), (int64_t) row.size(), s, {}, d);
    CHECK(d.ids.size() == 2 && d.ids[0] == 0 && d.ids[1] == 1, "min-p + top-p kept %zu candidates", d.ids.size());
    // and the other order of the two thresholds: min-p alone would keep three
    Sampling only_min = s;
    only_min.top_p = 1.0f;
    std::vector<float> row2 = row_of({1.0, 0.6, 0.5, 0.35, 0.25});
    Dist d2;
    omph::model::distribution(row2.data(), (int64_t) row2.size(), only_min, {}, d2);
    CHECK(same_ids(d2, {0, 1, 2}), "min-p alone kept the wrong ids");
}

void test_temperature() {
    Sampling s;
    s.temperature = 0.5f;
    std::vector<float> row = row_of({1.0, std::exp(1.0)});  // logits 0 and 1
    Dist d;
    omph::model::distribution(row.data(), (int64_t) row.size(), s, {}, d);
    // without top-p the kept tokens stay in id order
    CHECK(d.ids.size() == 2 && d.ids[0] == 0 && d.ids[1] == 1, "temperature: wrong ids");
    CHECK(std::fabs(d.w[1] / d.w[0] - std::exp(2.0)) < 1e-4, "temperature 0.5: the ratio is not e^2");
}

void test_draw() {
    Sampling s;
    s.temperature = 1.0f;
    std::vector<float> row = row_of({9.0, 1.0});
    Dist d;
    omph::model::distribution(row.data(), (int64_t) row.size(), s, {}, d);
    std::mt19937_64 rng(1234);
    int64_t big = 0;
    for (int i = 0; i < 20000; ++i) {
        const int32_t id = omph::model::draw(d, -1, rng);
        CHECK(id == 0 || id == 1, "the draw returned an id outside the kept set: %d", id);
        big += id == 0;
    }
    CHECK(big > 17800 && big < 18200, "the 90 %% candidate won %lld of 20000 draws", (long long) big);
    for (int i = 0; i < 100; ++i) {
        CHECK(omph::model::draw(d, 0, rng) == 1, "skip does not exclude the token");
    }
}

} // namespace

int main() {
    test_penalties();
    test_top_k();
    test_min_p();
    test_top_p();
    test_min_p_with_top_p();
    test_temperature();
    test_draw();
    if (omph_test::failures == 0) {
        std::printf("test_sampler: ok\n");
    }
    return omph_test::failures;
}
