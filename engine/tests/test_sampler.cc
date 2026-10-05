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
    // the row of a step with two accepted drafts: reset, then one next() per row
    omph::model::PenaltyWindow pen;
    pen.reset(seq, s.penalty_last_n);
    pen.next(seq, extra[0]);
    pen.next(seq, extra[1]);
    pen.apply(got.data(), nv, s);
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

// The window is extended across the rows of a step and across the steps: every
// row must carry the penalties a brute-force window of that row's last
// `penalty_last_n` tokens would (random sequences, drafts and rewinds).
void test_penalty_window() {
    Sampling s;
    s.repeat_penalty = 1.15f;
    s.presence_penalty = 0.2f;
    s.frequency_penalty = 0.05f;
    s.penalty_last_n = 37;
    const int64_t nv = 24;
    const std::vector<double> w = [] {
        std::vector<double> v(24);
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = 0.3 + 0.1 * (double) (i % 7);
        }
        return v;
    }();
    std::mt19937_64 rng(20261004);
    std::uniform_int_distribution<int32_t> id(0, 31);  // 24..31 are outside the row
    std::vector<int32_t> seq;
    omph::model::PenaltyWindow pen;
    for (int step = 0; step < 400; ++step) {
        const int rows = 1 + (int) (rng() % 5);  // the drafts of this step
        std::vector<int32_t> extra;
        for (int r = 0; r < rows; ++r) {
            extra.push_back(id(rng));
        }
        seq.push_back(id(rng));
        if (rng() % 20 == 0 && seq.size() > 5) {
            seq.resize(seq.size() - 3);  // a speculative rollback
        }
        pen.reset(seq, s.penalty_last_n);
        omph::model::PenaltyWindow again;  // resetting twice is the same window
        again.reset(seq, s.penalty_last_n);
        for (int r = 0; r < rows; ++r) {
            pen.next(seq, extra[(size_t) r]);
            again.next(seq, extra[(size_t) r]);
            std::vector<float> got = row_of(w, nv);
            std::vector<float> twin = got;
            std::vector<float> want = got;
            pen.apply(got.data(), nv, s);
            again.apply(twin.data(), nv, s);
            // the row's window: the last `penalty_last_n - (r + 1)` tokens of
            // the sequence, plus the r + 1 drafts accepted so far
            const int64_t keep = std::max<int64_t>(0, s.penalty_last_n - (r + 1));
            std::vector<int32_t> win(seq.end() - std::min<int64_t>(keep, (int64_t) seq.size()), seq.end());
            win.insert(win.end(), extra.begin(), extra.begin() + r + 1);
            penalties_ref(want.data(), nv, s, win);
            bool equal = got == twin;
            for (int64_t i = 0; i < nv && equal; ++i) {
                equal = got[i] == want[i];
            }
            CHECK(equal, "step %d row %d: the incremental penalties differ", step, r);
            if (!equal) {
                return;  // one report is enough, the rest would repeat it
            }
        }
    }
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

// The greedy pick ignores non-finite values, exactly as the device's packed
// argmax does (kernels/elementwise.hip): a corrupted row must not give the two
// paths two different tokens (#318).
void test_argmax_finite() {
    const float plain[] = {1.0f, 3.0f, 2.0f};
    CHECK(omph::model::argmax_finite(plain, 3) == 1, "the plain maximum");
    const float ties[] = {2.0f, 1.0f, 2.0f};
    CHECK(omph::model::argmax_finite(ties, 3) == 0, "the lowest id wins a tie");
    const float nan_first[] = {std::nanf(""), 1.0f, 2.0f};
    CHECK(omph::model::argmax_finite(nan_first, 3) == 2, "a leading NaN never wins");
    const float nan_last[] = {1.0f, 2.0f, std::nanf("")};
    CHECK(omph::model::argmax_finite(nan_last, 3) == 1, "a trailing NaN is ignored");
    const float inf[] = {1.0f, INFINITY, -INFINITY};
    CHECK(omph::model::argmax_finite(inf, 3) == 0, "an infinity is not finite either");
    const float none[] = {std::nanf(""), INFINITY, -INFINITY};
    CHECK(omph::model::argmax_finite(none, 3) == 0, "a row with no finite value gives id 0");
    Sampling g;
    std::vector<float> row = {1.0f, std::nanf(""), 3.0f, -INFINITY};
    std::mt19937_64 rng(3);
    CHECK(omph::model::sample_row(row, g, {}, rng) == 2, "greedy ignores the NaN");
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
    // a set that is empty anyway must not fault (distribution() never leaves one, #336)
    Dist empty;
    CHECK(omph::model::draw(empty, -1, rng) == 0, "the empty draw must not fault");
}

// An empty candidate set used to reach draw()'s front() (#336): a subnormal
// temperature, a min_p above 1, or a row with no finite value. All three must
// fall back to the greedy token alone, p = 1.
void test_empty_candidates() {
    const std::vector<double> w = {1.0, 0.6, 0.3, 0.1};
    // temperature 1e-39: 1/t is +inf, every scaled logit is NaN or -inf
    Sampling sub;
    sub.temperature = 1e-39f;
    std::vector<float> row = row_of(w);
    Dist d;
    omph::model::distribution(row.data(), (int64_t) row.size(), sub, {}, d);
    CHECK(d.ids.size() == 1 && d.ids[0] == 0, "a subnormal temperature kept %zu candidates", d.ids.size());
    CHECK(d.total == 1.0, "the fallback is not a point mass: %f", d.total);
    std::mt19937_64 rng(5);
    CHECK(omph::model::draw(d, -1, rng) == 0, "the point mass draws its token");
    // min_p > 1 removes everything (the C ABI refuses it, distribution is the backstop)
    Sampling mp;
    mp.temperature = 1.0f;
    mp.min_p = 1.5f;
    std::vector<float> row2 = row_of(w);
    Dist d2;
    omph::model::distribution(row2.data(), (int64_t) row2.size(), mp, {}, d2);
    CHECK(d2.ids.size() == 1 && d2.ids[0] == 0, "min_p > 1 kept %zu candidates", d2.ids.size());
    // a row with no finite value: id 0, the host's all-non-finite rule
    Sampling plain;
    plain.temperature = 1.0f;
    std::vector<float> nan_row = {std::nanf(""), INFINITY, -INFINITY};
    Dist d3;
    omph::model::distribution(nan_row.data(), (int64_t) nan_row.size(), plain, {}, d3);
    CHECK(d3.ids.size() == 1 && d3.ids[0] == 0, "an all-non-finite row kept %zu candidates", d3.ids.size());
    CHECK(omph::model::sample_row(nan_row, plain, {}, rng) == 0, "the all-NaN row must give id 0");
}

// top_p 0 keeps the single most likely token, as llama.cpp does; the server
// used to turn it into 1.0 (#336).
void test_top_p_zero() {
    Sampling s;
    s.temperature = 1.0f;
    s.top_p = 0.0f;
    std::vector<float> row = row_of({1.0, 0.8, 0.2});
    Dist d;
    omph::model::distribution(row.data(), (int64_t) row.size(), s, {}, d);
    CHECK(d.ids.size() == 1 && d.ids[0] == 0, "top_p 0 kept %zu candidates", d.ids.size());
    CHECK(d.total == 1.0, "top_p 0 is not a point mass: %f", d.total);
    // and the most likely token, not the first id
    std::vector<float> row2 = row_of({1.0, 2.0, 0.5});
    Dist d2;
    omph::model::distribution(row2.data(), (int64_t) row2.size(), s, {}, d2);
    CHECK(d2.ids.size() == 1 && d2.ids[0] == 1, "top_p 0 must keep the argmax");
}

} // namespace

int main() {
    test_penalties();
    test_penalty_window();
    test_top_k();
    test_min_p();
    test_top_p();
    test_min_p_with_top_p();
    test_temperature();
    test_argmax_finite();
    test_draw();
    test_empty_candidates();
    test_top_p_zero();
    if (omph_test::failures == 0) {
        std::printf("test_sampler: ok\n");
    }
    return omph_test::failures;
}
