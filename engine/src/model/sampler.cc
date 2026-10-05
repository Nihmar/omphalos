#include "model/sampler.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace omph::model {

void PenaltyWindow::add(const int32_t id, const int32_t d) {
    if (d == 0) {
        return;
    }
    const auto it = std::lower_bound(v_.begin(), v_.end(), id,
                                     [](const Cnt & c, const int32_t k) { return c.id < k; });
    if (it != v_.end() && it->id == id) {
        it->n += d;
        if (it->n <= 0) {
            v_.erase(it);
        }
        return;
    }
    if (d > 0) {
        v_.insert(it, Cnt{id, d});
    }
}

void PenaltyWindow::reset(const std::vector<int32_t> & seq, const int64_t penalty_last_n) {
    const int64_t size = (int64_t) seq.size();
    // the window the sequence ends with: the last min(size, penalty_last_n)
    const int64_t want = std::max<int64_t>(0, size - std::max<int64_t>(0, penalty_last_n));
    if (penalty_last_n != last_n_ || size < end_) {
        // a new window length, or a sequence that was rewound: start over
        v_.clear();
        added_.clear();
        first_ = want;
        end_ = size;
        last_n_ = penalty_last_n;
        extras_ = 0;
        scratch_.assign(seq.begin() + (std::ptrdiff_t) want, seq.end());
        std::sort(scratch_.begin(), scratch_.end());
        for (size_t i = 0; i < scratch_.size();) {
            size_t j = i;
            while (j < scratch_.size() && scratch_[j] == scratch_[i]) {
                ++j;
            }
            v_.push_back(Cnt{scratch_[i], (int32_t) (j - i)});
            i = j;
        }
        return;
    }
    // The sequence grew: the drafts the last step added are tokens of it now
    // (the sequence's own entries take over), the new ones join, and whatever
    // the last step's rows had pushed out of the window comes back.
    for (const int32_t id : added_) {
        add(id, -1);
    }
    added_.clear();
    extras_ = 0;
    last_n_ = penalty_last_n;
    for (int64_t i = end_; i < size; ++i) {
        add(seq[(size_t) i], 1);
    }
    end_ = size;
    for (; first_ < want; ++first_) {
        add(seq[(size_t) first_], -1);
    }
    for (; want < first_;) {
        --first_;
        add(seq[(size_t) first_], 1);
    }
}

void PenaltyWindow::next(const std::vector<int32_t> & seq, const int32_t extra) {
    if (last_n_ <= 0) {
        // a zero-length window holds nothing (llama.cpp: penalties off)
        added_.clear();
        extras_ = 0;
        v_.clear();
        return;
    }
    ++extras_;
    // the sequence's part of the window holds all of it while it is shorter
    // than what the drafts leave it: then nothing is dropped
    const int64_t len = std::min<int64_t>(end_, std::max<int64_t>(0, last_n_ - extras_));
    for (const int64_t want = end_ - len; first_ < want; ++first_) {
        add(seq[(size_t) first_], -1);
    }
    // More accepted drafts than the window holds (#340): the oldest extras
    // leave too, so the window stays exactly `last_n_` tokens. Before this the
    // window grew with the accepted drafts (row 5 of a step with 6 accepted
    // n-gram drafts penalized 6 tokens where plain sampling penalizes 4).
    while (extras_ > last_n_ && !added_.empty()) {
        add(added_.front(), -1);
        added_.erase(added_.begin());
        --extras_;
    }
    add(extra, 1);
    added_.push_back(extra);
}

void PenaltyWindow::apply(float * row, const int64_t nv, const Sampling & s) const {
    for (const Cnt & c : v_) {
        if (c.id < 0 || c.id >= nv || c.n <= 0) {
            continue;
        }
        // llama.cpp's rule: a negative logit is multiplied, a positive one
        // divided (dividing both would make negative logits more likely)
        if (s.repeat_penalty != 1.0f) {
            row[c.id] = row[c.id] <= 0.0f ? row[c.id] * s.repeat_penalty : row[c.id] / s.repeat_penalty;
        }
        row[c.id] -= s.presence_penalty + (float) c.n * s.frequency_penalty;
    }
}

void distribution(float * row, const int64_t nv, const Sampling & s, const PenaltyWindow & pen, Dist & d) {
    if (s.penalizes()) {
        pen.apply(row, nv, s);
    }
    const float inv_t = 1.0f / s.temperature;
    const float best = *std::max_element(row, row + nv);
    std::vector<std::pair<float, int32_t>> cand;  // (scaled logit, id), in id order
    cand.reserve(4096);
    // A subnormal temperature (1e-39, #336) gives inv_t = inf and z = NaN for
    // the best token: nothing passes the cut. Skip the fill then; the fallback
    // below keeps the best token alone.
    if (std::isfinite(inv_t) && inv_t > 0.0f) {
        for (int64_t i = 0; i < nv; ++i) {
            const float z = (row[i] - best) * inv_t;
            if (z > -30.0f) {
                cand.emplace_back(z, (int32_t) i);
            }
        }
    }
    const auto higher = [](const auto & a, const auto & b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    if (s.top_k > 0 && (size_t) s.top_k < cand.size()) {
        std::nth_element(cand.begin(), cand.begin() + s.top_k, cand.end(), higher);
        cand.resize((size_t) s.top_k);
        std::sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) { return a.second < b.second; });
    }
    if (s.min_p > 0.0f) {  // relative to the best, whose weight is exp(0) = 1
        const float floor = std::log(s.min_p);
        cand.erase(std::remove_if(cand.begin(), cand.end(), [&](const auto & c) { return c.first < floor; }),
                   cand.end());
    }
    // top-p's mass is the mass min-p kept: with the pre-min-p total the two
    // truncations did not compose (a min_p that removed a lot of mass made
    // top-p keep more candidates than it asks for).
    double sum = 0.0;
    for (const auto & c : cand) {
        sum += std::exp((double) c.first);
    }
    if (s.top_p < 1.0f) {
        // the smallest descending prefix whose mass (of all the candidates') reaches top_p
        size_t sorted = 0;
        size_t keep = cand.size();
        double acc = 0.0;
        for (size_t want = 64;; want *= 4) {
            const size_t m = std::min(want, cand.size());
            std::partial_sort(cand.begin() + (std::ptrdiff_t) sorted, cand.begin() + (std::ptrdiff_t) m, cand.end(),
                              higher);
            for (; sorted < m; ++sorted) {
                acc += std::exp((double) cand[sorted].first) / sum;
                if (acc >= (double) s.top_p) {
                    keep = sorted + 1;
                    break;
                }
            }
            if (keep < cand.size() || m == cand.size()) {
                break;
            }
        }
        cand.resize(keep);
    }
    if (cand.empty()) {
        // Nothing survived: inv_t was not a positive finite number, min_p > 1,
        // or the row holds no finite value. Keep the greedy pick alone, p = 1
        // (the host rule of argmax_finite, #336).
        d.ids.assign(1, argmax_finite(row, nv));
        d.w.assign(1, 1.0);
        d.total = 1.0;
        return;
    }
    d.ids.resize(cand.size());
    d.w.resize(cand.size());
    d.total = 0.0;
    for (size_t i = 0; i < cand.size(); ++i) {
        d.ids[i] = cand[i].second;
        d.w[i] = std::exp((double) cand[i].first);
        d.total += d.w[i];
    }
}

int32_t draw(const Dist & d, const int32_t skip, std::mt19937_64 & rng) {
    if (d.ids.empty()) {
        return 0;  // distribution() never leaves the set empty (#336); do not fault
    }
    double total = d.total;
    for (size_t i = 0; i < d.ids.size() && skip >= 0; ++i) {
        if (d.ids[i] == skip) total -= d.w[i];
    }
    if (!(total > 0.0)) {
        return d.ids.front();  // only `skip` was left (not reached: it was kept with p = 1)
    }
    std::uniform_real_distribution<double> u(0.0, total);
    double r = u(rng);
    int32_t last = d.ids.front();
    for (size_t i = 0; i < d.ids.size(); ++i) {
        if (d.ids[i] == skip) continue;
        last = d.ids[i];
        r -= d.w[i];
        if (r <= 0.0) return d.ids[i];
    }
    return last;
}

int32_t argmax_finite(const float * row, const int64_t nv) {
    int32_t best = 0;
    float best_v = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < nv; ++i) {
        if (std::isfinite(row[i]) && row[i] > best_v) {
            best_v = row[i];
            best = (int32_t) i;
        }
    }
    return best;
}

int32_t sample_row(std::vector<float> & logits, const Sampling & s, const std::vector<int32_t> & seq,
                   std::mt19937_64 & rng) {
    PenaltyWindow pen;
    if (s.penalizes()) {
        pen.reset(seq, s.penalty_last_n);
    }
    if (s.temperature <= 0.0f) {
        if (s.penalizes()) {  // llama.cpp applies them before the greedy pick too
            pen.apply(logits.data(), (int64_t) logits.size(), s);
        }
        return argmax_finite(logits.data(), (int64_t) logits.size());
    }
    Dist d;
    distribution(logits.data(), (int64_t) logits.size(), s, pen, d);
    return draw(d, -1, rng);
}

} // namespace omph::model
