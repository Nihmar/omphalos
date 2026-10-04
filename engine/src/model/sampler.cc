#include "model/sampler.hh"

#include <algorithm>
#include <cmath>
#include <utility>

namespace omph::model {

void apply_penalties(float * row, const int64_t nv, const Sampling & s, const std::vector<int32_t> & seq,
                     const int32_t * extra, const int64_t n_extra) {
    const int64_t window = std::max<int64_t>(0, (int64_t) s.penalty_last_n - n_extra);
    const int64_t from = std::max<int64_t>(0, (int64_t) seq.size() - window);
    std::vector<int32_t> ids(seq.begin() + (std::ptrdiff_t) from, seq.end());
    if (n_extra > 0) {
        ids.insert(ids.end(), extra, extra + n_extra);
    }
    std::sort(ids.begin(), ids.end());
    for (size_t i = 0; i < ids.size();) {
        size_t j = i;
        while (j < ids.size() && ids[j] == ids[i]) {
            ++j;
        }
        const int32_t id = ids[i];
        if (id >= 0 && id < nv) {
            // llama.cpp's rule: a negative logit is multiplied, a positive one
            // divided (dividing both would make negative logits more likely)
            if (s.repeat_penalty != 1.0f) {
                row[id] = row[id] <= 0.0f ? row[id] * s.repeat_penalty : row[id] / s.repeat_penalty;
            }
            row[id] -= s.presence_penalty + (float) (j - i) * s.frequency_penalty;
        }
        i = j;
    }
}

void distribution(float * row, const int64_t nv, const Sampling & s, const std::vector<int32_t> & seq, Dist & d,
                  const int32_t * extra, const int64_t n_extra) {
    if (s.penalizes()) {
        apply_penalties(row, nv, s, seq, extra, n_extra);
    }
    const float inv_t = 1.0f / s.temperature;
    const float best = *std::max_element(row, row + nv);
    std::vector<std::pair<float, int32_t>> cand;  // (scaled logit, id), in id order
    cand.reserve(4096);
    for (int64_t i = 0; i < nv; ++i) {
        const float z = (row[i] - best) * inv_t;
        if (z > -30.0f) {
            cand.emplace_back(z, (int32_t) i);
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
    double sum = 0.0;
    for (const auto & c : cand) {
        sum += std::exp((double) c.first);
    }
    if (s.min_p > 0.0f) {  // relative to the best, whose weight is exp(0) = 1
        const float floor = std::log(s.min_p);
        cand.erase(std::remove_if(cand.begin(), cand.end(), [&](const auto & c) { return c.first < floor; }),
                   cand.end());
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

int32_t sample_row(std::vector<float> & logits, const Sampling & s, const std::vector<int32_t> & seq,
                   std::mt19937_64 & rng) {
    if (s.temperature <= 0.0f) {
        if (s.penalizes()) {  // llama.cpp applies them before the greedy pick too
            apply_penalties(logits.data(), (int64_t) logits.size(), s, seq);
        }
        return (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    Dist d;
    distribution(logits.data(), (int64_t) logits.size(), s, seq, d);
    return draw(d, -1, rng);
}

} // namespace omph::model
