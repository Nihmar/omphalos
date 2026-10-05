// Prompt lookup (#199): the n-gram index the generator proposes speculative
// drafts from. A HIP-free header of its own so the CPU tests can drive
// propose() on sequences with image placeholders (#334).
#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace omph::model {

class NgramIndex {
public:
    static constexpr int kMin = 4;
    static constexpr int kMax = 8;

    // Indexes the n-grams of c that end before its last token, then proposes up
    // to k tokens after the longest earlier match of c's tail. A continuation
    // is cut at the first negative id: an image's rows are negative placeholder
    // ids (#334), not vocabulary entries, and a draft must never be one (the
    // embedding lookup would read out of bounds).
    void propose(const std::vector<int32_t> & c, const int64_t k, std::vector<int32_t> & out) {
        out.clear();
        const int64_t len = (int64_t) c.size();
        for (int64_t e = std::max<int64_t>(done_, kMin); e < len; ++e) {  // continuation c[e] known
            for (int n = kMin; n <= kMax && n <= e; ++n) {
                map_[n][hash(c, e - n, n)] = e;
            }
        }
        done_ = std::max(done_, len);
        for (int n = kMax; n >= kMin; --n) {
            if (len <= n) {
                continue;
            }
            const auto it = map_[n].find(hash(c, len - n, n));
            if (it == map_[n].end()) {
                continue;
            }
            const int64_t p = it->second;
            if (!std::equal(c.begin() + (p - n), c.begin() + p, c.begin() + (len - n))) {
                continue;
            }
            out.assign(c.begin() + p, c.begin() + std::min<int64_t>(p + k, len));
            const auto neg = std::find_if(out.begin(), out.end(),
                                          [](const int32_t t) { return t < 0; });
            out.erase(neg, out.end());
            return;
        }
    }

private:
    static uint64_t hash(const std::vector<int32_t> & c, const int64_t at, const int n) {
        uint64_t h = 0xcbf29ce484222325ull ^ (uint64_t) n;
        for (int i = 0; i < n; ++i) {
            h = (h ^ (uint32_t) c[(size_t) (at + i)]) * 0x100000001b3ull;
        }
        return h;
    }
    std::unordered_map<uint64_t, int64_t> map_[kMax + 1];
    int64_t done_ = 0;  // n-grams ending (exclusive) before this index are in
};

} // namespace omph::model
