// The prompt-lookup n-gram index (#199), driven on the CPU: the generator's
// drafts must never continue into an image's placeholder ids (#334).
#include "check.hh"
#include "model/ngram.hh"

#include <cstdint>
#include <vector>

namespace {

using omph::model::NgramIndex;

std::vector<int32_t> propose(NgramIndex & ix, const std::vector<int32_t> & c, const int64_t k) {
    std::vector<int32_t> out;
    ix.propose(c, k, out);
    return out;
}

void test_repeat() {
    NgramIndex ix;
    const std::vector<int32_t> c = {1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4};
    // the tail {1,2,3,4} occurred before, followed by 5, 6, 7
    CHECK(propose(ix, c, 3) == (std::vector<int32_t>{5, 6, 7}), "the n-gram proposal is wrong");
    // at most k tokens are proposed
    CHECK(propose(ix, c, 2) == (std::vector<int32_t>{5, 6}), "k does not cap the proposal");
    // the index grows with the sequence: a 4-gram with no known continuation
    NgramIndex grown;
    CHECK(propose(grown, {1, 2, 3, 4, 5}, 3).empty(), "a 4-gram needs a known continuation");
    CHECK(propose(grown, {1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4}, 3) == (std::vector<int32_t>{5, 6, 7}),
          "the grown index lost the n-gram");
}

// An image's rows are negative placeholder ids: a proposal that enters one must
// stop before it (a negative id would index the embedding table out of bounds).
void test_image_placeholder() {
    // the continuation is entirely an image: nothing is proposed
    NgramIndex inside;
    const std::vector<int32_t> c = {1, 2, 3, 4, -9, -9, 1, 2, 3, 4};
    CHECK(propose(inside, c, 4).empty(), "a proposal entered an image");
    // text, then the image: only the text before it is proposed
    NgramIndex before;
    const std::vector<int32_t> c2 = {1, 2, 3, 4, 5, -3, -3, 1, 2, 3, 4};
    CHECK(propose(before, c2, 4) == (std::vector<int32_t>{5}), "the cut kept the image ids");
    // a negative id inside the matched n-gram does not matter, the match still
    // works and the cut only looks at the continuation
    NgramIndex neg_gram;
    const std::vector<int32_t> c3 = {-1, -2, 3, 4, 5, -1, -2, 3, 4};
    CHECK(propose(neg_gram, c3, 3) == (std::vector<int32_t>{5}),
          "a negative id in the repeated n-gram broke the match");
}

} // namespace

int main() {
    test_repeat();
    test_image_placeholder();
    if (omph_test::failures == 0) {
        std::printf("test_ngram: ok\n");
    }
    return omph_test::failures;
}
