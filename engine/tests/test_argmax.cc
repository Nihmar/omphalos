// The packed greedy argmax: the device kernel's packing and the host rule must
// agree, an all-non-finite row included (#318, #335).
#include "check.hh"
#include "kernels/argmax_key.hh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using omph::kernels::argmax_key_index;
using omph::kernels::argmax_key_pack;

namespace {

const float kInf = std::numeric_limits<float>::infinity();
const float kMax = std::numeric_limits<float>::max();

// The whole-row reduction the kernel does: the maximum of the packed keys.
int32_t packed_argmax(const std::vector<float> & row) {
    unsigned long long best = 0;
    for (size_t i = 0; i < row.size(); ++i) {
        const unsigned long long k = argmax_key_pack(row[i], (unsigned) i);
        best = k > best ? k : best;
    }
    return argmax_key_index(best);
}

void test_pack() {
    CHECK(packed_argmax({1.0f, 3.0f, 2.0f}) == 1, "the plain maximum");
    CHECK(packed_argmax({2.0f, 1.0f, 2.0f}) == 0, "the lowest index wins a tie");
    CHECK(packed_argmax({-0.0f, 0.0f}) == 0, "-0.0 ties with +0.0");
    CHECK(packed_argmax({-kMax, kMax}) == 1, "the sign is respected");
    // a non-finite value never beats a finite one, not even -inf
    CHECK(packed_argmax({std::nanf(""), 1.0f, 2.0f}) == 2, "a leading NaN never wins");
    CHECK(packed_argmax({std::nanf(""), -kMax}) == 1, "a finite value beats a NaN");
    CHECK(packed_argmax({-kInf, 0.0f}) == 1, "-inf is not finite");
    CHECK(packed_argmax({kMax, kInf}) == 0, "+inf is not finite");
    // ... and a row with nothing finite gives index 0, not -1 (#335)
    CHECK(packed_argmax({std::nanf(""), kInf, -kInf}) == 0, "a row with no finite value gives id 0");
    CHECK(packed_argmax({kInf, std::nanf("")}) == 0, "the first non-finite wins");
    // the empty row: the kernel's initial key, index 0
    CHECK(argmax_key_index(0) == 0, "the initial key must give id 0");
    CHECK(packed_argmax({}) == 0, "an empty row gives id 0");
}

} // namespace

int main() {
    test_pack();
    if (omph_test::failures == 0) {
        std::printf("test_argmax: ok\n");
    }
    return omph_test::failures;
}
