// The packed greedy argmax (#102, #318): shared by the device kernel and the
// host tests, so a CPU test can pin the packing (#335). HIP-free on purpose:
// the CPU test build has no ROCm.
#pragma once

#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define OMPH_ARGMAX_HD __host__ __device__
#else
#define OMPH_ARGMAX_HD
#endif

namespace omph::kernels {

// max over i of (orderable(x[i]) << 32) | (0xffffffff - i), so the token is
// argmax_key_index(*key) — the lowest index among the maxima, as the host's
// argmax_finite picks it. -0.0 counts as +0.0. A non-finite value never beats
// a finite one (orderable 0); a row with nothing finite gives its first index
// (#335), not -1.
OMPH_ARGMAX_HD inline unsigned long long argmax_key_pack(const float v, const unsigned i) {
    if (!__builtin_isfinite(v)) {
        return (unsigned long long) (0xffffffffu - i);
    }
    uint32_t u = __builtin_bit_cast(uint32_t, v);
    u = u == 0x80000000u ? 0u : u;  // -0.0 ties with +0.0
    const uint32_t ord = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long) ord << 32) | (0xffffffffu - i);
}

// The token index of a packed key. key 0 is the key of an empty row (the
// kernel never ran on a value): id 0, the same the host's argmax_finite gives.
inline int32_t argmax_key_index(const unsigned long long key) {
    if (key == 0) {
        return 0;
    }
    return (int32_t) (0xffffffffu - (uint32_t) (key & 0xffffffffu));
}

} // namespace omph::kernels
