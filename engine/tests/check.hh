// A minimal test harness: CHECK records a failure and goes on; main returns
// the failure count (0 = pass).
#pragma once

#include <cstdio>

namespace omph_test {
inline int failures = 0;
}

#define CHECK(cond, ...)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            ++omph_test::failures;                                               \
            std::printf("FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);          \
            std::printf(__VA_ARGS__);                                            \
            std::printf("\n");                                                   \
        }                                                                        \
    } while (0)
