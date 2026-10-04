// Parsing a tool's command line without half-reading a value (#306).
//
// std::atoi/atof stop at the first character they cannot read and return what
// they have: `--top-p 0.95--top-k` used to set top_p to 0.95 and leave the
// rest as a stray argument, and `--port 7070x` started a server on 7070. A
// typo in an option must be a refusal that names the flag, not a silently
// different configuration.
#pragma once

#include <cerrno>
#include <cstdio>
#include <cstdlib>

namespace omph::cli {

// The whole string is an integer, or the process exits 2 saying which flag got
// what. `flag` is the option as the user wrote it ("--ctx").
inline long long integer(const char * s, const char * flag) {
    char * end = nullptr;
    errno = 0;
    const long long v = std::strtoll(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        std::fprintf(stderr, "%s wants an integer, got '%s'\n", flag, s);
        std::exit(2);
    }
    return v;
}

inline unsigned long long uinteger(const char * s, const char * flag) {
    char * end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        std::fprintf(stderr, "%s wants an unsigned integer, got '%s'\n", flag, s);
        std::exit(2);
    }
    return v;
}

inline double number(const char * s, const char * flag) {
    char * end = nullptr;
    errno = 0;
    const double v = std::strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0') {
        std::fprintf(stderr, "%s wants a number, got '%s'\n", flag, s);
        std::exit(2);
    }
    return v;
}

} // namespace omph::cli
