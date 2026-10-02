// SHA-256 (FIPS 180-4) of a byte range, for the .omph header's record of the
// GGUF it was converted from (#178). Small and dependency-free on purpose.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace omph::format {

// Lowercase hex digest of [data, data + n).
std::string sha256_hex(const uint8_t * data, size_t n);

} // namespace omph::format
