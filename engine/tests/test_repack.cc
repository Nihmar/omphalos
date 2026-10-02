// The repack is a lossless re-layout (AGENTS.md: weights stay bit-exact). For
// every type with a repacked layout: random blocks -> repack -> unrepack must
// give the source back byte for byte, whatever the destination buffer held
// before (a repack that ORs into its output is only right on a zeroed buffer),
// and the layout must not be larger than the source (#94).
#include "check.hh"
#include "format/repack.hh"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

const char * name_of(const uint32_t t) {
    switch (t) {
        case 10: return "Q2_K";
        case 12: return "Q4_K";
        case 14: return "Q6_K";
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 29: return "IQ1_M";
        default: return "?";
    }
}

void round_trip(const uint32_t type, const int64_t n_blocks, const uint8_t fill,
                std::mt19937 & rng) {
    const int64_t bb = omph::format::quant_block_bytes(type);
    std::vector<uint8_t> src((size_t) (bb * n_blocks));
    for (uint8_t & b : src) {
        b = (uint8_t) rng();
    }
    const int64_t want = omph::format::repacked_bytes(type, n_blocks);
    // the destination arrives full of `fill`: repack_any resizes it in place
    std::vector<uint8_t> packed((size_t) want, fill);
    CHECK(omph::format::repack_any(type, src.data(), n_blocks, packed), "%s: repack refused",
          name_of(type));
    CHECK((int64_t) packed.size() == want, "%s: %zu bytes, layout says %lld", name_of(type),
          packed.size(), (long long) want);
    std::vector<uint8_t> back(src.size(), (uint8_t) ~fill);
    CHECK(omph::format::unrepack_any(type, packed.data(), n_blocks, back), "%s: unrepack refused",
          name_of(type));
    size_t first_bad = src.size();
    for (size_t i = 0; i < src.size(); ++i) {
        if (back[i] != src[i]) {
            first_bad = i;
            break;
        }
    }
    CHECK(first_bad == src.size(),
          "%s (%lld blocks, dst prefilled 0x%02x): first mismatch at byte %zu (block %zu, "
          "offset %zu)",
          name_of(type), (long long) n_blocks, fill, first_bad, first_bad / (size_t) bb,
          first_bad % (size_t) bb);
}

} // namespace

int main() {
    std::mt19937 rng(12345);
    const uint32_t types[] = {10, 12, 14, 16, 17, 18, 21, 22, 23, 29};
    for (const uint32_t type : types) {
        const int64_t bb = omph::format::quant_block_bytes(type);
        CHECK(bb > 0, "%s: no block size", name_of(type));
        // Sections are padded to 128 B, so a tensor costs at most a few
        // alignments more than its blocks; nothing per block (#94).
        const int64_t n = 4096;
        const int64_t packed = omph::format::repacked_bytes(type, n);
        CHECK(packed <= bb * n + 8 * 128, "%s: %lld bytes repacked for %lld source bytes",
              name_of(type), (long long) packed, (long long) (bb * n));
        for (const int64_t n_blocks : {1, 3, 257}) {
            for (const uint8_t fill : {(uint8_t) 0x00, (uint8_t) 0xff, (uint8_t) 0xa5}) {
                round_trip(type, n_blocks, fill, rng);
            }
        }
    }
    std::vector<uint8_t> out;
    CHECK(!omph::format::repack_any(0, nullptr, 1, out), "F32 has no repacked layout");
    // the engine's layouts (#178): IQ3_S tiles round trip, rows a multiple of 16
    for (const uint32_t type : types) {
        for (const int64_t rows : {16, 48}) {
            const int64_t k = 512;
            const int64_t bb = omph::format::quant_block_bytes(type);
            std::vector<uint8_t> src((size_t) (rows * (k / 256) * bb));
            for (auto & c : src) c = (uint8_t) rng();
            std::vector<uint8_t> lay, back;
            const bool ok = omph::format::to_engine_layout(type, src.data(), rows, k, lay);
            CHECK(ok && (int64_t) lay.size() == omph::format::engine_layout_bytes(type, rows, k),
                  "%s: engine layout size", name_of(type));
            CHECK(omph::format::from_engine_layout(type, lay.data(), rows, k, back) && back == src,
                  "%s (%lld rows): engine layout round trip", name_of(type), (long long) rows);
        }
    }
    CHECK(omph::format::engine_layout(21) == 2 && omph::format::iq3s_tiles_bytes(16, 2) == 2 * 1760,
          "IQ3_S takes the tiles, 1760 bytes per tile and block");
    CHECK(omph::format::engine_layout_bytes(21, 8, 512) == 0, "IQ3_S tiles need rows %% 16 == 0");
    std::printf("test_repack: %d failure(s)\n", omph_test::failures);
    return omph_test::failures == 0 ? 0 : 1;
}
