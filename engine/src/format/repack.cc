#include "format/repack.hh"

#include <cstring>
#include <stdexcept>

namespace omph::format {
namespace {

#pragma pack(push, 1)
struct BlockQ4K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
#pragma pack(pop)

static_assert(sizeof(BlockQ4K) == 144, "Q4_K block must be 144 bytes");

// ggml-common.h get_scale_min_k4: eight 6-bit scales and eight 6-bit mins
// packed into 12 bytes.
inline void get_scale_min_k4(const int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

inline void put_scale_min_k4(const int j, uint8_t * q, const uint8_t d, const uint8_t m) {
    if (j < 4) {
        q[j] = (uint8_t) ((q[j] & 0xC0) | (d & 63));
        q[j + 4] = (uint8_t) ((q[j + 4] & 0xC0) | (m & 63));
    } else {
        q[j + 4] = (uint8_t) ((d & 0xF) | ((m & 0xF) << 4));
        q[j - 4] = (uint8_t) ((q[j - 4] & 0x3F) | ((d >> 4) << 6));
        q[j] = (uint8_t) ((q[j] & 0x3F) | ((m >> 4) << 6));
    }
}

inline int64_t align128(const int64_t v) { return (v + 127) & ~(int64_t) 127; }

} // namespace

Q4kLayout q4k_layout(const int64_t n_blocks) {
    Q4kLayout l;
    l.n_blocks = n_blocks;
    l.qs_off = 0;
    l.sc_off = align128(n_blocks * 128);
    l.mn_off = align128(l.sc_off + n_blocks * 8);
    l.d_off = align128(l.mn_off + n_blocks * 8);
    l.dmin_off = align128(l.d_off + n_blocks * 2);
    l.total = align128(l.dmin_off + n_blocks * 2);
    return l;
}

void repack_q4k(const void * src, const int64_t n_blocks, void * dst) {
    const Q4kLayout l = q4k_layout(n_blocks);
    const auto * in = static_cast<const BlockQ4K *>(src);
    uint8_t * out = static_cast<uint8_t *>(dst);
    uint8_t * qs = out + l.qs_off;
    uint8_t * sc = out + l.sc_off;
    uint8_t * mn = out + l.mn_off;
    uint8_t * d = out + l.d_off;
    uint8_t * dmin = out + l.dmin_off;

    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(d + 2 * b, &in[b].d, 2);
        std::memcpy(dmin + 2 * b, &in[b].dmin, 2);
        for (int s = 0; s < 8; ++s) {
            uint8_t sval = 0;
            uint8_t mval = 0;
            get_scale_min_k4(s, in[b].scales, sval, mval);
            sc[b * 8 + s] = sval;
            mn[b * 8 + s] = mval;
        }
        // ggml group j (32 bytes) holds the low nibbles for sub-block 2j and the
        // high nibbles for sub-block 2j+1; here every sub-block gets 16 bytes.
        uint8_t * sub = qs + b * 128;
        for (int j = 0; j < 4; ++j) {
            const uint8_t * g = in[b].qs + j * 32;
            for (int i = 0; i < 32; ++i) {
                const uint8_t packed = g[i];
                sub[(2 * j) * 16 + i / 2] |= (uint8_t) ((packed & 0xF) << (4 * (i & 1)));
                sub[(2 * j + 1) * 16 + i / 2] |= (uint8_t) ((packed >> 4) << (4 * (i & 1)));
            }
        }
    }
}

void unrepack_q4k(const void * src, const int64_t n_blocks, void * dst) {
    const Q4kLayout l = q4k_layout(n_blocks);
    const uint8_t * in = static_cast<const uint8_t *>(src);
    const uint8_t * qs = in + l.qs_off;
    const uint8_t * sc = in + l.sc_off;
    const uint8_t * mn = in + l.mn_off;
    const uint8_t * d = in + l.d_off;
    const uint8_t * dmin = in + l.dmin_off;
    auto * out = static_cast<BlockQ4K *>(dst);

    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(&out[b].d, d + 2 * b, 2);
        std::memcpy(&out[b].dmin, dmin + 2 * b, 2);
        std::memset(out[b].scales, 0, sizeof(out[b].scales));
        for (int s = 0; s < 8; ++s) {
            put_scale_min_k4(s, out[b].scales, sc[b * 8 + s], mn[b * 8 + s]);
        }
        std::memset(out[b].qs, 0, sizeof(out[b].qs));
        const uint8_t * sub = qs + b * 128;
        for (int j = 0; j < 4; ++j) {
            uint8_t * g = out[b].qs + j * 32;
            for (int i = 0; i < 32; ++i) {
                const uint8_t lo = sub[(2 * j) * 16 + i / 2] >> (4 * (i & 1));
                const uint8_t hi = sub[(2 * j + 1) * 16 + i / 2] >> (4 * (i & 1));
                g[i] = (uint8_t) ((lo & 0xF) | ((hi & 0xF) << 4));
            }
        }
    }
}

} // namespace omph::format
