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

#pragma pack(push, 1)
struct BlockIq4Xs {
    uint16_t d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
#pragma pack(pop)

static_assert(sizeof(BlockIq4Xs) == 136, "IQ4_XS block must be 136 bytes");

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

// ------------------------------------------------------------------ IQ4_XS

const int8_t kIq4Codebook[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                 1,    13,   25,  38,  53,  69,  89,  113};

namespace {

inline int iq4xs_scale(const int ib, const uint8_t * scales_l, const uint16_t scales_h) {
    return ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
}

} // namespace

Iq4Layout iq4_layout(const int64_t n_blocks) {
    Iq4Layout l;
    l.n_blocks = n_blocks;
    l.qs_off = 0;
    l.sc_off = align128(n_blocks * 128);
    l.d_off = align128(l.sc_off + n_blocks * 8);
    l.total = align128(l.d_off + n_blocks * 2);
    return l;
}

void repack_iq4_xs(const void * src, const int64_t n_blocks, void * dst) {
    const Iq4Layout l = iq4_layout(n_blocks);
    const auto * in = static_cast<const BlockIq4Xs *>(src);
    uint8_t * out = static_cast<uint8_t *>(dst);
    uint8_t * qs = out + l.qs_off;
    uint8_t * sc = out + l.sc_off;
    uint8_t * d = out + l.d_off;

    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(d + 2 * b, &in[b].d, 2);
        std::memcpy(qs + b * 128, in[b].qs, 128);
        for (int ib = 0; ib < 8; ++ib) {
            sc[b * 8 + ib] = (uint8_t) iq4xs_scale(ib, in[b].scales_l, in[b].scales_h);
        }
    }
}

void unrepack_iq4_xs(const void * src, const int64_t n_blocks, void * dst) {
    const Iq4Layout l = iq4_layout(n_blocks);
    const uint8_t * in = static_cast<const uint8_t *>(src);
    const uint8_t * qs = in + l.qs_off;
    const uint8_t * sc = in + l.sc_off;
    const uint8_t * d = in + l.d_off;
    auto * out = static_cast<BlockIq4Xs *>(dst);

    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(&out[b].d, d + 2 * b, 2);
        std::memcpy(out[b].qs, qs + b * 128, 128);
        out[b].scales_l[0] = 0;
        out[b].scales_l[1] = 0;
        out[b].scales_l[2] = 0;
        out[b].scales_l[3] = 0;
        uint16_t hi = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const uint8_t ls = sc[b * 8 + ib];
            out[b].scales_l[ib / 2] |= (uint8_t) ((ls & 0xF) << (4 * (ib % 2)));
            hi |= (uint16_t) ((ls >> 4) & 3) << (2 * ib);
        }
        out[b].scales_h = hi;
    }
}

// ------------------------------------------------------------------ IQ3_XXS

namespace {

#pragma pack(push, 1)
struct BlockIq3Xxs {
    uint16_t d;
    uint8_t qs[96];  // 64 bytes of grid indices, then 32 bytes of scale/sign words
};
#pragma pack(pop)

static_assert(sizeof(BlockIq3Xxs) == 98, "IQ3_XXS block must be 98 bytes");

} // namespace

Iq3XxsLayout iq3_xxs_layout(const int64_t n_blocks) {
    Iq3XxsLayout l;
    l.n_blocks = n_blocks;
    l.qs_off = 0;
    l.aux_off = align128(n_blocks * 64);
    l.d_off = align128(l.aux_off + n_blocks * 32);
    l.total = align128(l.d_off + n_blocks * 2);
    return l;
}

void repack_iq3_xxs(const void * src, const int64_t n_blocks, void * dst) {
    const Iq3XxsLayout l = iq3_xxs_layout(n_blocks);
    const auto * in = static_cast<const BlockIq3Xxs *>(src);
    uint8_t * out = static_cast<uint8_t *>(dst);
    uint8_t * qs = out + l.qs_off;
    uint8_t * aux = out + l.aux_off;
    uint8_t * d = out + l.d_off;

    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(d + 2 * b, &in[b].d, 2);
        std::memcpy(qs + b * 64, in[b].qs, 64);
        std::memcpy(aux + b * 32, in[b].qs + 64, 32);
    }
}

void unrepack_iq3_xxs(const void * src, const int64_t n_blocks, void * dst) {
    const Iq3XxsLayout l = iq3_xxs_layout(n_blocks);
    const uint8_t * in = static_cast<const uint8_t *>(src);
    const uint8_t * qs = in + l.qs_off;
    const uint8_t * aux = in + l.aux_off;
    const uint8_t * d = in + l.d_off;
    auto * out = static_cast<BlockIq3Xxs *>(dst);

    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(&out[b].d, d + 2 * b, 2);
        std::memcpy(out[b].qs, qs + b * 64, 64);
        std::memcpy(out[b].qs + 64, aux + b * 32, 32);
    }
}

// ------------------------------------------------------------------ IQ3_S

namespace {

#pragma pack(push, 1)
struct BlockIq3S {
    uint16_t d;
    uint8_t qs[64];
    uint8_t qh[8];
    uint8_t signs[32];
    uint8_t scales[4];
};
#pragma pack(pop)

static_assert(sizeof(BlockIq3S) == 110, "IQ3_S block must be 110 bytes");

} // namespace

Iq3sLayout iq3s_layout(const int64_t n_blocks) {
    Iq3sLayout l;
    l.n_blocks = n_blocks;
    l.qs_off = 0;
    l.qh_off = align128(n_blocks * 64);
    l.signs_off = align128(l.qh_off + n_blocks * 8);
    l.scales_off = align128(l.signs_off + n_blocks * 32);
    l.d_off = align128(l.scales_off + n_blocks * 4);
    l.total = align128(l.d_off + n_blocks * 2);
    return l;
}

void repack_iq3_s(const void * src, const int64_t n_blocks, void * dst) {
    const Iq3sLayout l = iq3s_layout(n_blocks);
    const auto * in = static_cast<const BlockIq3S *>(src);
    uint8_t * out = static_cast<uint8_t *>(dst);
    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(out + l.d_off + 2 * b, &in[b].d, 2);
        std::memcpy(out + l.qs_off + b * 64, in[b].qs, 64);
        std::memcpy(out + l.qh_off + b * 8, in[b].qh, 8);
        std::memcpy(out + l.signs_off + b * 32, in[b].signs, 32);
        std::memcpy(out + l.scales_off + b * 4, in[b].scales, 4);
    }
}

void unrepack_iq3_s(const void * src, const int64_t n_blocks, void * dst) {
    const Iq3sLayout l = iq3s_layout(n_blocks);
    const uint8_t * in = static_cast<const uint8_t *>(src);
    auto * out = static_cast<BlockIq3S *>(dst);
    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(&out[b].d, in + l.d_off + 2 * b, 2);
        std::memcpy(out[b].qs, in + l.qs_off + b * 64, 64);
        std::memcpy(out[b].qh, in + l.qh_off + b * 8, 8);
        std::memcpy(out[b].signs, in + l.signs_off + b * 32, 32);
        std::memcpy(out[b].scales, in + l.scales_off + b * 4, 4);
    }
}

// ------------------------------------------------------------------ Q2_K

namespace {

#pragma pack(push, 1)
struct BlockQ2K {
    uint8_t scales[16];
    uint8_t qs[64];
    uint16_t d;
    uint16_t dmin;
};
#pragma pack(pop)

static_assert(sizeof(BlockQ2K) == 84, "Q2_K block must be 84 bytes");

} // namespace

Q2kLayout q2k_layout(const int64_t n_blocks) {
    Q2kLayout l;
    l.n_blocks = n_blocks;
    l.sc_off = 0;
    l.qs_off = align128(n_blocks * 16);
    l.d_off = align128(l.qs_off + n_blocks * 64);
    l.dmin_off = align128(l.d_off + n_blocks * 2);
    l.total = align128(l.dmin_off + n_blocks * 2);
    return l;
}

void repack_q2k(const void * src, const int64_t n_blocks, void * dst) {
    const Q2kLayout l = q2k_layout(n_blocks);
    const auto * in = static_cast<const BlockQ2K *>(src);
    uint8_t * out = static_cast<uint8_t *>(dst);
    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(out + l.sc_off + b * 16, in[b].scales, 16);
        std::memcpy(out + l.qs_off + b * 64, in[b].qs, 64);
        std::memcpy(out + l.d_off + 2 * b, &in[b].d, 2);
        std::memcpy(out + l.dmin_off + 2 * b, &in[b].dmin, 2);
    }
}

void unrepack_q2k(const void * src, const int64_t n_blocks, void * dst) {
    const Q2kLayout l = q2k_layout(n_blocks);
    const uint8_t * in = static_cast<const uint8_t *>(src);
    auto * out = static_cast<BlockQ2K *>(dst);
    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(out[b].scales, in + l.sc_off + b * 16, 16);
        std::memcpy(out[b].qs, in + l.qs_off + b * 64, 64);
        std::memcpy(&out[b].d, in + l.d_off + 2 * b, 2);
        std::memcpy(&out[b].dmin, in + l.dmin_off + 2 * b, 2);
    }
}

} // namespace omph::format