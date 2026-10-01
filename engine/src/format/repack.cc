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

inline int64_t align128(const int64_t v) { return (v + 127) & ~(int64_t) 127; }

} // namespace

Q4kLayout q4k_layout(const int64_t n_blocks) {
    Q4kLayout l;
    l.n_blocks = n_blocks;
    l.qs_off = 0;
    l.meta_off = align128(n_blocks * 128);
    l.total = align128(l.meta_off + n_blocks * 16);
    return l;
}

void repack_q4k(const void * src, const int64_t n_blocks, void * dst) {
    const Q4kLayout l = q4k_layout(n_blocks);
    const auto * in = static_cast<const BlockQ4K *>(src);
    uint8_t * out = static_cast<uint8_t *>(dst);
    uint8_t * qs = out + l.qs_off;
    uint8_t * meta = out + l.meta_off;

    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(meta + 16 * b, in[b].scales, 12);
        std::memcpy(meta + 16 * b + 12, &in[b].d, 2);
        std::memcpy(meta + 16 * b + 14, &in[b].dmin, 2);
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
    const uint8_t * meta = in + l.meta_off;
    auto * out = static_cast<BlockQ4K *>(dst);

    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(out[b].scales, meta + 16 * b, 12);
        std::memcpy(&out[b].d, meta + 16 * b + 12, 2);
        std::memcpy(&out[b].dmin, meta + 16 * b + 14, 2);
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

Iq4Layout iq4_layout(const int64_t n_blocks) {
    Iq4Layout l;
    l.n_blocks = n_blocks;
    l.qs_off = 0;
    l.meta_off = align128(n_blocks * 128);
    l.total = align128(l.meta_off + n_blocks * 8);
    return l;
}

void repack_iq4_xs(const void * src, const int64_t n_blocks, void * dst) {
    const Iq4Layout l = iq4_layout(n_blocks);
    const auto * in = static_cast<const BlockIq4Xs *>(src);
    uint8_t * out = static_cast<uint8_t *>(dst);
    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(out + l.meta_off + 8 * b, in[b].scales_l, 4);
        std::memcpy(out + l.meta_off + 8 * b + 4, &in[b].scales_h, 2);
        std::memcpy(out + l.meta_off + 8 * b + 6, &in[b].d, 2);
        std::memcpy(out + l.qs_off + 128 * b, in[b].qs, 128);
    }
}

void unrepack_iq4_xs(const void * src, const int64_t n_blocks, void * dst) {
    const Iq4Layout l = iq4_layout(n_blocks);
    const uint8_t * in = static_cast<const uint8_t *>(src);
    auto * out = static_cast<BlockIq4Xs *>(dst);
    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(out[b].scales_l, in + l.meta_off + 8 * b, 4);
        std::memcpy(&out[b].scales_h, in + l.meta_off + 8 * b + 4, 2);
        std::memcpy(&out[b].d, in + l.meta_off + 8 * b + 6, 2);
        std::memcpy(out[b].qs, in + l.qs_off + 128 * b, 128);
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

namespace {

// GGUF IQ3_XXS aux word: four 7-bit sign indices s_l at bits [7l, 7l + 7) and a
// 4-bit scale at [28, 32). Repacked: for group l, the even-weight sign bits
// (s_l bits 0, 2, 4, 6 = weights 0, 2, 4, 6) reversed in [4l, 4l + 4) and the
// odd ones (bits 1, 3, 5) reversed in [16 + 3l, 16 + 3l + 3); scale unchanged.
// The kernel then shifts each field straight into the pair-ordered sign word
// of repack.hh (weight 7's sign, the parity, it computes). Lossless.
uint32_t iq3xxs_aux_to_pairs(const uint32_t a) {
    uint32_t t = a & 0xF0000000u;
    for (int l = 0; l < 4; ++l) {
        const uint32_t sl = (a >> (7 * l)) & 127u;
        for (int k = 0; k < 4; ++k) {
            t |= ((sl >> (2 * k)) & 1u) << (4 * l + 3 - k);
        }
        for (int k = 0; k < 3; ++k) {
            t |= ((sl >> (2 * k + 1)) & 1u) << (16 + 3 * l + 2 - k);
        }
    }
    return t;
}

uint32_t iq3xxs_pairs_to_aux(const uint32_t t) {
    uint32_t a = t & 0xF0000000u;
    for (int l = 0; l < 4; ++l) {
        uint32_t sl = 0;
        for (int k = 0; k < 4; ++k) {
            sl |= ((t >> (4 * l + 3 - k)) & 1u) << (2 * k);
        }
        for (int k = 0; k < 3; ++k) {
            sl |= ((t >> (16 + 3 * l + 2 - k)) & 1u) << (2 * k + 1);
        }
        a |= sl << (7 * l);
    }
    return a;
}

} // namespace

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
        for (int sub = 0; sub < 8; ++sub) {
            uint32_t a = 0;
            std::memcpy(&a, in[b].qs + 64 + 4 * sub, 4);
            a = iq3xxs_aux_to_pairs(a);
            std::memcpy(aux + b * 32 + 4 * sub, &a, 4);
        }
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
        for (int sub = 0; sub < 8; ++sub) {
            uint32_t t = 0;
            std::memcpy(&t, aux + b * 32 + 4 * sub, 4);
            t = iq3xxs_pairs_to_aux(t);
            std::memcpy(out[b].qs + 64 + 4 * sub, &t, 4);
        }
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

namespace {

// GGUF sign order (bit w = weight w) <-> the pair order of the repacked IQ3_S
// signs (weight 2p at bit 15 - p, weight 2p + 1 at bit 31 - p).
uint32_t iq3s_signs_to_pairs(const uint32_t s) {
    uint32_t t = 0;
    for (int w = 0; w < 32; ++w) {
        const int pos = (w & 1) ? 31 - w / 2 : 15 - w / 2;
        t |= ((s >> w) & 1u) << pos;
    }
    return t;
}

uint32_t iq3s_pairs_to_signs(const uint32_t t) {
    uint32_t s = 0;
    for (int w = 0; w < 32; ++w) {
        const int pos = (w & 1) ? 31 - w / 2 : 15 - w / 2;
        s |= ((t >> pos) & 1u) << w;
    }
    return s;
}

} // namespace

void repack_iq3_s(const void * src, const int64_t n_blocks, void * dst) {
    const Iq3sLayout l = iq3s_layout(n_blocks);
    const auto * in = static_cast<const BlockIq3S *>(src);
    uint8_t * out = static_cast<uint8_t *>(dst);
    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(out + l.d_off + 2 * b, &in[b].d, 2);
        std::memcpy(out + l.qs_off + b * 64, in[b].qs, 64);
        std::memcpy(out + l.qh_off + b * 8, in[b].qh, 8);
        for (int sub = 0; sub < 8; ++sub) {
            uint32_t sg = 0;
            std::memcpy(&sg, in[b].signs + 4 * sub, 4);
            sg = iq3s_signs_to_pairs(sg);
            std::memcpy(out + l.signs_off + b * 32 + 4 * sub, &sg, 4);
        }
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
        for (int sub = 0; sub < 8; ++sub) {
            uint32_t sg = 0;
            std::memcpy(&sg, in + l.signs_off + b * 32 + 4 * sub, 4);
            sg = iq3s_pairs_to_signs(sg);
            std::memcpy(out[b].signs + 4 * sub, &sg, 4);
        }
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

int64_t quant_block_bytes(const uint32_t type) {
    switch (type) {
        case 10: return 84;    // Q2_K   (layout only, no kernel yet)
        case 29: return 56;    // IQ1_M  (identity layout: the GGUF bytes)
        case 12: return 144;   // Q4_K
        case 14: return 210;   // Q6_K
        case 16: return 66;    // IQ2_XXS
        case 17: return 74;    // IQ2_XS
        case 22: return 82;    // IQ2_S
        case 18: return 98;    // IQ3_XXS
        case 21: return 110;   // IQ3_S
        case 23: return 136;   // IQ4_XS
        default: return 0;
    }
}

int64_t repacked_bytes(const uint32_t type, const int64_t n_blocks) {
    switch (type) {
        case 10: return q2k_layout(n_blocks).total;
        case 12: return q4k_layout(n_blocks).total;
        case 14: return q6k_layout(n_blocks).total;
        case 16: return iq2_xxs_layout(n_blocks).total;
        case 17: return iq2_xs_layout(n_blocks).total;
        case 18: return iq3_xxs_layout(n_blocks).total;
        case 21: return iq3s_layout(n_blocks).total;
        case 22: return iq2s_layout(n_blocks).total;
        case 23: return iq4_layout(n_blocks).total;
        case 29: return n_blocks * 56;  // IQ1_M: kept as in the GGUF
        default: return 0;
    }
}

bool repack_any(const uint32_t type, const void * src, const int64_t n_blocks,
                std::vector<uint8_t> & dst) {
    if (type == 12) {
        dst.resize((size_t) q4k_layout(n_blocks).total);
        repack_q4k(src, n_blocks, dst.data());
    } else if (type == 23) {
        dst.resize((size_t) iq4_layout(n_blocks).total);
        repack_iq4_xs(src, n_blocks, dst.data());
    } else if (type == 18) {
        dst.resize((size_t) iq3_xxs_layout(n_blocks).total);
        repack_iq3_xxs(src, n_blocks, dst.data());
    } else if (type == 21) {
        dst.resize((size_t) iq3s_layout(n_blocks).total);
        repack_iq3_s(src, n_blocks, dst.data());
    } else if (type == 14) {
        dst.resize((size_t) q6k_layout(n_blocks).total);
        repack_q6k(src, n_blocks, dst.data());
    } else if (type == 17) {
        dst.resize((size_t) iq2_xs_layout(n_blocks).total);
        repack_iq2_xs(src, n_blocks, dst.data());
    } else if (type == 22) {
        dst.resize((size_t) iq2s_layout(n_blocks).total);
        repack_iq2_s(src, n_blocks, dst.data());
    } else if (type == 16) {
        dst.resize((size_t) iq2_xxs_layout(n_blocks).total);
        repack_iq2_xxs(src, n_blocks, dst.data());
    } else if (type == 10) {
        dst.resize((size_t) q2k_layout(n_blocks).total);
        repack_q2k(src, n_blocks, dst.data());
    } else if (type == 29) {
        // IQ1_M: one small tensor in this model; its GEMV reads the GGUF blocks
        // as they are (56-byte blocks, 8-byte aligned fields).
        dst.resize((size_t) n_blocks * 56);
        std::memcpy(dst.data(), src, dst.size());
    } else {
        return false;
    }
    return true;
}

bool unrepack_any(const uint32_t type, const void * src, const int64_t n_blocks,
                  std::vector<uint8_t> & dst) {
    if (type == 12) unrepack_q4k(src, n_blocks, dst.data());
    else if (type == 23) unrepack_iq4_xs(src, n_blocks, dst.data());
    else if (type == 18) unrepack_iq3_xxs(src, n_blocks, dst.data());
    else if (type == 21) unrepack_iq3_s(src, n_blocks, dst.data());
    else if (type == 14) unrepack_q6k(src, n_blocks, dst.data());
    else if (type == 17) unrepack_iq2_xs(src, n_blocks, dst.data());
    else if (type == 22) unrepack_iq2_s(src, n_blocks, dst.data());
    else if (type == 16) unrepack_iq2_xxs(src, n_blocks, dst.data());
    else if (type == 10) unrepack_q2k(src, n_blocks, dst.data());
    else if (type == 29) std::memcpy(dst.data(), src, (size_t) n_blocks * 56);
    else return false;
    return true;
}

// ------------------------------------------------------------------ Q6_K

namespace {
#pragma pack(push, 1)
struct BlockQ6K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};
#pragma pack(pop)
static_assert(sizeof(BlockQ6K) == 210, "Q6_K block must be 210 bytes");
} // namespace

Q6kLayout q6k_layout(const int64_t n_blocks) {
    Q6kLayout l;
    l.n_blocks = n_blocks;
    l.ql_off = 0;
    l.qh_off = align128(n_blocks * 128);
    l.sc_off = align128(l.qh_off + n_blocks * 64);
    l.d_off = align128(l.sc_off + n_blocks * 16);
    l.total = align128(l.d_off + n_blocks * 2);
    return l;
}

void repack_q6k(const void * src, const int64_t n_blocks, void * dst) {
    const Q6kLayout l = q6k_layout(n_blocks);
    const auto * in = static_cast<const BlockQ6K *>(src);
    uint8_t * out = static_cast<uint8_t *>(dst);
    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(out + l.ql_off + b * 128, in[b].ql, 128);
        std::memcpy(out + l.qh_off + b * 64, in[b].qh, 64);
        std::memcpy(out + l.sc_off + b * 16, in[b].scales, 16);
        std::memcpy(out + l.d_off + 2 * b, &in[b].d, 2);
    }
}

void unrepack_q6k(const void * src, const int64_t n_blocks, void * dst) {
    const Q6kLayout l = q6k_layout(n_blocks);
    const uint8_t * in = static_cast<const uint8_t *>(src);
    auto * out = static_cast<BlockQ6K *>(dst);
    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(out[b].ql, in + l.ql_off + b * 128, 128);
        std::memcpy(out[b].qh, in + l.qh_off + b * 64, 64);
        std::memcpy(out[b].scales, in + l.sc_off + b * 16, 16);
        std::memcpy(&out[b].d, in + l.d_off + 2 * b, 2);
    }
}

// ----------------------------------------------------------- IQ2_XS / IQ2_XXS

namespace {
#pragma pack(push, 1)
struct BlockIq2Xs {
    uint16_t d;
    uint16_t qs[32];
    uint8_t scales[8];
};
#pragma pack(pop)
static_assert(sizeof(BlockIq2Xs) == 74, "IQ2_XS block must be 74 bytes");

#pragma pack(push, 1)
struct BlockIq2Xxs {
    uint16_t d;
    uint16_t qs[32];
};
#pragma pack(pop)
static_assert(sizeof(BlockIq2Xxs) == 66, "IQ2_XXS block must be 66 bytes");
} // namespace

Iq2Layout iq2_xs_layout(const int64_t n_blocks) {
    Iq2Layout l;
    l.n_blocks = n_blocks;
    l.qs_off = 0;
    l.sc_off = align128(n_blocks * 64);
    l.d_off = align128(l.sc_off + n_blocks * 8);
    l.total = align128(l.d_off + n_blocks * 2);
    return l;
}

namespace {

// IQ2_XS entry: 9-bit grid index (bits 0-8) and a 7-bit sign index (9-15).
// Repacked, the sign bits are permuted as for IQ3_XXS (#75): weights 0/2/4/6
// reversed in bits 9-12, weights 1/3/5 reversed in bits 13-15. Lossless.
uint16_t iq2xs_entry_to_pairs(const uint16_t e) {
    const uint32_t s = (uint32_t) e >> 9;
    uint32_t t = 0;
    for (int k = 0; k < 4; ++k) {
        t |= ((s >> (2 * k)) & 1u) << (3 - k);
    }
    for (int k = 0; k < 3; ++k) {
        t |= ((s >> (2 * k + 1)) & 1u) << (4 + 2 - k);
    }
    return (uint16_t) ((e & 0x1FFu) | (t << 9));
}

uint16_t iq2xs_pairs_to_entry(const uint16_t e) {
    const uint32_t t = (uint32_t) e >> 9;
    uint32_t s = 0;
    for (int k = 0; k < 4; ++k) {
        s |= ((t >> (3 - k)) & 1u) << (2 * k);
    }
    for (int k = 0; k < 3; ++k) {
        s |= ((t >> (4 + 2 - k)) & 1u) << (2 * k + 1);
    }
    return (uint16_t) ((e & 0x1FFu) | (s << 9));
}

} // namespace

void repack_iq2_xs(const void * src, const int64_t n_blocks, void * dst) {
    const Iq2Layout l = iq2_xs_layout(n_blocks);
    const auto * in = static_cast<const BlockIq2Xs *>(src);
    uint8_t * out = static_cast<uint8_t *>(dst);
    for (int64_t b = 0; b < n_blocks; ++b) {
        for (int e = 0; e < 32; ++e) {
            const uint16_t v = iq2xs_entry_to_pairs(in[b].qs[e]);
            std::memcpy(out + l.qs_off + b * 64 + 2 * e, &v, 2);
        }
        std::memcpy(out + l.sc_off + b * 8, in[b].scales, 8);
        std::memcpy(out + l.d_off + 2 * b, &in[b].d, 2);
    }
}

void unrepack_iq2_xs(const void * src, const int64_t n_blocks, void * dst) {
    const Iq2Layout l = iq2_xs_layout(n_blocks);
    const uint8_t * in = static_cast<const uint8_t *>(src);
    auto * out = static_cast<BlockIq2Xs *>(dst);
    for (int64_t b = 0; b < n_blocks; ++b) {
        for (int e = 0; e < 32; ++e) {
            uint16_t v = 0;
            std::memcpy(&v, in + l.qs_off + b * 64 + 2 * e, 2);
            out[b].qs[e] = iq2xs_pairs_to_entry(v);
        }
        std::memcpy(out[b].scales, in + l.sc_off + b * 8, 8);
        std::memcpy(&out[b].d, in + l.d_off + 2 * b, 2);
    }
}

Iq2Layout iq2_xxs_layout(const int64_t n_blocks) {
    Iq2Layout l;
    l.n_blocks = n_blocks;
    l.qs_off = 0;
    l.sc_off = 0;
    l.d_off = align128(n_blocks * 64);
    l.total = align128(l.d_off + n_blocks * 2);
    return l;
}

void repack_iq2_xxs(const void * src, const int64_t n_blocks, void * dst) {
    const Iq2Layout l = iq2_xxs_layout(n_blocks);
    const auto * in = static_cast<const BlockIq2Xxs *>(src);
    uint8_t * out = static_cast<uint8_t *>(dst);
    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(out + l.qs_off + b * 64, in[b].qs, 64);
        // Each sub-block's second word (four 7-bit sign indices + scale) gets
        // the IQ3_XXS sign permutation (#75); lossless, unrepack inverts it.
        for (int sub = 0; sub < 8; ++sub) {
            uint32_t a = 0;
            std::memcpy(&a, out + l.qs_off + b * 64 + 8 * sub + 4, 4);
            a = iq3xxs_aux_to_pairs(a);
            std::memcpy(out + l.qs_off + b * 64 + 8 * sub + 4, &a, 4);
        }
        std::memcpy(out + l.d_off + 2 * b, &in[b].d, 2);
    }
}

void unrepack_iq2_xxs(const void * src, const int64_t n_blocks, void * dst) {
    const Iq2Layout l = iq2_xxs_layout(n_blocks);
    const uint8_t * in = static_cast<const uint8_t *>(src);
    auto * out = static_cast<BlockIq2Xxs *>(dst);
    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(out[b].qs, in + l.qs_off + b * 64, 64);
        for (int sub = 0; sub < 8; ++sub) {
            uint32_t t = 0;
            std::memcpy(&t, reinterpret_cast<const uint8_t *>(out[b].qs) + 8 * sub + 4, 4);
            t = iq3xxs_pairs_to_aux(t);
            std::memcpy(reinterpret_cast<uint8_t *>(out[b].qs) + 8 * sub + 4, &t, 4);
        }
        std::memcpy(&out[b].d, in + l.d_off + 2 * b, 2);
    }
}

// ------------------------------------------------------------------ IQ2_S

namespace {
#pragma pack(push, 1)
struct BlockIq2S {
    uint16_t d;
    uint8_t qs[64];  // 32 index bytes, then 32 sign bytes
    uint8_t qh[8];
    uint8_t scales[8];
};
#pragma pack(pop)
static_assert(sizeof(BlockIq2S) == 82, "IQ2_S block must be 82 bytes");
} // namespace

Iq2sLayout iq2s_layout(const int64_t n_blocks) {
    Iq2sLayout l;
    l.n_blocks = n_blocks;
    l.qs_off = 0;
    l.signs_off = align128(n_blocks * 32);
    l.qh_off = align128(l.signs_off + n_blocks * 32);
    l.sc_off = align128(l.qh_off + n_blocks * 8);
    l.d_off = align128(l.sc_off + n_blocks * 8);
    l.total = align128(l.d_off + n_blocks * 2);
    return l;
}

void repack_iq2_s(const void * src, const int64_t n_blocks, void * dst) {
    const Iq2sLayout l = iq2s_layout(n_blocks);
    const auto * in = static_cast<const BlockIq2S *>(src);
    uint8_t * out = static_cast<uint8_t *>(dst);
    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(out + l.qs_off + b * 32, in[b].qs, 32);
        for (int sub = 0; sub < 8; ++sub) {  // same sign order as IQ3_S (#73)
            uint32_t sg = 0;
            std::memcpy(&sg, in[b].qs + 32 + 4 * sub, 4);
            sg = iq3s_signs_to_pairs(sg);
            std::memcpy(out + l.signs_off + b * 32 + 4 * sub, &sg, 4);
        }
        std::memcpy(out + l.qh_off + b * 8, in[b].qh, 8);
        std::memcpy(out + l.sc_off + b * 8, in[b].scales, 8);
        std::memcpy(out + l.d_off + 2 * b, &in[b].d, 2);
    }
}

void unrepack_iq2_s(const void * src, const int64_t n_blocks, void * dst) {
    const Iq2sLayout l = iq2s_layout(n_blocks);
    const uint8_t * in = static_cast<const uint8_t *>(src);
    auto * out = static_cast<BlockIq2S *>(dst);
    for (int64_t b = 0; b < n_blocks; ++b) {
        std::memcpy(out[b].qs, in + l.qs_off + b * 32, 32);
        for (int sub = 0; sub < 8; ++sub) {
            uint32_t sg = 0;
            std::memcpy(&sg, in + l.signs_off + b * 32 + 4 * sub, 4);
            sg = iq3s_pairs_to_signs(sg);
            std::memcpy(out[b].qs + 32 + 4 * sub, &sg, 4);
        }
        std::memcpy(out[b].qh, in + l.qh_off + b * 8, 8);
        std::memcpy(out[b].scales, in + l.sc_off + b * 8, 8);
        std::memcpy(&out[b].d, in + l.d_off + 2 * b, 2);
    }
}

} // namespace omph::format