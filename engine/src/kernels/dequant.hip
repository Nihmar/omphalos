// Dequantization kernels: ggml quantized blocks -> f32/f16 (PLAN.md §9, §10).
//
// One thread per output element; the index math mirrors `dequantize_row_*`
// from ggml-quants.c so the results match the reference bit for bit (verified
// against the NumPy port, which is itself bit-exact vs ggml).
#include "kernels/dequant.hh"

#include "kernels/iq_tables.hh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace omph::kernels {
namespace {

using namespace omph::quant;

__device__ __forceinline__ float f16_at(const uint8_t * p) {
    return __half2float(*reinterpret_cast<const __half *>(p));
}

__device__ __forceinline__ uint16_t u16_at(const uint8_t * p) {
    return *reinterpret_cast<const uint16_t *>(p);
}

__device__ __forceinline__ uint32_t u32_at(const uint8_t * p) {
    return *reinterpret_cast<const uint32_t *>(p);
}

__device__ __forceinline__ float signed_mag(float v, bool neg) {
    return neg ? -v : v;
}

template <typename T>
__device__ __forceinline__ T to_out(float v);

template <>
__device__ __forceinline__ float to_out<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ __half to_out<__half>(float v) {
    return __float2half(v);
}

__device__ __forceinline__ long long thread_index() {
    return (long long) blockIdx.x * blockDim.x + threadIdx.x;
}

// --------------------------------------------------------------- F32 / BF16

template <typename T>
__global__ void dequant_f32_kernel(const float * __restrict__ src, T * __restrict__ dst,
                                   const long long n) {
    const long long e = thread_index();
    if (e < n) {
        dst[e] = to_out<T>(src[e]);
    }
}

template <typename T>
__global__ void dequant_bf16_kernel(const uint16_t * __restrict__ src, T * __restrict__ dst,
                                    const long long n) {
    const long long e = thread_index();
    if (e >= n) {
        return;
    }
    const uint32_t bits = (uint32_t) src[e] << 16;  // exact widening
    float v;
    __builtin_memcpy(&v, &bits, 4);
    dst[e] = to_out<T>(v);
}

// -------------------------------------------------------------------- Q2_K

template <typename T>
__global__ void dequant_q2_k_kernel(const uint8_t * __restrict__ src, T * __restrict__ dst,
                                    const long long n) {
    const long long e = thread_index();
    if (e >= n) {
        return;
    }
    const uint8_t * blk = src + (e >> 8) * 84;
    const int r = (int) (e & 255);
    const int n_half = r >> 7;
    const int rr = r & 127;
    const int group = rr >> 5;
    const int within = rr & 31;
    const int half = within >> 4;
    const int l = within & 15;

    const float d = f16_at(blk + 80);
    const float dmin = f16_at(blk + 82);
    const uint8_t sc = blk[n_half * 8 + group * 2 + half];
    const uint8_t q = blk[16 + n_half * 32 + half * 16 + l];

    const float dl = d * (float) (sc & 15);
    const float ml = dmin * (float) (sc >> 4);
    dst[e] = to_out<T>(dl * (float) ((q >> (group * 2)) & 3) - ml);
}

// -------------------------------------------------------------------- Q4_K

__device__ __forceinline__ void scale_min_k4(const int j, const uint8_t * q, uint8_t & d,
                                             uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (uint8_t) ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = (uint8_t) ((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

template <typename T>
__global__ void dequant_q4_k_kernel(const uint8_t * __restrict__ src, T * __restrict__ dst,
                                    const long long n) {
    const long long e = thread_index();
    if (e >= n) {
        return;
    }
    const uint8_t * blk = src + (e >> 8) * 144;
    const int r = (int) (e & 255);
    const int j = r >> 6;
    const int within = r & 63;
    const int high = within >> 5;
    const int l = within & 31;

    const float d = f16_at(blk);
    const float dmin = f16_at(blk + 2);
    uint8_t s = 0;
    uint8_t m = 0;
    scale_min_k4(2 * j + high, blk + 4, s, m);
    const uint8_t q = blk[16 + j * 32 + l];

    const float dl = d * (float) s;
    const float ml = dmin * (float) m;
    dst[e] = to_out<T>(dl * (float) (high ? (q >> 4) : (q & 15)) - ml);
}

// -------------------------------------------------------------------- Q6_K

template <typename T>
__global__ void dequant_q6_k_kernel(const uint8_t * __restrict__ src, T * __restrict__ dst,
                                    const long long n) {
    const long long e = thread_index();
    if (e >= n) {
        return;
    }
    const uint8_t * blk = src + (e >> 8) * 210;
    const int r = (int) (e & 255);
    const int half = r >> 7;
    const int rr = r & 127;
    const int group = rr >> 5;
    const int l = rr & 31;
    const int is = l >> 4;

    const float d = f16_at(blk + 208);
    const uint8_t ql0 = blk[half * 64 + l];
    const uint8_t ql1 = blk[half * 64 + 32 + l];
    const uint8_t qh = blk[128 + half * 32 + l];
    const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192);

    int q;
    if (group == 0) {
        q = (ql0 & 0xF) | (((qh >> 0) & 3) << 4);
    } else if (group == 1) {
        q = (ql1 & 0xF) | (((qh >> 2) & 3) << 4);
    } else if (group == 2) {
        q = (ql0 >> 4) | (((qh >> 4) & 3) << 4);
    } else {
        q = (ql1 >> 4) | (((qh >> 6) & 3) << 4);
    }
    const float s = (float) sc[half * 8 + is + 2 * group];
    dst[e] = to_out<T>(d * s * (float) (q - 32));
}

// ----------------------------------------------------------------- IQ4_XS

template <typename T>
__global__ void dequant_iq4_xs_kernel(const uint8_t * __restrict__ src, T * __restrict__ dst,
                                      const long long n) {
    const long long e = thread_index();
    if (e >= n) {
        return;
    }
    const uint8_t * blk = src + (e >> 8) * 136;
    const int r = (int) (e & 255);
    const int ib = r >> 5;
    const int j = r & 31;

    const float d = f16_at(blk);
    const uint16_t scales_h = u16_at(blk + 2);
    const uint8_t * scales_l = blk + 4;
    const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                   (((scales_h >> (2 * ib)) & 3) << 4);
    const float dl = d * (float) (ls - 32);
    const uint8_t q = blk[8 + ib * 16 + (j & 15)];
    const int idx = (j < 16) ? (q & 15) : (q >> 4);
    dst[e] = to_out<T>(dl * (float) kKvaluesIq4nl[idx]);
}

// ---------------------------------------------------------------- IQ2_XXS

template <typename T>
__global__ void dequant_iq2_xxs_kernel(const uint8_t * __restrict__ src, T * __restrict__ dst,
                                       const long long n) {
    const long long e = thread_index();
    if (e >= n) {
        return;
    }
    const uint8_t * blk = src + (e >> 8) * 66;
    const int r = (int) (e & 255);
    const int ib32 = r >> 5;
    const int j = r & 31;
    const int l = j >> 3;
    const int jj = j & 7;

    const float d = f16_at(blk);
    const uint32_t lo = u32_at(blk + 2 + 8 * ib32);
    const uint32_t hi = u32_at(blk + 2 + 8 * ib32 + 4);
    const uint8_t idx = (uint8_t) ((lo >> (8 * l)) & 0xFF);
    const uint8_t signs = kKsignsIq2xs[(hi >> (7 * l)) & 127];
    const float db = (d * (0.5f + (float) (hi >> 28))) * 0.25f;
    const uint64_t grid = kIq2xxsGrid[idx];
    const float g = (float) ((grid >> (8 * jj)) & 0xFF);
    dst[e] = to_out<T>(signed_mag(db * g, (signs & kKmaskIq2xs[jj]) != 0));
}

// ----------------------------------------------------------------- IQ2_XS

template <typename T>
__global__ void dequant_iq2_xs_kernel(const uint8_t * __restrict__ src, T * __restrict__ dst,
                                      const long long n) {
    const long long e = thread_index();
    if (e >= n) {
        return;
    }
    const uint8_t * blk = src + (e >> 8) * 74;
    const int r = (int) (e & 255);
    const int ib32 = r >> 5;
    const int j = r & 31;
    const int l = j >> 3;
    const int jj = j & 7;

    const float d = f16_at(blk);
    const uint16_t q16 = u16_at(blk + 2 + 2 * (4 * ib32 + l));
    const uint8_t sc = blk[66 + ib32];
    const float db = (d * (0.5f + (float) ((l < 2) ? (sc & 15) : (sc >> 4)))) * 0.25f;
    const uint64_t grid = kIq2xsGrid[q16 & 511];
    const uint8_t signs = kKsignsIq2xs[q16 >> 9];
    const float g = (float) ((grid >> (8 * jj)) & 0xFF);
    dst[e] = to_out<T>(signed_mag(db * g, (signs & kKmaskIq2xs[jj]) != 0));
}

// ------------------------------------------------------------------ IQ2_S

template <typename T>
__global__ void dequant_iq2_s_kernel(const uint8_t * __restrict__ src, T * __restrict__ dst,
                                     const long long n) {
    const long long e = thread_index();
    if (e >= n) {
        return;
    }
    const uint8_t * blk = src + (e >> 8) * 82;
    const int r = (int) (e & 255);
    const int ib32 = r >> 5;
    const int j = r & 31;
    const int l = j >> 3;
    const int jj = j & 7;

    const float d = f16_at(blk);
    const uint8_t * qs = blk + 2;
    const uint8_t * qh = blk + 66;
    const uint8_t * scales = blk + 74;

    const uint16_t idx =
        (uint16_t) (qs[4 * ib32 + l] | ((qh[ib32] << (8 - 2 * l)) & 0x300));
    const uint8_t sign_byte = qs[32 + 4 * ib32 + l];
    const uint8_t sc = scales[ib32];
    const float db = (d * (0.5f + (float) ((l < 2) ? (sc & 15) : (sc >> 4)))) * 0.25f;
    const uint64_t grid = kIq2sGrid[idx];
    const float g = (float) ((grid >> (8 * jj)) & 0xFF);
    dst[e] = to_out<T>(signed_mag(db * g, (sign_byte & kKmaskIq2xs[jj]) != 0));
}

// ---------------------------------------------------------------- IQ3_XXS

template <typename T>
__global__ void dequant_iq3_xxs_kernel(const uint8_t * __restrict__ src, T * __restrict__ dst,
                                       const long long n) {
    const long long e = thread_index();
    if (e >= n) {
        return;
    }
    const uint8_t * blk = src + (e >> 8) * 98;
    const int r = (int) (e & 255);
    const int ib32 = r >> 5;
    const int j = r & 31;
    const int l = j >> 3;
    const int jj = j & 7;

    const float d = f16_at(blk);
    const uint8_t * qs = blk + 2;
    const uint32_t ss = u32_at(qs + 64 + 4 * ib32);
    const float db = (d * (0.5f + (float) (ss >> 28))) * 0.5f;
    const uint8_t sign_byte = kKsignsIq2xs[(ss >> (7 * l)) & 127];
    const uint8_t gidx = qs[8 * ib32 + 2 * l + ((jj < 4) ? 0 : 1)];
    const uint32_t grid = kIq3xxsGrid[gidx];
    const float g = (float) ((grid >> (8 * (jj & 3))) & 0xFF);
    dst[e] = to_out<T>(signed_mag(db * g, (sign_byte & kKmaskIq2xs[jj]) != 0));
}

// ----------------------------------------------------------------- IQ3_S

template <typename T>
__global__ void dequant_iq3_s_kernel(const uint8_t * __restrict__ src, T * __restrict__ dst,
                                     const long long n) {
    const long long e = thread_index();
    if (e >= n) {
        return;
    }
    const uint8_t * blk = src + (e >> 8) * 110;
    const int r = (int) (e & 255);
    const int k = r >> 6;
    const int rr = r & 63;
    const int half = rr >> 5;
    const int rem = rr & 31;
    const int l = rem >> 3;
    const int jj = rem & 7;

    const float d = f16_at(blk);
    const uint8_t * qs = blk + 2;
    const uint8_t * qh = blk + 66;
    const uint8_t * signs = blk + 74;
    const uint8_t * scales = blk + 106;

    const uint8_t sc = scales[k];
    const float db = d * (1.0f + 2.0f * (float) ((half == 0) ? (sc & 15) : (sc >> 4)));
    const uint8_t qh_b = qh[2 * k + half];
    const int shift = (jj < 4) ? (8 - 2 * l) : (7 - 2 * l);
    const uint16_t gidx =
        (uint16_t) (qs[16 * k + 8 * half + 2 * l + ((jj < 4) ? 0 : 1)] |
                    ((qh_b << shift) & 0x100));
    const uint8_t sign_byte = signs[8 * k + 4 * half + l];
    const uint32_t grid = kIq3sGrid[gidx];
    const float g = (float) ((grid >> (8 * (jj & 3))) & 0xFF);
    dst[e] = to_out<T>(signed_mag(db * g, (sign_byte & kKmaskIq2xs[jj]) != 0));
}

// ----------------------------------------------------------------- IQ1_M

template <typename T>
__global__ void dequant_iq1_m_kernel(const uint8_t * __restrict__ src, T * __restrict__ dst,
                                     const long long n) {
    const long long e = thread_index();
    if (e >= n) {
        return;
    }
    const uint8_t * blk = src + (e >> 8) * 56;
    const int r = (int) (e & 255);
    const int ib = r >> 5;
    const int j = r & 31;
    const int l = j >> 3;
    const int jj = j & 7;

    const uint8_t * qs = blk;
    const uint8_t * qh = blk + 32;
    const uint16_t * sc16 = reinterpret_cast<const uint16_t *>(blk + 48);

    const uint16_t scale_u16 = (uint16_t) ((sc16[0] >> 12) | ((sc16[1] >> 8) & 0x00F0) |
                                           ((sc16[2] >> 4) & 0x0F00) | (sc16[3] & 0xF000));
    const float d = __half2float(*reinterpret_cast<const __half *>(&scale_u16));

    const int sh = 6 * (ib % 2) + 3 * (l >> 1);
    const float dl = d * (2.0f * (float) ((sc16[ib >> 1] >> sh) & 7) + 1.0f);

    const uint8_t qh_a = qh[2 * ib + 0];
    const uint8_t qh_b = qh[2 * ib + 1];
    const int shift = ((l & 1) == 0) ? 8 : 4;
    const uint8_t qh_sel = ((l >> 1) == 0) ? qh_a : qh_b;
    const uint16_t idx = (uint16_t) (qs[4 * ib + l] | ((qh_sel << shift) & 0x700));

    bool neg;
    if (l == 0) {
        neg = (qh_a & 0x08) != 0;
    } else if (l == 1) {
        neg = (qh_a & 0x80) != 0;
    } else if (l == 2) {
        neg = (qh_b & 0x08) != 0;
    } else {
        neg = (qh_b & 0x80) != 0;
    }
    const float delta = neg ? -0.125f : 0.125f;
    const uint64_t grid = kIq1sGrid[idx];
    const int8_t g = (int8_t) ((grid >> (8 * jj)) & 0xFF);
    dst[e] = to_out<T>(dl * ((float) g + delta));
}

// -------------------------------------------------------------- dispatcher

template <typename T>
bool launch(const uint32_t type, const void * src, void * dst, const int64_t n,
            hipStream_t stream) {
    const auto * s = static_cast<const uint8_t *>(src);
    auto * d = static_cast<T *>(dst);
    const long long ll = (long long) n;
    const unsigned grid = (unsigned) ((ll + kDequantBlock - 1) / kDequantBlock);

    if (type == 0) {  // F32
        dequant_f32_kernel<T><<<grid, kDequantBlock, 0, stream>>>(reinterpret_cast<const float *>(s), d, ll);
        return true;
    }
    if (type == 30) {  // BF16
        dequant_bf16_kernel<T><<<grid, kDequantBlock, 0, stream>>>(reinterpret_cast<const uint16_t *>(s), d, ll);
        return true;
    }
    if (n % 256 != 0) {
        return false;
    }
    switch (type) {
        case 10: dequant_q2_k_kernel<T><<<grid, kDequantBlock, 0, stream>>>(s, d, ll); return true;
        case 12: dequant_q4_k_kernel<T><<<grid, kDequantBlock, 0, stream>>>(s, d, ll); return true;
        case 14: dequant_q6_k_kernel<T><<<grid, kDequantBlock, 0, stream>>>(s, d, ll); return true;
        case 16: dequant_iq2_xxs_kernel<T><<<grid, kDequantBlock, 0, stream>>>(s, d, ll); return true;
        case 17: dequant_iq2_xs_kernel<T><<<grid, kDequantBlock, 0, stream>>>(s, d, ll); return true;
        case 18: dequant_iq3_xxs_kernel<T><<<grid, kDequantBlock, 0, stream>>>(s, d, ll); return true;
        case 21: dequant_iq3_s_kernel<T><<<grid, kDequantBlock, 0, stream>>>(s, d, ll); return true;
        case 22: dequant_iq2_s_kernel<T><<<grid, kDequantBlock, 0, stream>>>(s, d, ll); return true;
        case 23: dequant_iq4_xs_kernel<T><<<grid, kDequantBlock, 0, stream>>>(s, d, ll); return true;
        case 29: dequant_iq1_m_kernel<T><<<grid, kDequantBlock, 0, stream>>>(s, d, ll); return true;
        default: return false;
    }
}

} // namespace

namespace {

// Rows ids[t] of a table (row_bytes each, host-mapped or device) to dst back
// to back, one workgroup per row (#223).
__global__ void gather_rows_kernel(const uint8_t * __restrict__ table, const long long row_bytes,
                                   const int32_t * __restrict__ ids, uint8_t * __restrict__ dst) {
    const long long t = blockIdx.x;
    const uint8_t * s = table + (long long) ids[t] * row_bytes;
    uint8_t * d = dst + t * row_bytes;
    if (row_bytes % 4 == 0) {
        for (long long i = threadIdx.x; i < row_bytes / 4; i += blockDim.x) {
            reinterpret_cast<uint32_t *>(d)[i] = reinterpret_cast<const uint32_t *>(s)[i];
        }
    } else {
        for (long long i = threadIdx.x; i < row_bytes; i += blockDim.x) {
            d[i] = s[i];
        }
    }
}

} // namespace

bool dequantize_rows(const uint32_t type, const void * table, const int64_t row_bytes, const int32_t * ids,
                     const int64_t n_rows, const int64_t ne, void * staging, float * dst, hipStream_t stream) {
    if (table == nullptr || ids == nullptr || staging == nullptr || dst == nullptr || n_rows <= 0 ||
        row_bytes <= 0 || ne <= 0) {
        return false;
    }
    gather_rows_kernel<<<(unsigned) n_rows, 256, 0, stream>>>(static_cast<const uint8_t *>(table),
                                                              (long long) row_bytes, ids,
                                                              static_cast<uint8_t *>(staging));
    return hipGetLastError() == hipSuccess && dequantize(type, staging, dst, n_rows * ne, false, stream);
}

bool dequantize(const uint32_t type, const void * src, void * dst, const int64_t n,
                const bool out_f16, hipStream_t stream) {
    if (src == nullptr || dst == nullptr || n <= 0) {
        return false;
    }
    return out_f16 ? launch<__half>(type, src, dst, n, stream)
                   : launch<float>(type, src, dst, n, stream);
}

} // namespace omph::kernels
