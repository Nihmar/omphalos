"""NumPy dequantization for the ggml quantization types used by the model.

Line-by-line port of the ``dequantize_row_*`` functions of llama.cpp's
ggml-quants.c, preserving the C operation order in float32 so results are
bit-identical to ggml. Validated against libggml by ``tools/validate_dequant.py``.

Supported: F32, BF16, Q2_K, Q4_K, Q6_K, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
IQ3_XXS, IQ3_S, IQ4_XS — i.e. every type present in the IQ3_S-mtp GSQ-RCO file.
"""

from __future__ import annotations

import numpy as np
from gguf import GGMLQuantizationType as GT
from gguf.quants import GGML_QUANT_SIZES

from . import quant_tables as tbl

QK_K = 256
IQ1S_DELTA = np.float32(0.125)


# ----------------------------------------------------------------- helpers


def _f16(cols: np.ndarray) -> np.ndarray:
    """(nb, 2) uint8 -> (nb,) float32  (ggml_half -> float)."""
    return np.ascontiguousarray(cols).view(np.float16).reshape(-1).astype(np.float32)


def _f16b(cols: np.ndarray) -> np.ndarray:
    """(nb, 2) uint8 -> (nb, 1) float32."""
    return _f16(cols)[:, None]


def _bytes_u64(u: np.ndarray, nbytes: int) -> np.ndarray:
    """(...,) uint64 -> (..., nbytes) uint8, little-endian: byte j = (u >> 8j)."""
    shifts = (8 * np.arange(nbytes)).astype(np.uint64)
    return ((u[..., None] >> shifts) & np.uint64(0xFF)).astype(np.uint8)


def _bytes_u32(u: np.ndarray, nbytes: int) -> np.ndarray:
    shifts = (8 * np.arange(nbytes)).astype(np.uint32)
    return ((u[..., None] >> shifts) & np.uint32(0xFF)).astype(np.uint8)


def _apply_sign(mag: np.ndarray, sign_byte: np.ndarray, offset: int = 0) -> np.ndarray:
    """C pattern ``mag * (signs & kmask_iq2xs[j+offset] ? -1 : 1)``.

    mag: (..., n) float32 with n in {4, 8}; sign_byte: (...,) uint8.
    """
    n = mag.shape[-1]
    bits = (sign_byte[..., None] >> (np.arange(n, dtype=np.uint8) + np.uint8(offset))) & 1
    return np.where(bits.astype(bool), -mag, mag)


# ------------------------------------------------------------- K-quants


def dequantize_q2_K(blocks: np.ndarray) -> np.ndarray:
    nb = blocks.shape[0]
    scales = blocks[:, 0:16].reshape(nb, 2, 8)
    q = blocks[:, 16:80].reshape(nb, 2, 32)
    d = _f16b(blocks[:, 80:82])
    dmin = _f16b(blocks[:, 82:84])

    out = np.empty((nb, 2, 8, 16), dtype=np.float32)
    for s in range(4):
        for half in range(2):
            sc = scales[:, :, 2 * s + half]
            dl = d * (sc & np.uint8(0xF)).astype(np.float32)
            ml = dmin * (sc >> np.uint8(4)).astype(np.float32)
            v = ((q[:, :, half * 16:(half + 1) * 16] >> np.uint8(2 * s)) & np.uint8(3)).astype(np.float32)
            out[:, :, 2 * s + half, :] = dl[:, :, None] * v - ml[:, :, None]
    return out.reshape(-1)


def _get_scale_min_k4(sc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """q4_K/q5_K scale+min unpacker, (nb, 12) uint8 -> two (nb, 8) uint8."""
    nb = sc.shape[0]
    d = np.empty((nb, 8), np.uint8)
    m = np.empty((nb, 8), np.uint8)
    for j in range(8):
        if j < 4:
            d[:, j] = sc[:, j] & np.uint8(63)
            m[:, j] = sc[:, j + 4] & np.uint8(63)
        else:
            d[:, j] = (sc[:, j + 4] & np.uint8(0xF)) | ((sc[:, j - 4] >> np.uint8(6)) << np.uint8(4))
            m[:, j] = (sc[:, j + 4] >> np.uint8(4)) | ((sc[:, j] >> np.uint8(6)) << np.uint8(4))
    return d, m


def dequantize_q4_K(blocks: np.ndarray) -> np.ndarray:
    nb = blocks.shape[0]
    d = _f16b(blocks[:, 0:2])
    dmin = _f16b(blocks[:, 2:4])
    sc, sm = _get_scale_min_k4(blocks[:, 4:16])
    dl = d * sc.astype(np.float32)
    ml = dmin * sm.astype(np.float32)
    q = blocks[:, 16:144].reshape(nb, 4, 32)

    out = np.empty((nb, 4, 2, 32), dtype=np.float32)
    for j in range(4):
        out[:, j, 0, :] = dl[:, 2 * j, None] * (q[:, j] & np.uint8(0xF)) - ml[:, 2 * j, None]
        out[:, j, 1, :] = dl[:, 2 * j + 1, None] * (q[:, j] >> np.uint8(4)) - ml[:, 2 * j + 1, None]
    return out.reshape(-1)


def dequantize_q6_K(blocks: np.ndarray) -> np.ndarray:
    nb = blocks.shape[0]
    d = _f16b(blocks[:, 208:210])
    ql = blocks[:, 0:128].reshape(nb, 2, 64)
    qh = blocks[:, 128:192].reshape(nb, 2, 32)
    sc = np.ascontiguousarray(blocks[:, 192:208]).view(np.int8).reshape(nb, 2, 8).astype(np.float32)

    out = np.empty((nb, 2, 4, 32), dtype=np.float32)
    for half in range(2):
        ql_h = ql[:, half]
        qh_h = qh[:, half]
        s = sc[:, half]
        for lo in (0, 16):
            sl = slice(lo, lo + 16)
            qlo = ql_h[:, 32 + lo:48 + lo]
            qh_ = qh_h[:, sl]
            q1 = ((ql_h[:, sl] & np.uint8(0xF)) | (((qh_ >> np.uint8(0)) & np.uint8(3)) << np.uint8(4))).astype(np.float32) - np.float32(32)
            q2 = ((qlo & np.uint8(0xF)) | (((qh_ >> np.uint8(2)) & np.uint8(3)) << np.uint8(4))).astype(np.float32) - np.float32(32)
            q3 = ((ql_h[:, sl] >> np.uint8(4)) | (((qh_ >> np.uint8(4)) & np.uint8(3)) << np.uint8(4))).astype(np.float32) - np.float32(32)
            q4 = ((qlo >> np.uint8(4)) | (((qh_ >> np.uint8(6)) & np.uint8(3)) << np.uint8(4))).astype(np.float32) - np.float32(32)
            k = 0 if lo == 0 else 1
            out[:, half, 0, sl] = (d * s[:, k + 0, None]) * q1
            out[:, half, 1, sl] = (d * s[:, k + 2, None]) * q2
            out[:, half, 2, sl] = (d * s[:, k + 4, None]) * q3
            out[:, half, 3, sl] = (d * s[:, k + 6, None]) * q4
    return out.reshape(-1)


# ---------------------------------------------------------------- IQ2


def dequantize_iq2_xxs(blocks: np.ndarray) -> np.ndarray:
    nb = blocks.shape[0]
    d = _f16b(blocks[:, 0:2])
    field = blocks[:, 2:66].reshape(nb, 8, 8)
    idx = field[:, :, 0:4]
    hi = np.ascontiguousarray(field[:, :, 4:8]).view(np.uint32).reshape(nb, 8)

    db = (d * ((hi >> np.uint32(28)).astype(np.float32) + np.float32(0.5))) * np.float32(0.25)
    mag = _bytes_u64(tbl.IQ2XXS_GRID[idx.astype(np.int64)], 8).astype(np.float32)
    signs = tbl.KSIGNS_IQ2XS[((hi[:, :, None] >> (7 * np.arange(4)).astype(np.uint32)) & np.uint32(127)).astype(np.int64)]
    vals = _apply_sign(mag, signs)
    return (db[:, :, None, None] * vals).reshape(-1)


def dequantize_iq2_xs(blocks: np.ndarray) -> np.ndarray:
    nb = blocks.shape[0]
    d = _f16b(blocks[:, 0:2])
    qs = np.ascontiguousarray(blocks[:, 2:66]).view(np.uint16).reshape(nb, 8, 4)
    scales = blocks[:, 66:74].reshape(nb, 8, 1)

    db0 = (d[:, :, None] * ((scales & np.uint16(0xF)).astype(np.float32) + np.float32(0.5))) * np.float32(0.25)
    db1 = (d[:, :, None] * ((scales >> np.uint16(4)).astype(np.float32) + np.float32(0.5))) * np.float32(0.25)
    db = np.where(np.arange(4) < 2, db0, db1)

    idx = (qs & np.uint16(511)).astype(np.int64)
    signs = tbl.KSIGNS_IQ2XS[((qs >> np.uint16(9)) & np.uint16(127)).astype(np.int64)]
    mag = _bytes_u64(tbl.IQ2XS_GRID[idx], 8).astype(np.float32)
    vals = _apply_sign(mag, signs)
    return (db[:, :, :, None] * vals).reshape(-1)


def dequantize_iq2_s(blocks: np.ndarray) -> np.ndarray:
    nb = blocks.shape[0]
    d = _f16b(blocks[:, 0:2])
    idxb = blocks[:, 2:34].reshape(nb, 8, 4)
    signs = blocks[:, 34:66].reshape(nb, 8, 4)  # qs + QK_K/8: second half of the qs field
    qh = blocks[:, 66:74]
    scales = blocks[:, 74:82].reshape(nb, 8, 1)

    idx = (idxb.astype(np.uint16)
           | ((qh.astype(np.uint16)[:, :, None] << np.array([8, 6, 4, 2], np.uint16)) & np.uint16(0x300)))
    mag = _bytes_u64(tbl.IQ2S_GRID[idx.astype(np.int64)], 8).astype(np.float32)
    vals = _apply_sign(mag, signs)

    db0 = (d[:, :, None] * ((scales & np.uint16(0xF)).astype(np.float32) + np.float32(0.5))) * np.float32(0.25)
    db1 = (d[:, :, None] * ((scales >> np.uint16(4)).astype(np.float32) + np.float32(0.5))) * np.float32(0.25)
    db = np.where(np.arange(4) < 2, db0, db1)
    return (db[:, :, :, None] * vals).reshape(-1)


# ---------------------------------------------------------------- IQ3


def dequantize_iq3_xxs(blocks: np.ndarray) -> np.ndarray:
    nb = blocks.shape[0]
    d = _f16b(blocks[:, 0:2])
    qs = blocks[:, 2:98]
    gsrc = qs[:, 0:64].reshape(nb, 8, 8)
    ss = np.ascontiguousarray(qs[:, 64:96]).view(np.uint32).reshape(nb, 8)

    db = (d * ((ss >> np.uint32(28)).astype(np.float32) + np.float32(0.5))) * np.float32(0.5)
    out = np.empty((nb, 8, 4, 8), dtype=np.float32)
    for l in range(4):
        signs = tbl.KSIGNS_IQ2XS[((ss >> np.uint32(7 * l)) & np.uint32(127)).astype(np.int64)]
        g1 = tbl.IQ3XXS_GRID[gsrc[:, :, 2 * l].astype(np.int64)]
        g2 = tbl.IQ3XXS_GRID[gsrc[:, :, 2 * l + 1].astype(np.int64)]
        v1 = _bytes_u32(g1, 4).astype(np.float32)
        v2 = _bytes_u32(g2, 4).astype(np.float32)
        out[:, :, l, 0:4] = _apply_sign(v1, signs, 0)
        out[:, :, l, 4:8] = _apply_sign(v2, signs, 4)
    return (db[:, :, None, None] * out).reshape(-1)


def dequantize_iq3_s(blocks: np.ndarray) -> np.ndarray:
    nb = blocks.shape[0]
    d = _f16b(blocks[:, 0:2])
    qs = blocks[:, 2:66].reshape(nb, 8, 8)
    qh = blocks[:, 66:74]
    signs = blocks[:, 74:106].reshape(nb, 8, 4)
    scales = blocks[:, 106:110]

    out = np.empty((nb, 4, 2, 4, 8), dtype=np.float32)
    for k in range(4):
        scb = scales[:, k]
        db1 = d * (np.float32(1) + np.float32(2) * (scb & np.uint8(0xF)).astype(np.float32))[:, None]
        db2 = d * (np.float32(1) + np.float32(2) * (scb >> np.uint8(4)).astype(np.float32))[:, None]
        for half in range(2):
            qhb = qh[:, 2 * k + half].astype(np.uint16)
            qsb = qs[:, 2 * k + half]
            sgb = signs[:, 2 * k + half]
            db = db1 if half == 0 else db2
            for l in range(4):
                g1 = tbl.IQ3S_GRID[(qsb[:, 2 * l].astype(np.uint16)
                                    | ((qhb << np.uint16(8 - 2 * l)) & np.uint16(0x100))).astype(np.int64)]
                g2 = tbl.IQ3S_GRID[(qsb[:, 2 * l + 1].astype(np.uint16)
                                    | ((qhb << np.uint16(7 - 2 * l)) & np.uint16(0x100))).astype(np.int64)]
                v1 = _bytes_u32(g1, 4).astype(np.float32)
                v2 = _bytes_u32(g2, 4).astype(np.float32)
                out[:, k, half, l, 0:4] = _apply_sign(v1, sgb[:, l], 0)
                out[:, k, half, l, 4:8] = _apply_sign(v2, sgb[:, l], 4)
            out[:, k, half] *= db[:, :, None]
    return out.reshape(-1)


# ---------------------------------------------------------------- IQ1 / IQ4


def dequantize_iq1_m(blocks: np.ndarray) -> np.ndarray:
    nb = blocks.shape[0]
    qs = blocks[:, 0:32]
    qh = blocks[:, 32:48]
    sc16 = np.ascontiguousarray(blocks[:, 48:56]).view(np.uint16).reshape(nb, 4)

    scale_u16 = ((sc16[:, 0] >> np.uint16(12))
                 | ((sc16[:, 1] >> np.uint16(8)) & np.uint16(0x00F0))
                 | ((sc16[:, 2] >> np.uint16(4)) & np.uint16(0x0F00))
                 | (sc16[:, 3] & np.uint16(0xF000)))
    d = scale_u16.view(np.float16).astype(np.float32)

    out = np.empty((nb, 8, 4, 8), dtype=np.float32)
    for ib in range(8):
        sh = 6 * (ib % 2)
        dl1 = d * (np.float32(2) * ((sc16[:, ib // 2] >> np.uint16(sh + 0)) & np.uint16(7)).astype(np.float32) + np.float32(1))
        dl2 = d * (np.float32(2) * ((sc16[:, ib // 2] >> np.uint16(sh + 3)) & np.uint16(7)).astype(np.float32) + np.float32(1))
        qh0 = qh[:, 2 * ib + 0].astype(np.uint16)
        qh1 = qh[:, 2 * ib + 1].astype(np.uint16)
        idx0 = qs[:, 4 * ib + 0].astype(np.uint16) | ((qh0 << np.uint16(8)) & np.uint16(0x700))
        idx1 = qs[:, 4 * ib + 1].astype(np.uint16) | ((qh0 << np.uint16(4)) & np.uint16(0x700))
        idx2 = qs[:, 4 * ib + 2].astype(np.uint16) | ((qh1 << np.uint16(8)) & np.uint16(0x700))
        idx3 = qs[:, 4 * ib + 3].astype(np.uint16) | ((qh1 << np.uint16(4)) & np.uint16(0x700))
        d0 = np.where((qh[:, 2 * ib + 0] & np.uint8(0x08)) != 0, -IQ1S_DELTA, IQ1S_DELTA)
        d1 = np.where((qh[:, 2 * ib + 0] & np.uint8(0x80)) != 0, -IQ1S_DELTA, IQ1S_DELTA)
        d2 = np.where((qh[:, 2 * ib + 1] & np.uint8(0x08)) != 0, -IQ1S_DELTA, IQ1S_DELTA)
        d3 = np.where((qh[:, 2 * ib + 1] & np.uint8(0x80)) != 0, -IQ1S_DELTA, IQ1S_DELTA)
        for l, (idx, delta, dl) in enumerate(((idx0, d0, dl1), (idx1, d1, dl1), (idx2, d2, dl2), (idx3, d3, dl2))):
            g = _bytes_u64(tbl.IQ1S_GRID[idx.astype(np.int64)], 8).view(np.int8).astype(np.float32)
            out[:, ib, l, :] = dl[:, None] * (g + delta[:, None])
    return out.reshape(-1)


def dequantize_iq4_xs(blocks: np.ndarray) -> np.ndarray:
    nb = blocks.shape[0]
    d = _f16(blocks[:, 0:2])
    scales_h = np.ascontiguousarray(blocks[:, 2:4]).view(np.uint16).reshape(nb)
    scales_l = blocks[:, 4:8]
    qs = blocks[:, 8:136].reshape(nb, 8, 16)
    kv = tbl.KVALUES_IQ4NL.astype(np.float32)

    out = np.empty((nb, 8, 32), dtype=np.float32)
    for ib in range(8):
        ls = (((scales_l[:, ib // 2] >> np.uint8(4 * (ib % 2))) & np.uint8(0xF)).astype(np.uint16)
              | (((scales_h >> np.uint16(2 * ib)) & np.uint16(3)) << np.uint16(4)))
        dl = d * (ls.astype(np.int32) - np.int32(32)).astype(np.float32)
        out[:, ib, 0:16] = dl[:, None] * kv[(qs[:, ib] & np.uint8(0xF)).astype(np.int64)]
        out[:, ib, 16:32] = dl[:, None] * kv[(qs[:, ib] >> np.uint8(4)).astype(np.int64)]
    return out.reshape(-1)


# ------------------------------------------------------------- bf16 / f32


def dequantize_bf16(blocks: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(blocks).view(np.uint16).reshape(-1)
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def dequantize_f32(blocks: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(blocks).view(np.float32).reshape(-1)


# ------------------------------------------------------------- dispatcher


_DEQUANTIZERS = {
    GT.F32: dequantize_f32,
    GT.BF16: dequantize_bf16,
    GT.Q2_K: dequantize_q2_K,
    GT.Q4_K: dequantize_q4_K,
    GT.Q6_K: dequantize_q6_K,
    GT.IQ1_M: dequantize_iq1_m,
    GT.IQ2_XXS: dequantize_iq2_xxs,
    GT.IQ2_XS: dequantize_iq2_xs,
    GT.IQ2_S: dequantize_iq2_s,
    GT.IQ3_XXS: dequantize_iq3_xxs,
    GT.IQ3_S: dequantize_iq3_s,
    GT.IQ4_XS: dequantize_iq4_xs,
}

SUPPORTED_TYPES = tuple(_DEQUANTIZERS)


def dequantize(raw, ggml_type: GT, n_elements: int | None = None) -> np.ndarray:
    """Dequantize raw ggml bytes/array of one tensor into float32."""
    data = (np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray, memoryview))
            else np.asarray(raw, dtype=np.uint8).reshape(-1))
    _, type_size = GGML_QUANT_SIZES[ggml_type]
    if data.size % type_size != 0:
        raise ValueError(f"{data.size} bytes is not a multiple of the {ggml_type.name} block size {type_size}")
    blocks = data.reshape(-1, type_size)
    out = _DEQUANTIZERS[ggml_type](blocks)
    if n_elements is not None:
        out = out[:n_elements]
    return out
