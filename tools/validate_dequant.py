#!/usr/bin/env python3
"""Validate the NumPy dequantization against ggml's own implementation (bit-exact).

Compares `omphalos_tools.quant.dequantize` with the exported
`dequantize_row_*` functions of libggml-base.so (via ctypes) on random blocks,
and, with `--model`, on real tensors read from a GGUF.

usage: uv run python validate_dequant.py [--lib PATH] [--model FILE.gguf]
"""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as GT
from gguf import GGUFReader
from gguf.quants import GGML_QUANT_SIZES

from omphalos_tools import quant

DEFAULT_LIB = Path(
    "/home/alessandro/Projects/llama.cpp-RX9060XT-16GB/build-hip/bin/libggml-base.so"
)

C_SYMBOL = {
    GT.BF16: "ggml_bf16_to_fp32_row",
    GT.Q2_K: "dequantize_row_q2_K",
    GT.Q4_K: "dequantize_row_q4_K",
    GT.Q6_K: "dequantize_row_q6_K",
    GT.IQ1_M: "dequantize_row_iq1_m",
    GT.IQ2_XXS: "dequantize_row_iq2_xxs",
    GT.IQ2_XS: "dequantize_row_iq2_xs",
    GT.IQ2_S: "dequantize_row_iq2_s",
    GT.IQ3_XXS: "dequantize_row_iq3_xxs",
    GT.IQ3_S: "dequantize_row_iq3_s",
    GT.IQ4_XS: "dequantize_row_iq4_xs",
}


def bind(lib: ctypes.CDLL, symbol: str):
    fn = getattr(lib, symbol)
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    fn.restype = None
    return fn


def ref_dequant(fn, raw: bytes, n: int) -> np.ndarray:
    buf = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
    out = np.zeros(n, dtype=np.float32)
    fn(ctypes.cast(buf, ctypes.c_void_p), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
       ctypes.c_int64(n))
    return out


def report(label: str, mine: np.ndarray, ref: np.ndarray) -> bool:
    a = mine.view(np.uint32)
    b = ref.view(np.uint32)
    # fp16 -> fp32 conversion may pick a different quiet-NaN payload than ggml;
    # treat NaN == NaN as a match (real weights never carry NaN scales).
    both_nan = np.isnan(mine) & np.isnan(ref)
    mism = (a != b) & ~both_nan
    if not mism.any():
        return True
    bad = np.flatnonzero(mism)
    print(f"  !! {label}: {bad.size}/{mine.size} differ; first idx "
          f"{bad[:4].tolist()}; mine={mine[bad[:2]].tolist()} ref={ref[bad[:2]].tolist()}")
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate NumPy dequant vs ggml")
    ap.add_argument("--lib", type=Path, default=DEFAULT_LIB)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--blocks", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    lib = ctypes.CDLL(str(args.lib))
    rng = np.random.default_rng(args.seed)
    np.seterr(all="ignore")  # random fp16 scales may be inf/nan; bit patterns still must match

    tensors_by_type: dict[GT, list] = {}
    if args.model is not None:
        reader = GGUFReader(args.model)
        for t in reader.tensors:
            tensors_by_type.setdefault(t.tensor_type, []).append(t)

    print(f"{'type':>9}  {'random':>6}  {'tensor':>6}  details")
    failures = 0
    checked_tensor_types = set()
    for gt in quant.SUPPORTED_TYPES:
        block, tsize = GGML_QUANT_SIZES[gt]
        symbol = C_SYMBOL.get(gt)
        if symbol is None:
            print(f"{gt.name:>9}  {'n/a':>6}  {'n/a':>6}  (no oracle)")
            continue
        fn = bind(lib, symbol)

        raw = rng.integers(0, 256, size=args.blocks * tsize, dtype=np.uint8).tobytes()
        n = args.blocks * block
        ok_random = report(f"{gt.name} random", quant.dequantize(raw, gt, n), ref_dequant(fn, raw, n))

        ok_tensor = True
        details = ""
        if args.model is not None and tensors_by_type.get(gt):
            tensor = tensors_by_type[gt][0]
            blob = np.asarray(tensor.data).view(np.uint8).tobytes()
            n_elem = blob.__len__() // tsize * block
            mine = quant.dequantize(blob, gt, n_elem)
            ref = ref_dequant(fn, blob, n_elem)
            ok_tensor = report(f"{gt.name} tensor {tensor.name}", mine, ref)
            details = f"{tensor.name} ({n_elem} elems)"
            checked_tensor_types.add(gt)

        if not (ok_random and ok_tensor):
            failures += 1
        print(f"{gt.name:>9}  {'OK' if ok_random else 'FAIL':>6}  "
              f"{(('OK' if ok_tensor else 'FAIL') if details else '-'):>6}  {details}")

    print("all types bit-exact" if failures == 0 else f"{failures} type(s) failed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
