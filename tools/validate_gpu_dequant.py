#!/usr/bin/env python3
"""Compare an `omph-dequant` dump against the NumPy dequantization.

The NumPy port is bit-exact vs ggml (tools/validate_dequant.py), so a bit-exact
GPU dump validates the HIP kernels end to end.

usage: uv run python validate_gpu_dequant.py <model.gguf> <tensor> <dump.raw> [--f16]
"""

from __future__ import annotations

import argparse

import numpy as np

from omphalos_tools.model import Model


def main() -> None:
    ap = argparse.ArgumentParser(description="validate GPU dequant dump")
    ap.add_argument("model")
    ap.add_argument("tensor")
    ap.add_argument("dump")
    ap.add_argument("--f16", action="store_true")
    args = ap.parse_args()

    model = Model(args.model)
    ref = model.dequant(args.tensor).astype(np.float32)
    if args.f16:
        mine = np.fromfile(args.dump, dtype=np.float16).astype(np.float32)
        expect = ref.astype(np.float16).astype(np.float32)
    else:
        mine = np.fromfile(args.dump, dtype=np.float32)
        expect = ref

    if mine.size != expect.size:
        print(f"{args.tensor}: size mismatch {mine.size} vs {expect.size}")
        raise SystemExit(1)

    mismatch = mine.view(np.uint32) != expect.view(np.uint32)
    bad = int(mismatch.sum())
    d = np.abs(mine - expect)
    scale = float(np.abs(expect).max()) or 1.0
    print(f"{args.tensor}: n={mine.size} bit-exact={bad == 0} "
          f"mismatches={bad} max|d|={float(d.max()):.3e} rel={float(d.max()) / scale:.3e}")
    if bad:
        idx = np.flatnonzero(mismatch)[:4]
        print(f"  first: {idx.tolist()} mine={mine[idx].tolist()} ref={expect[idx].tolist()}")
    raise SystemExit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
