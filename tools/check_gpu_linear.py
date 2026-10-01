#!/usr/bin/env python3
"""Check omph-linear (GPU: dequant -> f16 -> hipBLASLt) against NumPy.

The f64 NumPy product is the reference; the GPU path rounds weights and
activations to f16, so the tolerance is ~1e-3 relative.

usage: uv run python check_gpu_linear.py <model.gguf> <weight-tensor> [--tokens N]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from omphalos_tools.golden import run_tool
from omphalos_tools.model import Model


def main() -> None:
    ap = argparse.ArgumentParser(description="check omph-linear vs NumPy")
    ap.add_argument("model")
    ap.add_argument("weight")
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--tool", default="../engine/build/omph-linear")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tol", type=float, default=3e-3)
    args = ap.parse_args()

    model = Model(args.model)
    weight = model.matrix(args.weight)  # (out, in)
    out_features, in_features = weight.shape

    rng = np.random.default_rng(args.seed)
    x = (rng.standard_normal((args.tokens, in_features)).astype(np.float32) * 0.1)
    ref = x.astype(np.float64) @ weight.astype(np.float64).T

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "x.f32"
        out_path = Path(tmp) / "y.f32"
        x.tofile(in_path)
        run_tool([args.tool, args.model, args.weight, str(in_path), str(out_path),
                  str(args.tokens)])
        y = np.fromfile(out_path, dtype=np.float32).reshape(args.tokens, out_features)

    diff = np.abs(y - ref)
    scale = float(np.abs(ref).max())
    rel = float(diff.max()) / scale
    ok = rel <= args.tol
    print(f"{args.weight}: tokens={args.tokens} max|d|={float(diff.max()):.3e} rel={rel:.3e} "
          f"tol={args.tol:g} {'OK' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
