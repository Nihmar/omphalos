#!/usr/bin/env python3
"""Compare omph-attn (GPU attention block, the engine's kernels) against the golden dump.

usage: uv run python check_gpu_attn.py <model.gguf> <layer> [--kv f32|q8q4|q4q4]
                                       [--window N] [--chunk N]

--kv f32 (default) checks every intermediate; with a quantized cache q and k are
Hadamard-rotated, so only the gate, the attention output and the block output
are compared. --window 0 sends every key through the quantized blocks;
--chunk 1 runs the decode path (one token per call, split-K attention).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from omphalos_tools.golden import load, load_index, rel_diff, run_tool

PAIRS = (
    ("q_full", "Qcur_full"),
    ("q", "Qcur"),
    ("k", "Kcur"),
    ("gate", "gate_reshaped"),
    ("attn_gated", "attn_gated"),
)
ROTATED = {"q", "k"}  # in the Hadamard basis with a quantized cache


def main() -> None:
    ap = argparse.ArgumentParser(description="check the GPU attention block")
    ap.add_argument("model")
    ap.add_argument("layer", type=int)
    ap.add_argument("--dump", default="../models/golden/cpu")
    ap.add_argument("--tool", default="../engine/build/omph-attn")
    ap.add_argument("--tol", type=float, default=None,
                    help="relative; default 0.08 for f32 (llama.cpp's own path is at ~2e-2), "
                         "0.15 with a quantized cache (its end-to-end KL is validated "
                         "separately, #58 / #61)")
    ap.add_argument("--kv", choices=("f32", "q8q4", "q4q4"), default="f32")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=0)
    args = ap.parse_args()
    pairs = [p for p in PAIRS if args.kv == "f32" or p[0] not in ROTATED]
    if args.tol is None:
        args.tol = 8e-2 if args.kv == "f32" else 0.15

    dump = Path(args.dump)
    idx = load_index(dump)
    il = args.layer

    x_entry = idx[f"attn_norm-{il}"]  # layer input (post attn_norm) -> we feed the same
    x = load(dump, x_entry)
    tokens = int(np.prod(x.shape[:-1]))
    x = x.reshape(tokens, -1)

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.f32"
        prefix = str(Path(tmp) / "t")
        np.ascontiguousarray(x, dtype=np.float32).tofile(in_path)
        run_tool([args.tool, args.model, str(il), str(in_path), prefix, str(tokens),
                        "--trace", "--kv", args.kv, "--window", str(args.window),
                        "--chunk", str(args.chunk)])
        got = {}
        for name, _ in pairs:
            got[name] = np.fromfile(f"{prefix}-{name}.f32", dtype=np.float32)
        out = np.fromfile(f"{prefix}.out.f32", dtype=np.float32)

    print(f"kv {args.kv}, window {args.window}, chunk {args.chunk or tokens}, {tokens} tokens")
    worst = (0.0, "")
    for name, dump_name in pairs:
        entry = idx.get(f"{dump_name}-{il}")
        if entry is None:
            print(f"  {name:12} -> {dump_name}-{il}: (not in dump)")
            continue
        ref = load(dump, entry)
        mine = got[name]
        if mine.size == ref.size:
            mine = mine.reshape(ref.shape)
        rel = rel_diff(mine, ref)
        worst = max(worst, (rel, name), key=lambda t: t[0])
        print(f"  {name:12} -> {dump_name}-{il}: rel={rel:.3e}  shape={ref.shape}")
    ref = load(dump, idx[f"attn_output-{il}"])
    rel = rel_diff(out.reshape(ref.shape), ref)
    worst = max(worst, (rel, "out"), key=lambda t: t[0])
    print(f"  {'out':12} -> attn_output-{il}: rel={rel:.3e}  shape={ref.shape}")
    print(f"worst: {worst[1]} rel={worst[0]:.3e} (tol {args.tol:g})")
    raise SystemExit(0 if worst[0] <= args.tol else 1)


if __name__ == "__main__":
    main()
