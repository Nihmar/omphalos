#!/usr/bin/env python3
"""Compare omph-attn (GPU attention block, the engine's kernels) against the golden dump.

usage: uv run python check_gpu_attn.py <model.gguf> <layer> [--kv f32|q8q4|q4q4]
                                       [--window N] [--chunk N] [--verify]

--kv f32 (default) checks every intermediate; with a quantized cache q and k are
Hadamard-rotated, so only the gate, the attention output and the block output
are compared. --window 0 sends every key through the quantized blocks;
--chunk 1 runs the decode path (one token per call, split-K attention).
--verify runs N = 2..tokens tokens in one call against the decode path with the
runner's fixed key chunks: bit-identical up to 8 (the engine's own invariant,
#161 -- a verification's rows are decode steps'), and within --tol up to 16,
where the kernel leaves the decode path's fixed chunks (#315).
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
    ap.add_argument("--verify", action="store_true",
                    help="N = 2..tokens tokens per call against the decode path, with the "
                         "runner's key chunks: bit-identical up to 8 (#161)")
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

    def run(prefix: str, chunk: int, key_chunk: int = 0) -> dict[str, np.ndarray]:
        run_tool([args.tool, args.model, str(il), str(in_path), prefix, str(tokens),
                  "--trace", "--kv", args.kv, "--window", str(args.window),
                  "--chunk", str(chunk), "--key-chunk", str(key_chunk)])
        tensors = {name: np.fromfile(f"{prefix}-{name}.f32", dtype=np.float32)
                   for name, _ in pairs}
        tensors["out"] = np.fromfile(f"{prefix}.out.f32", dtype=np.float32)
        return tensors

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.f32"
        prefix = str(Path(tmp) / "t")
        np.ascontiguousarray(x, dtype=np.float32).tofile(in_path)
        traced = run(prefix, args.chunk)
        got = {name: traced[name] for name, _ in pairs}
        out = traced["out"]

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
    ok = worst[0] <= args.tol
    if args.verify:
        # N tokens in one call against N decode steps, both with the runner's
        # fixed key chunks: the engine's own invariant up to 8 tokens (#161),
        # and the per-query path's window indices beyond it (#315).
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "in.f32"
            np.ascontiguousarray(x, dtype=np.float32).tofile(in_path)
            ref = run(str(Path(tmp) / "v1"), 1, -1)["attn_gated"]
            print(f"verify: N tokens in one call against {tokens} decode steps, "
                  f"key chunks as the runner gives them")
            for n in range(2, tokens + 1):
                mine = run(str(Path(tmp) / f"v{n}"), n, -1)["attn_gated"]
                exact = np.array_equal(mine, ref)
                rel = rel_diff(mine, ref)
                good = exact if n <= 8 else rel <= args.tol
                note = "bit-identical" if exact else f"rel={rel:.3e}"
                print(f"  {n:2} tokens: {note} (n <= 8 must be bit-identical) "
                      f"{'ok' if good else 'FAIL'}")
                ok = ok and good
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
