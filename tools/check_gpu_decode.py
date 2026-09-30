#!/usr/bin/env python3
"""Greedy decode: engine vs NumPy reference (milestone 2 exit criterion).

usage: uv run python check_gpu_decode.py [--generate N] [--tol 0.15]
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from omphalos_tools.model import Model
from omphalos_tools.reference import Reference


def rel_diff(mine: np.ndarray, ref: np.ndarray) -> float:
    scale = max(float(np.abs(ref).max()), 1e-12)
    return float(np.abs(mine - ref).max()) / scale


def main() -> None:
    ap = argparse.ArgumentParser(description="compare greedy decode with the reference")
    ap.add_argument("--model", default="../models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")
    ap.add_argument("--dump", default="../models/golden/cpu")
    ap.add_argument("--tool", default="../engine/build/omph-run")
    ap.add_argument("--generate", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.15)
    args = ap.parse_args()

    tokens = [int(v) for v in (Path(args.dump) / "tokens.txt").read_text().split()]
    print(f"prompt ({len(tokens)}): {tokens}", flush=True)

    # ---- reference (NumPy, incremental)
    print("reference: loading model …", flush=True)
    ref = Reference(Model(args.model))
    t0 = time.time()
    logits = ref.forward(tokens)
    prefill_logits = logits
    print(f"reference: prefill {time.time() - t0:.1f}s", flush=True)
    gen_ref: list[int] = []
    nxt = int(logits[-1].argmax())
    for i in range(args.generate):
        gen_ref.append(nxt)
        if i + 1 == args.generate:
            break
        t0 = time.time()
        logits = ref.forward([nxt], start_pos=len(tokens) + i)
        nxt = int(logits[-1].argmax())
        print(f"reference: step {i + 1} {time.time() - t0:.1f}s -> {nxt}", flush=True)
    print(f"reference: generated {gen_ref}", flush=True)

    # ---- engine
    with tempfile.TemporaryDirectory() as tmp:
        tok_path = Path(tmp) / "tokens.txt"
        tok_path.write_text(" ".join(str(t) for t in tokens))
        gen_path = Path(tmp) / "gen.txt"
        t0 = time.time()
        out = subprocess.run(
            [args.tool, args.model, str(tok_path), str(Path(tmp) / "logits.f32"),
             "--generate", str(args.generate), "--gen-out", str(gen_path)],
            check=True, capture_output=True, text=True)
        print(f"engine: {time.time() - t0:.1f}s", flush=True)
        print(out.stdout.strip(), flush=True)
        gen_gpu = [int(v) for v in gen_path.read_text().split()]
        got = np.fromfile(Path(tmp) / "logits.f32", dtype=np.float32)

    rel = rel_diff(got.reshape(-1, prefill_logits.shape[-1])[-1], prefill_logits[-1])
    print(f"prefill logits (last token): rel={rel:.3e}  (tol {args.tol:g})")
    print(f"engine   : generated {gen_gpu}")
    same = gen_ref == gen_gpu
    print(f"greedy sequences identical: {same}")
    ok = same and rel <= args.tol
    print(f"verdict: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
