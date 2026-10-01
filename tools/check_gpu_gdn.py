#!/usr/bin/env python3
"""Compare omph-gdn (GPU gated delta net block, the engine's gdn_step) against the golden dump.

usage: uv run python check_gpu_gdn.py <model.gguf> <layer> [--chunk N]

gdn_step fuses the conv, the L2 norms, the gates, the delta rule and the gated
norm, so the check sees the projections, the gated-norm output and the block
output. --chunk 1 runs the decode path (one token per call, the conv state
carried across calls).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from omphalos_tools.golden import load, load_index, rel_diff, run_tool

PAIRS = (
    ("qkv", "linear_attn_qkv_mixed"),
    ("z", "z"),
    ("final", "final_output"),
)


def main() -> None:
    ap = argparse.ArgumentParser(description="check the GPU gated delta net block")
    ap.add_argument("model")
    ap.add_argument("layer", type=int)
    ap.add_argument("--dump", default="../models/golden/cpu")
    ap.add_argument("--tool", default="../engine/build/omph-gdn")
    ap.add_argument("--tol", type=float, default=0.15, help="relative")
    ap.add_argument("--chunk", type=int, default=0)
    args = ap.parse_args()

    dump = Path(args.dump)
    idx = load_index(dump)
    il = args.layer

    x = load(dump, idx[f"attn_norm-{il}"])
    tokens = int(np.prod(x.shape[:-1]))
    x = x.reshape(tokens, -1)

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.f32"
        prefix = str(Path(tmp) / "t")
        np.ascontiguousarray(x, dtype=np.float32).tofile(in_path)
        run_tool([args.tool, args.model, str(il), str(in_path), prefix, str(tokens),
                        "--trace", "--chunk", str(args.chunk)])
        got = {name: np.fromfile(f"{prefix}-{name}.f32", dtype=np.float32) for name, _ in PAIRS}
        out = np.fromfile(f"{prefix}.out.f32", dtype=np.float32)

    print(f"chunk {args.chunk or tokens}, {tokens} tokens")
    worst = (0.0, "")
    for name, dump_name in PAIRS:
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
    ref = load(dump, idx[f"linear_attn_out-{il}"])
    rel = rel_diff(out.reshape(ref.shape), ref)
    worst = max(worst, (rel, "out"), key=lambda t: t[0])
    print(f"  {'out':12} -> linear_attn_out-{il}: rel={rel:.3e}  shape={ref.shape}")
    print(f"worst: {worst[1]} rel={worst[0]:.3e} (tol {args.tol:g})")
    raise SystemExit(0 if worst[0] <= args.tol else 1)


if __name__ == "__main__":
    main()
