#!/usr/bin/env python3
"""Layer-by-layer validation of the NumPy reference against the golden dump.

usage: uv run python validate_reference.py [--max-layers N] [--tol X]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from omphalos_tools.golden import load, load_index, rel_diff
from omphalos_tools.model import Model
from omphalos_tools.reference import Captures, Reference

COMPARE = ("attn_norm", "attn_residual", "attn_post_norm", "ffn_out", "l_out")


def main() -> None:
    ap = argparse.ArgumentParser(description="validate reference vs golden dump")
    ap.add_argument("--model", default="../models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")
    ap.add_argument("--dump", default="../models/golden/cpu")
    ap.add_argument("--max-layers", type=int, default=None)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    dump = Path(args.dump)
    idx = load_index(dump)
    tokens = [int(v) for v in (dump / "tokens.txt").read_text().split()]
    model = Model(args.model)
    ref = Reference(model)
    cap = Captures()

    print(f"tokens ({len(tokens)}): {tokens}")
    logits = ref.forward(tokens, cap=cap, max_layers=args.max_layers)

    for name in ("model.input_embed",):
        if name in idx and name in cap.tensors:
            rel = rel_diff(cap.tensors[name], load(dump, idx[name]))
            print(f"{name}: rel={rel:.3e}")

    n = args.max_layers or model.hp.n_layer
    worst = (0.0, "")
    for il in range(n):
        parts = []
        for kind in COMPARE:
            name = f"{kind}-{il}"
            if name not in idx or name not in cap.tensors:
                parts.append(f"{kind}: -")
                continue
            refv = load(dump, idx[name])
            mine = cap.tensors[name]
            if mine.shape != refv.shape and mine.size == refv.size:
                mine = mine.reshape(refv.shape)
            rel = rel_diff(mine, refv)
            if rel > worst[0]:
                worst = (rel, name)
            flag = "" if rel <= args.tol else "  <<<"
            parts.append(f"{kind}: {rel:8.2e}{flag}")
        if not args.quiet or any("<<<" in p for p in parts):
            print(f"layer {il:2d}  " + "  ".join(parts))

    print(f"worst: {worst[1]} rel={worst[0]:.3e} (tol {args.tol:g})")

    if args.max_layers is None and "result_output" in idx:
        ref_logits = load(dump, idx["result_output"])
        mine = logits[-1].astype(np.float64)
        refv = ref_logits[-1].astype(np.float64)
        top1_mine, top1_ref = int(np.argmax(mine)), int(np.argmax(refv))
        p = np.exp(refv - refv.max())
        p /= p.sum()
        q = np.exp(mine - mine.max())
        q /= q.sum()
        kl = float(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30))))
        print(f"logits(last): top1 mine={top1_mine} ref={top1_ref} "
              f"match={top1_mine == top1_ref}  KL(ref||mine)={kl:.3e}  "
              f"max|d|={float(np.abs(mine - refv).max()):.3e}")


if __name__ == "__main__":
    main()
