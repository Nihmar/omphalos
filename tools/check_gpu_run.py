#!/usr/bin/env python3
"""Compare omph-run (full 64-layer forward pass) against the golden dump.

usage: uv run python check_gpu_run.py <model.gguf> [--dump DIR] [--layers] [--tol 0.15]
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from omph_model import omph_file
from omphalos_tools.golden import load, load_index, rel_diff


def main() -> None:
    ap = argparse.ArgumentParser(description="check the full naive GPU forward pass")
    ap.add_argument("model")
    ap.add_argument("--dump", default="../models/golden/cpu")
    ap.add_argument("--tool", default="../engine/build/omph-run")
    ap.add_argument("--layers", action="store_true", help="also compare the per-layer outputs")
    ap.add_argument("--tol", type=float, default=0.15)
    args = ap.parse_args()

    dump = Path(args.dump)
    idx = load_index(dump)
    tokens = [int(t) for t in (dump / "tokens.txt").read_text().split()]
    print(f"prompt: {tokens}")

    with tempfile.TemporaryDirectory() as tmp:
        tok_path = Path(tmp) / "tokens.txt"
        tok_path.write_text(" ".join(str(t) for t in tokens))
        logits_path = Path(tmp) / "logits.f32"
        cmd = [args.tool, omph_file(args.model), str(tok_path), str(logits_path)]
        if args.layers:
            trace = Path(tmp) / "trace"
            trace.mkdir()
            cmd += ["--trace-dir", str(trace)]
        subprocess.run(cmd, check=True)
        got = np.fromfile(logits_path, dtype=np.float32)

        worst = (0.0, "")
        if args.layers:
            for il in range(64):
                entry = idx.get(f"l_out-{il}")
                if entry is None:
                    continue
                ref = load(dump, entry)
                mine = np.fromfile(trace / f"l_out-{il}.f32", dtype=np.float32).reshape(ref.shape)
                rel = rel_diff(mine, ref)
                worst = max(worst, (rel, f"l_out-{il}"), key=lambda t: t[0])
                if il % 8 == 0 or rel > args.tol:
                    print(f"  l_out-{il:<3}: rel={rel:.3e}")
            print(f"  worst layer: {worst[1]} rel={worst[0]:.3e}")

    ref = load(dump, idx["result_output"])
    ref_last = ref.reshape(-1, ref.shape[-1])[-1]          # the dump keeps the last token only
    mine_last = got.reshape(-1, ref.shape[-1])[-1]
    print("\nlogits (last token):")
    print(f"  ref argmax : {int(ref_last.argmax())}")
    print(f"  gpu argmax : {int(mine_last.argmax())}")
    rel = float(np.abs(mine_last - ref_last).max()) / max(float(np.abs(ref_last).max()), 1e-12)
    print(f"  rel={rel:.3e}")
    # rank agreement on the last token
    top_ref = set(np.argsort(-ref_last)[:5].tolist())
    top_mine = set(np.argsort(-mine_last)[:5].tolist())
    print(f"  top-5 overlap: {len(top_ref & top_mine)}/5")
    ok = worst[0] <= args.tol and rel <= args.tol and int(ref_last.argmax()) == int(
        mine_last.argmax())
    print(f"verdict: {'PASS' if ok else 'FAIL'} (tol {args.tol:g})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
