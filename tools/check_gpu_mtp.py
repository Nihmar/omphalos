#!/usr/bin/env python3
"""The engine's MTP draft head against the NumPy MTP block fed llama.cpp's own h.

usage: uv run python check_gpu_mtp.py [--model M.gguf] [--dump DIR] [--tol 0.15]

The golden dump holds llama.cpp's `h_nextn` rows (its output_norm hidden, the
MTP input) for the prompt and its final logits. The NumPy reference runs the
MTP block over positions 0..n with the pairs (h_{p-1}, t_p) (h = 0 at p = 0),
t_n the greedy token, then a second draft chained on its output g and on the
GPU's first draft token (so both chains continue from the same token). omph-run
--mtp-out does the same on the GPU from its own h. Each draft's logits must be
within the tolerance, and the GPU's greedy draft must be the reference's or
within --tie (nats) of it: the GPU reads its own h and a Q8/Q4 cache, so a
near-tie may flip (#124).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from omph_model import omph_file
from omphalos_tools.golden import load, load_index, rel_diff, run_tool
from omphalos_tools.model import Model
from omphalos_tools.reference import Reference


def main() -> None:
    ap = argparse.ArgumentParser(description="check the GPU MTP draft head")
    ap.add_argument("--model", default="../models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")
    ap.add_argument("--dump", default="../models/golden/cpu")
    ap.add_argument("--tool", default="../engine/build/omph-run")
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--tie", type=float, default=0.1)
    args = ap.parse_args()

    dump = Path(args.dump)
    idx = load_index(dump)
    tokens = [int(t) for t in (dump / "tokens.txt").read_text().split()]
    h = load(dump, idx["h_nextn"]).reshape(len(tokens), -1)
    t_next = int(load(dump, idx["result_output"]).reshape(-1).argmax())

    ref = Reference(Model(args.model))
    seq = tokens + [t_next]
    h_in = np.concatenate([np.zeros((1, h.shape[1]), np.float32), h], axis=0)
    logits, g = ref.mtp(seq, h_in, 0)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "mtp.f32"
        gen = Path(tmp) / "gen.txt"
        run_tool([args.tool, omph_file(args.model), str(dump / "tokens.txt"), str(Path(tmp) / "x.f32"),
                  "--last-logits", "--gemv", "--generate", "2", "--gen-out", str(gen),
                  "--mtp-out", str(out)])
        got = np.fromfile(out, dtype=np.float32).reshape(2, -1)

    d1 = int(got[0].argmax())  # chain the reference on the GPU's first draft
    logits2, _ = ref.mtp([d1], g[-1:], len(seq))
    want = np.stack([logits[-1], logits2[-1]])

    ok = True
    for i in range(2):
        rel = rel_diff(got[i], want[i])
        pick = int(got[i].argmax())
        gap = float(want[i].max() - want[i][pick])
        same = gap <= args.tie
        top_g = set(np.argsort(-got[i])[:5].tolist())
        top_w = set(np.argsort(-want[i])[:5].tolist())
        print(f"draft {i + 1}: argmax gpu {int(got[i].argmax())} ref {int(want[i].argmax())}  "
              f"(gap {gap:.3f})  rel={rel:.3e}  top-5 overlap {len(top_g & top_w)}/5")
        ok = ok and same and rel <= args.tol
    print(f"verdict: {'PASS' if ok else 'FAIL'} (tol {args.tol:g})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
