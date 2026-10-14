#!/usr/bin/env python3
"""#453: how much of the head a draft prefix needs, if its rows are ordered by
frequency instead of by id (BPE merge order).

Offline, no GPU: the head (output.weight, Q4_K) dequantized with gguf-py, the
probe vectors = the target's layer-63 hidden states from `omph-run --trace-dir`
RMS-normalized as the head's input is (output_norm), the true top-16 per position
over the whole head, and then for each prefix set:

  - id order: the first N ids, what OMPH_DRAFT_VOCAB = 98304 does today;
  - frequency order: the N ids most frequent in a workload, the permuted layout
    the issue proposes (a lossless re-layout: only the row order changes).

Reported per prefix: how many of the true top-16 are inside (recall) and whether
the prefix's argmax is the head's (top-1 agreement). The permutation must be
baked in for every workload, so the frequency of one text is evaluated on the
others too - that cross-use is the real question.

    uv run python freq_prefix_probe.py --states /tmp/tr-it/l_out-63.f32 \
        --freq /tmp/gen-it.tok --freq /tmp/gen-code.tok --sizes 16384,32768,65536,98304
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gguf import GGUFReader, quants

MODEL = str(Path(__file__).resolve().parent.parent / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", required=True, help="l_out-63.f32 from --trace-dir (the head's input)")
    ap.add_argument("--freq", action="append", required=True, help="a token-id file whose histogram orders the head")
    ap.add_argument("--sizes", default="16384,32768,65536,98304")
    ap.add_argument("--probes", type=int, default=512)
    ap.add_argument("--gen", action="append", default=[],
                    help="a generated token-id file (a --gen-out): the OOV a prefix would leave")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    r = GGUFReader(MODEL)
    head_t = next(t for t in r.tensors if t.name == "output.weight")
    norm_t = next(t for t in r.tensors if t.name == "output_norm.weight")
    head = quants.dequantize(head_t.data, head_t.tensor_type).astype(np.float32)
    head = head.reshape(-1, head_t.shape[0])
    n_vocab, ne = head.shape
    norm = quants.dequantize(norm_t.data, norm_t.tensor_type).astype(np.float32).reshape(-1)
    print(f"head {head.shape} ({head.nbytes / 2**30:.2f} GiB dequantized), norm {norm.shape}")

    x = np.fromfile(args.states, dtype=np.float32).reshape(-1, ne)[: args.probes]
    x = x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-6) * norm
    print(f"probes {x.shape}")

    logits = x @ head.T  # (probes, n_vocab)
    top16 = np.argpartition(-logits, 16, axis=-1)[:, :16]
    top1 = logits.argmax(axis=-1)

    freqs = []
    for f in args.freq:
        # the --gen-out files are text: space-separated ids
        ids = np.array([int(t) for t in Path(f).read_text().split()], dtype=np.int64)
        hist = np.bincount(ids, minlength=n_vocab).astype(np.float64)
        order = np.argsort(-hist, kind="stable")
        freqs.append((Path(f).name, order))
        print(f"frequency from {Path(f).name}: {len(ids)} tokens, "
              f"{int((hist > 0).sum())} distinct, top-5 ids {order[:5].tolist()}")

    # The other side of the question: the share of a *generation*'s ids that a
    # prefix still misses is what makes the head fall back to its whole 682 MiB
    # (draft_oov_ >= 5 %), so a permutation has to be judged on that too.
    gen_ids = np.array([int(t) for f in args.gen for t in Path(f).read_text().split()], dtype=np.int64)
    print(f"\nOOV on {len(gen_ids)} generated ids from {[Path(f).name for f in args.gen]}:")
    for n in sizes:
        for name, order in [("id order (0..N-1)", np.arange(n_vocab))] + freqs:
            inside = np.isin(gen_ids, order[:n]).mean()
            print(f"   {n:>8} {name:>34} oov {1 - inside:>7.4f}")

    print(f"\n{'prefix':>8} {'order':>34} {'recall@16':>10} {'top-1 agree':>12}")
    for n in sizes:
        for name, order in [("id order (0..N-1)", np.arange(n_vocab))] + freqs:
            inside = np.isin(top16, order[:n]).sum(axis=-1)
            agree = np.isin(top1, order[:n]).mean()
            print(f"{n:>8} {name:>34} {inside.mean():>10.4f} {agree:>12.4f}")


if __name__ == "__main__":
    main()
