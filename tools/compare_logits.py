"""Compare two omph-run logits dumps: KL(ref || test) per position and top-1 agreement.

usage: uv run python compare_logits.py <ref.f32> <test.f32> [--vocab N | --model M.gguf]
                                       [--skip N]

Both files are (tokens, vocab) f32, as written by omph-run. The vocabulary size
comes from --vocab, or from the model's token_embd with --model, or defaults
to this model's 248320. `--skip` drops the first N positions (the earliest ones
see almost no context and dominate nothing, but they can be excluded when
comparing long-context behaviour).
"""

import argparse
import sys

import numpy as np

N_VOCAB = 248320


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--model", default=None, help="take the vocabulary size from this GGUF")
    ap.add_argument("--skip", type=int, default=0)
    args = ap.parse_args()

    vocab = args.vocab
    if vocab is None and args.model is not None:
        from gguf import GGUFReader

        te = next((t for t in GGUFReader(args.model).tensors if t.name == "token_embd.weight"),
                  None)
        if te is None:
            print(f"{args.model}: no token_embd.weight", file=sys.stderr)
            return 1
        vocab = int(te.shape[-1])
    vocab = vocab or N_VOCAB
    arrays = []
    for path in (args.ref, args.test):
        flat = np.fromfile(path, dtype=np.float32)
        if flat.size == 0 or flat.size % vocab != 0:
            print(f"{path}: {flat.size} floats is not a whole number of {vocab}-wide rows "
                  f"(wrong --vocab?)", file=sys.stderr)
            return 1
        arrays.append(flat.reshape(-1, vocab))
    ref, test = arrays
    if ref.shape != test.shape:
        print(f"shape mismatch: {ref.shape} vs {test.shape}", file=sys.stderr)
        return 1
    if not 0 <= args.skip < len(ref):
        print(f"--skip {args.skip} leaves no positions of {len(ref)}", file=sys.stderr)
        return 1
    ref = ref[args.skip :]
    test = test[args.skip :]

    kl = np.empty(len(ref))
    for i in range(len(ref)):  # row by row: a (512, 248320) f64 block is 1 GB
        lp = log_softmax(ref[i])
        lq = log_softmax(test[i])
        kl[i] = float((np.exp(lp) * (lp - lq)).sum())
    top1 = float((ref.argmax(axis=-1) == test.argmax(axis=-1)).mean())

    print(f"positions {len(ref)}")
    print(f"KL mean {kl.mean():.6f}  median {np.median(kl):.6f}  "
          f"p99 {np.percentile(kl, 99):.6f}  max {kl.max():.6f} nats")
    print(f"top-1 agreement {top1 * 100:.2f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
