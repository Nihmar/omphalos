"""Compare two omph-run logits dumps: KL(ref || test) per position and top-1 agreement.

usage: uv run python compare_logits.py <ref.f32> <test.f32> [--vocab N] [--skip N]

Both files are (tokens, vocab) f32, as written by omph-run. `--skip` drops the
first N positions (the earliest ones see almost no context and dominate nothing,
but they can be excluded when comparing long-context behaviour).
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
    ap.add_argument("--vocab", type=int, default=N_VOCAB)
    ap.add_argument("--skip", type=int, default=0)
    args = ap.parse_args()

    ref = np.fromfile(args.ref, dtype=np.float32).reshape(-1, args.vocab)
    test = np.fromfile(args.test, dtype=np.float32).reshape(-1, args.vocab)
    if ref.shape != test.shape:
        print(f"shape mismatch: {ref.shape} vs {test.shape}", file=sys.stderr)
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
