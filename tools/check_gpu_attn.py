#!/usr/bin/env python3
"""Compare omph-attn (GPU attention block, the engine's kernels) against the golden dump.

usage: uv run python check_gpu_attn.py <model.gguf> <layer> [--kv f32|q8q4|q4q4]
                                       [--window N] [--chunk N] [--verify]

--kv f32 (default) checks every intermediate; with a quantized cache q and k are
Hadamard-rotated, so only the gate, the attention output and the block output
are compared. --window (512 since #318, the engine's default) sizes the FP16 ring, and 0 sends
every key through the quantized blocks;
--chunk 1 runs the decode path (one token per call, split-K attention).
--key-chunk -1 (default) makes both comparisons use the runner's fixed key chunks
(#136/#161) for a --ctx-position cache, as the engine does, instead of the
kernel's own split.
--verify runs N = 2..--verify-tokens tokens in one call against the decode path:
bit-identical up to 8 (the engine's own invariant, #161), and within --tol up to
16, where the kernel leaves the decode path's fixed chunks (#315). The rows past
the dump's prompt are built from its own (the reference is the decode path).
--identity-tokens N adds the long-sequence witness of the same invariant: with
N rows (the ring wraps, the fixed key chunks split the sequence) 8 tokens per
call must be the decode steps' rows bit for bit -- and it prints what the
kernel's own split gives instead, which is the evidence the check can tell the
two apart (#318).
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
    ap.add_argument("--window", type=int, default=512,
                    help="the FP16 ring's slots (512: the engine's default since #318; 0: "
                         "every key through the quantized blocks)")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--key-chunk", type=int, default=-1,
                    help="-1 (default): the runner's fixed key chunks, as the engine gives "
                         "them (#136/#161); 0: the kernel's own split")
    ap.add_argument("--ctx", type=int, default=8192,
                    help="the KV capacity --key-chunk -1 mirrors (8192, omph-run's default)")
    ap.add_argument("--verify", action="store_true",
                    help="N = 2..--verify-tokens tokens per call against the decode path: "
                         "bit-identical up to 8 (#161)")
    ap.add_argument("--verify-tokens", type=int, default=16,
                    help="how far the --verify sweep goes (16: the last token count the "
                         "per_query branch serves, #315); 0: only the dump's own prompt")
    ap.add_argument("--identity-tokens", type=int, default=2048,
                    help="rows for the long-sequence bit-identity witness (#161/#318): at "
                         "16 rows the fixed key chunks are a no-op; 0: off")
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

    def run(path: Path, prefix: str, chunk: int, rows: int,
            key_chunk: int | None = None) -> dict[str, np.ndarray]:
        run_tool([args.tool, args.model, str(il), str(path), prefix, str(rows),
                  "--trace", "--kv", args.kv, "--window", str(args.window),
                  "--chunk", str(chunk),
                  "--key-chunk", str(args.key_chunk if key_chunk is None else key_chunk),
                  "--ctx", str(args.ctx)])
        tensors = {name: np.fromfile(f"{prefix}-{name}.f32", dtype=np.float32)
                   for name, _ in pairs}
        tensors["out"] = np.fromfile(f"{prefix}.out.f32", dtype=np.float32)
        return tensors

    def synthetic(n: int) -> np.ndarray:
        """n rows from the dump's: the first `tokens` are its own, the rest are
        copies scaled per position, so no two positions carry the same K/V."""
        rows = np.empty((n, x.shape[1]), dtype=np.float32)
        rows[:tokens] = x
        for i in range(tokens, n):
            rows[i] = x[i % tokens] * (1.0 + 1e-3 * (i - tokens + 1))
        return rows

    def sweep(rows: np.ndarray) -> bool:
        """N tokens in one call against N decode steps: the engine's own invariant
        up to 8 tokens (#161), and the per-query window path's beyond it -- the
        branch a prompt's last chunk of 9..16 tokens takes (#315)."""
        n_max = len(rows)
        good = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "in.f32"
            np.ascontiguousarray(rows, dtype=np.float32).tofile(path)
            ref = run(path, str(Path(tmp) / "v1"), 1, n_max)["attn_gated"]
            print(f"verify: N tokens in one call against N decode steps ({n_max} rows, "
                  f"the first {min(n_max, tokens)} from the dump)")
            for n in range(2, n_max + 1):
                mine = run(path, str(Path(tmp) / f"v{n}"), n, n_max)["attn_gated"]
                exact = np.array_equal(mine, ref)
                rel = rel_diff(mine, ref)
                fine = exact if n <= 8 else rel <= args.tol
                note = "bit-identical" if exact else f"rel={rel:.3e}"
                print(f"  {n:2} tokens: {note} (n <= 8 must be bit-identical) "
                      f"{'ok' if fine else 'FAIL'}")
                good = good and fine
        return good

    def identity(n_tokens: int) -> bool:
        """The #161 invariant where it has teeth: on a long sequence the ring has
        wrapped and the fixed key chunks split it, so 8 tokens per call must be
        the decode steps' rows bit for bit. With the kernel's own split they are
        not (printed, not judged): that is what makes this a witness."""
        path_rows = synthetic(max(n_tokens, tokens))
        n = len(path_rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "in.f32"
            np.ascontiguousarray(path_rows, dtype=np.float32).tofile(path)
            ref = run(path, str(Path(tmp) / "i1"), 1, n)["attn_gated"]
            eight = run(path, str(Path(tmp) / "i8"), 8, n)["attn_gated"]
            exact = np.array_equal(ref, eight)
            print(f"identity: {n} tokens, 8 per call against 1, the runner's key chunks: "
                  f"{'bit-identical' if exact else 'DIFFERENT'} (must be bit-identical)")
            if args.key_chunk != 0:
                plain = run(path, str(Path(tmp) / "i8k"), 8, n, 0)["attn_gated"]
                print(f"  the same 8-per-call with the kernel's own split: "
                      f"rel={rel_diff(plain, ref):.3e} (the fixed chunks are what holds it)")
        return exact

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.f32"
        prefix = str(Path(tmp) / "t")
        np.ascontiguousarray(x, dtype=np.float32).tofile(in_path)
        traced = run(in_path, prefix, args.chunk, tokens)
        got = {name: traced[name] for name, _ in pairs}
        out = traced["out"]

    print(f"kv {args.kv}, window {args.window}, chunk {args.chunk or tokens}, {tokens} tokens, "
          f"key chunks {'the runner\'s' if args.key_chunk < 0 else args.key_chunk}")
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
        # up to 16: the last token count the per_query branch serves (#315); more
        # rows than the dump has are built from its own
        n_verify = min(max(args.verify_tokens, tokens), 16)
        ok = sweep(synthetic(n_verify) if n_verify > tokens else x) and ok
    if args.identity_tokens > 0:
        ok = identity(args.identity_tokens) and ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
