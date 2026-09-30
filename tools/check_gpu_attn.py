#!/usr/bin/env python3
"""Compare omph-attn (GPU attention block) against the golden dump.

usage: uv run python check_gpu_attn.py <model.gguf> <layer> [--tokens-from-dump]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PAIRS = (
    ("q_full", "Qcur_full"),
    ("q_normed", "Qcur_normed"),
    ("k_normed", "Kcur_normed"),
    ("q", "Qcur"),
    ("k", "Kcur"),
    ("gate", "gate_reshaped"),
    ("attn_gated", "attn_gated"),
)


def load_index(dump: Path) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    for line in (dump / "index.jsonl").read_text().splitlines():
        entry = json.loads(line)
        idx[entry["name"]] = entry
    return idx


def load(dump: Path, entry: dict) -> np.ndarray:
    dtype = {"f32": np.float32, "f16": np.float16}[entry["type"]]
    flat = np.fromfile(dump / entry["file"], dtype=dtype).astype(np.float32)
    return flat.reshape(tuple(reversed(entry["ne"])))


def rel_diff(mine: np.ndarray, ref: np.ndarray) -> float:
    d = np.abs(mine - ref)
    scale = max(float(np.abs(ref).max()), 1e-12)
    return float(d.max()) / scale


def main() -> None:
    ap = argparse.ArgumentParser(description="check the GPU attention block")
    ap.add_argument("model")
    ap.add_argument("layer", type=int)
    ap.add_argument("--dump", default="../models/golden/cpu")
    ap.add_argument("--tool", default="../engine/build/omph-attn")
    ap.add_argument("--tol", type=float, default=8e-2, help="relative (llama.cpp's own path is at ~2e-2)")
    args = ap.parse_args()

    dump = Path(args.dump)
    idx = load_index(dump)
    il = args.layer

    x_entry = idx[f"attn_norm-{il}"]  # layer input (post attn_norm) -> we feed the same
    x = load(dump, x_entry)
    tokens = int(np.prod(x.shape[:-1]))
    x = x.reshape(tokens, -1)

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.f32"
        prefix = str(Path(tmp) / "t")
        np.ascontiguousarray(x, dtype=np.float32).tofile(in_path)
        subprocess.run([args.tool, args.model, str(il), str(in_path), prefix, str(tokens),
                        "--trace"], check=True, capture_output=True)
        got = {}
        for name, _ in PAIRS:
            got[name] = np.fromfile(f"{prefix}-{name}.f32", dtype=np.float32)
        out = np.fromfile(f"{prefix}.out.f32", dtype=np.float32)

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
    ref = load(dump, idx[f"attn_output-{il}"])
    rel = rel_diff(out.reshape(ref.shape), ref)
    worst = max(worst, (rel, "out"), key=lambda t: t[0])
    print(f"  {'out':12} -> attn_output-{il}: rel={rel:.3e}  shape={ref.shape}")
    print(f"worst: {worst[1]} rel={worst[0]:.3e} (tol {args.tol:g})")
    raise SystemExit(0 if worst[0] <= args.tol else 1)


if __name__ == "__main__":
    main()
