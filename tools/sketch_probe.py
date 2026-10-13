"""#418 probe: can a rank-r sketch of the head keep the top-16 candidates?

Offline, no GPU: the head's first 98304 rows (Q4_K) dequantized with gguf-py, the
probe vectors = the target's layer-63 hidden states (real, from --trace-dir)
RMS-normalized as the head's input is, then the true top-16 against the sketch's
(a fixed-seed Gaussian projection), for r = 128 / 256 / 512.
    uv run python /tmp/sketch_probe.py
"""
from pathlib import Path

import numpy as np
from gguf import GGUFReader, quants

MODEL = str(Path(__file__).resolve().parent.parent / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")
ROWS = 98304
PROBES = 256
RANKS = (128, 256, 512)

r = GGUFReader(MODEL)
head_t = next(t for t in r.tensors if t.name == "output.weight")
print("head:", head_t.name, head_t.tensor_type, head_t.shape, head_t.data.dtype)
h = quants.dequantize(head_t.data, head_t.tensor_type).astype(np.float32)
h = h.reshape(-1, head_t.shape[0])[:ROWS]
print("head dequantized:", h.shape, "GB", h.nbytes / 2**30)

norm_t = next(t for t in r.tensors if t.name == "output_norm.weight")
nw = norm_t.data.astype(np.float32).reshape(-1)

x = np.fromfile("/tmp/tr/l_out-63.f32", dtype=np.float32).reshape(-1, h.shape[1])
x = x[:PROBES]
rms = np.sqrt((x * x).mean(axis=1, keepdims=True) + 1e-6)
x = (x / rms) * nw
print("probes:", x.shape)

true = x @ h.T                      # (PROBES, ROWS)
k = 16
true_top = np.argsort(-true, axis=1)[:, :k]
print("true top-1 (as ids):", true_top[:5, 0])

rng = np.random.default_rng(418)
for rank in RANKS:
    p = (rng.standard_normal((h.shape[1], rank)) / np.sqrt(rank)).astype(np.float32)
    # the Johnson-Lindenstrauss sketch: project both sides with the same P
    sk = h @ p                       # (ROWS, rank): the projected vocabulary
    xq = x @ p                       # (PROBES, rank): the projected queries
    approx = xq @ sk.T               # (PROBES, ROWS)
    ap_top = np.argsort(-approx, axis=1)[:, :k]
    hit = np.array([len(set(a) & set(t)) for a, t in zip(ap_top, true_top)])
    top1 = (ap_top[:, 0] == true_top[:, 0]).mean()
    print(f"rank {rank:4d}: recall@16 {hit.mean()/k*100:5.1f}%   top-1 {top1*100:5.1f}%   "
          f"sketch bytes {ROWS*rank*2/2**20:6.1f} MiB (f16)")
