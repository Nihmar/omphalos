"""#418 follow-up: a two-stage draft head. Does a rank-r sketch's top-M shortlist
contain the head's true top-16 (so the exact re-rank inside it is exact)?

Offline, no GPU: the full head (Q4_K), 128 real hidden states (--trace-dir),
RMS-normalized. For each probe: the sketch's top-M shortlist, the exact logits
re-ranked inside it, compared with the exact top-16 over the whole vocabulary -
and the same for the *prefix* head the engine uses today (first 98304 rows).
    uv run python /tmp/shortlist_probe.py
"""
from pathlib import Path

import numpy as np
from gguf import GGUFReader, quants

ROOT = Path("/home/alessandro/Projects/omphalos")
MODEL = str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")
PREFIX = 98304
PROBES = 128
RANKS = (256, 512)
SHORTLIST = 4096

r = GGUFReader(MODEL)
head_t = next(t for t in r.tensors if t.name == "output.weight")
h = quants.dequantize(head_t.data, head_t.tensor_type).astype(np.float32).reshape(-1, head_t.shape[0])
print(f"head {h.shape} ({h.nbytes / 2**30:.2f} GiB)", flush=True)
norm_t = next(t for t in r.tensors if t.name == "output_norm.weight")
nw = norm_t.data.astype(np.float32).reshape(-1)
x = np.fromfile("/tmp/tr/l_out-63.f32", dtype=np.float32).reshape(-1, h.shape[1])[:PROBES]
rms = np.sqrt((x * x).mean(axis=1, keepdims=True) + 1e-6)
x = (x / rms) * nw
print(f"probes {x.shape}", flush=True)

true = x @ h.T                                     # (PROBES, vocab)
true_top = np.argsort(-true, axis=1)[:, :16]
k = 16

# the engine's today: the prefix head only
pref_scores = true[:, :PREFIX]
pref_top = np.argsort(-pref_scores, axis=1)[:, :k]
pref_hit = np.array([len(set(a) & set(t)) for a, t in zip(pref_top, true_top)])
print(f"prefix head (98304 rows), recall@16 vs the true top-16: {pref_hit.mean()/k*100:5.1f}%  "
      f"contains the whole top-16: {(pref_hit==k).mean()*100:5.1f}%", flush=True)

rng = np.random.default_rng(418)
for rank in RANKS:
    p = (rng.standard_normal((h.shape[1], rank)) / np.sqrt(rank)).astype(np.float32)
    sk = h @ p                                     # (vocab, rank)
    print(f"rank {rank}: sketch {h.shape[0]*rank*2/2**20:.0f} MiB f16", flush=True)
    hits = []
    full_hits = 0
    for i in range(x.shape[0]):
        xq = x[i] @ p                              # (rank,)
        s = xq @ sk.T                              # (vocab,)
        short = np.argpartition(-s, SHORTLIST)[:SHORTLIST]
        exact = x[i] @ h[short].T                   # re-rank inside the shortlist
        top = short[np.argsort(-exact)[:k]]
        hh = len(set(top.tolist()) & set(true_top[i].tolist()))
        hits.append(hh)
        full_hits += hh == k
    hits = np.array(hits)
    print(f"  two-stage: recall@16 {hits.mean()/k*100:5.1f}%   contains the whole top-16 "
          f"{full_hits/x.shape[0]*100:5.1f}%", flush=True)
