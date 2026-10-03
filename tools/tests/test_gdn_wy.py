"""The chunked WY form of the delta rule (engine/src/kernels/gdn.hip, #240) against
the per-token recurrence of the NumPy reference, in float64.

The engine's prefill computes, per 64-token chunk with h the state at its start,
g the cumulative log decay and Gam_ts = exp(g_t - g_s):
    A = strict_lower(beta_t (k_t . k_s) Gam_ts), T = (I + A)^-1
    D = T beta (V - gamma K h)
    o_t = gamma_t q_t h + sum_{s<=t} (q_t . k_s) Gam_ts D_s
    h' = gamma_C h + (K gamma_C / gamma)^T D
This checks that algebra, including a partial last chunk padded with beta = 0
and no decay, as the kernels pad it.
"""

from __future__ import annotations

import numpy as np

C = 64


def sequential(q, k, v, beta, gate, s):
    o = np.empty_like(v)
    for t in range(len(q)):
        s = s * np.exp(gate[t])
        d = (v[t] - s.T @ k[t]) * beta[t]
        s = s + np.outer(k[t], d)
        o[t] = s.T @ q[t]
    return o, s


def chunked(q, k, v, beta, gate, h):
    n_tok = len(q)
    o = np.empty_like(v)
    for c0 in range(0, n_tok, C):
        n = min(C, n_tok - c0)
        pad = ((0, C - n), (0, 0))
        qc, kc, vc = (np.pad(x[c0:c0 + n], pad) for x in (q, k, v))
        b = np.pad(beta[c0:c0 + n], (0, C - n))          # padded rows: beta 0 ...
        g = np.cumsum(np.pad(gate[c0:c0 + n], (0, C - n)))  # ... and no decay
        gam = np.exp(g[:, None] - g[None, :])
        a = np.tril(b[:, None] * (kc @ kc.T) * gam, -1)
        t_inv = np.linalg.inv(np.eye(C) + a)
        d = t_inv @ (b[:, None] * (vc - np.exp(g)[:, None] * (kc @ h)))
        p = np.tril((qc @ kc.T) * gam)
        o[c0:c0 + n] = (np.exp(g)[:, None] * (qc @ h) + p @ d)[:n]
        h = np.exp(g[-1]) * h + (kc * np.exp(g[-1] - g)[:, None]).T @ d
    return o, h


def test_wy_matches_the_recurrence():
    rng = np.random.default_rng(240)
    n_tok, dim = 200, 128  # three full chunks and a partial one
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    q = unit(rng.standard_normal((n_tok, dim))) / np.sqrt(dim)
    k = unit(rng.standard_normal((n_tok, dim)))
    v = rng.standard_normal((n_tok, dim)) * 2
    beta = 1 / (1 + np.exp(-rng.standard_normal(n_tok)))
    gate = -np.abs(rng.standard_normal(n_tok)) * 0.5
    s0 = rng.standard_normal((dim, dim)) * 0.1
    o1, s1 = sequential(q, k, v, beta, gate, s0)
    o2, s2 = chunked(q, k, v, beta, gate, s0)
    assert np.linalg.norm(o1 - o2) / np.linalg.norm(o1) < 1e-12
    assert np.linalg.norm(s1 - s2) / np.linalg.norm(s1) < 1e-12
