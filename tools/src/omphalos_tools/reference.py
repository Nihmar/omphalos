"""NumPy reference forward pass for Qwen3.8-27B (arch `qwen35`) — PLAN.md §9.

Faithful port of llama.cpp's `src/models/qwen35.cpp` + `delta-net-base.cpp`
(gated delta net autoregressive path) and the standard attention/FFN blocks.

One token at a time, with a KV cache for the 16 full-attention layers and
conv/SSM state for the 48 gated delta net layers. Weights are dequantized on
demand per layer and reused across the tokens of a single call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Model

F32 = np.float32


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def rms_norm(x: np.ndarray, w: np.ndarray | None, eps: float) -> np.ndarray:
    ms = np.mean(x * x, axis=-1, keepdims=True)
    y = x / np.sqrt(ms + F32(eps))
    return y * w if w is not None else y


def l2_norm_gdn(x: np.ndarray, eps: float) -> np.ndarray:
    """models.h: rms_norm(x, eps/n) * 1/sqrt(n) — n = last dim."""
    n = x.shape[-1]
    return rms_norm(x, None, eps / n) * F32(1.0 / np.sqrt(n))


@dataclass
class Captures:
    """Optional per-layer activations, in ggml layout (n_tokens, n_embd)."""
    tensors: dict[str, np.ndarray] = field(default_factory=dict)

    def put(self, name: str, value: np.ndarray) -> None:
        self.tensors[name] = np.asarray(value)


class Reference:
    def __init__(self, model: Model):
        self.m = model
        self.hp = model.hp
        self.kv_k: list[np.ndarray | None] = [None] * self.hp.n_layer
        self.kv_v: list[np.ndarray | None] = [None] * self.hp.n_layer
        self.conv: list[np.ndarray | None] = [None] * self.hp.n_layer
        self.ssm: list[np.ndarray | None] = [None] * self.hp.n_layer

    # ------------------------------------------------------------------ rope

    def rope_cache(self) -> np.ndarray:
        """(n_rot_pairs,) frequencies, as in ggml_mrope_cache_init (text-only)."""
        hp = self.hp
        i = np.arange(hp.n_rot_pairs, dtype=F32)
        return (hp.rope_freq_base ** (-2.0 * i / hp.n_rot)).astype(F32)

    def apply_rope(self, x: np.ndarray, pos: np.ndarray, freq: np.ndarray) -> np.ndarray:
        """x: (T, H, head_dim); NeoX-style rotation on the first n_rot dims."""
        n_rot = self.hp.n_rot
        half = n_rot // 2
        theta = pos[:, None].astype(F32) * freq[None, :]           # (T, half)
        cos = np.cos(theta)[:, None, :]                            # (T, 1, half)
        sin = np.sin(theta)[:, None, :]
        x = x.copy()
        x0 = x[..., :half].copy()
        x1 = x[..., half:n_rot].copy()
        x[..., :half] = x0 * cos - x1 * sin
        x[..., half:n_rot] = x0 * sin + x1 * cos
        return x

    # ------------------------------------------------------------- deltanet

    def layer_deltanet(self, il: int, x: np.ndarray, cap: Captures | None) -> np.ndarray:
        hp = self.hp
        m = self.m
        p = f"blk.{il}."

        T = x.shape[0]
        n_kh, n_vh = hp.ssm_group, hp.ssm_dt_rank          # 16, 48
        s_k = hp.ssm_state                                  # 128
        s_v = hp.ssm_inner // n_vh                          # 128
        conv_c = 2 * n_kh * s_k + n_vh * s_v                # 10240

        Wqkv = m.matrix(p + "attn_qkv.weight")             # (10240, 5120)
        Wz = m.matrix(p + "attn_gate.weight")              # (6144, 5120)
        Wbeta = m.matrix(p + "ssm_beta.weight")            # (48, 5120)
        Walpha = m.matrix(p + "ssm_alpha.weight")          # (48, 5120)
        dt_bias = m.vector(p + "ssm_dt.bias")              # (48,)
        ssm_a = m.vector(p + "ssm_a")                      # (48,)
        conv_w = m.matrix(p + "ssm_conv1d.weight")         # (10240, 4)
        Wout = m.matrix(p + "ssm_out.weight")              # (5120, 6144)
        norm_w = m.vector(p + "ssm_norm.weight")           # (128,)

        qkv = x @ Wqkv.T                                   # (T, 10240)
        z = x @ Wz.T                                       # (T, 6144)
        beta = sigmoid(x @ Wbeta.T)                        # (T, 48)
        alpha_raw = x @ Walpha.T                           # (T, 48)
        a_softplus = softplus(alpha_raw + dt_bias)         # (T, 48)
        gate = a_softplus * ssm_a                          # (T, 48)
        if cap is not None:
            cap.put(f"linear_attn_qkv_mixed-{il}", qkv)
            cap.put(f"z-{il}", z)
            cap.put(f"beta_sigmoid-{il}", beta)
            cap.put(f"alpha-{il}", alpha_raw)
            cap.put(f"a_softplus-{il}", a_softplus)
            cap.put(f"gate-{il}", gate)

        # depthwise causal conv1d over the fused qkv channels
        cache = self.conv[il]
        if cache is None:
            cache = np.zeros((hp.ssm_conv_kernel - 1, conv_c), dtype=F32)
        inp = np.concatenate([cache, qkv], axis=0)         # (T + k - 1, C)
        out = np.zeros((T, conv_c), dtype=F32)
        for tap in range(hp.ssm_conv_kernel):
            out += inp[tap:tap + T] * conv_w[:, tap]
        self.conv[il] = inp[-(hp.ssm_conv_kernel - 1):]
        conv_silu = silu(out)
        if cap is not None:
            cap.put(f"conv_output_raw-{il}", out)
            cap.put(f"conv_output_silu-{il}", conv_silu)

        q = conv_silu[:, 0:n_kh * s_k].reshape(T, n_kh, s_k)
        k = conv_silu[:, n_kh * s_k:2 * n_kh * s_k].reshape(T, n_kh, s_k)
        v = conv_silu[:, 2 * n_kh * s_k:].reshape(T, n_vh, s_v)
        if cap is not None:
            cap.put(f"q_conv-{il}", q)
            cap.put(f"k_conv-{il}", k)
            cap.put(f"v_conv-{il}", v)

        q = l2_norm_gdn(q, hp.eps)
        k = l2_norm_gdn(k, hp.eps)
        if cap is not None:
            cap.put(f"q_conv_normed-{il}", q)
            cap.put(f"k_conv_normed-{il}", k)

        rep = n_vh // n_kh
        # ggml_repeat tiles along the head axis: v-head h uses k-head h % n_kh
        q = np.tile(q, (1, rep, 1))                        # (T, 48, 128)
        k = np.tile(k, (1, rep, 1))
        q = q * F32(1.0 / np.sqrt(s_k))
        if cap is not None:
            cap.put(f"q_conv_predelta-{il}", q)
            cap.put(f"k_conv_predelta-{il}", k)
            cap.put(f"v_conv_predelta-{il}", v)

        S = self.ssm[il]
        if S is None:
            S = np.zeros((n_vh, s_v, s_k), dtype=F32)      # [H, S_v, S_k]

        outs = np.empty((T, n_vh, s_v), dtype=F32)
        for t in range(T):
            g = np.exp(gate[t])[:, None, None]             # (H, 1, 1)
            b = beta[t][:, None]                           # (H, 1)
            S = S * g
            sk = np.einsum("hij,hi->hj", S, k[t])          # S^T k
            d = (v[t] - sk) * b
            S = S + k[t][:, :, None] * d[:, None, :]
            outs[t] = np.einsum("hij,hi->hj", S, q[t])
        self.ssm[il] = S
        if cap is not None:
            cap.put(f"attn_output-{il}", outs)
            cap.put(f"new_state-{il}", S)

        o_flat = outs.reshape(T, n_vh * s_v)               # ggml: head-major, dim fastest
        z_flat = z.reshape(T, n_vh, s_v)
        normed = rms_norm(o_flat.reshape(T * n_vh, s_v), norm_w, hp.eps).reshape(T, n_vh * s_v)
        gated = normed * silu(z_flat.reshape(T, n_vh * s_v))
        if cap is not None:
            cap.put(f"final_output-{il}", gated)

        out = gated @ Wout.T                               # (T, 5120)
        if cap is not None:
            cap.put(f"linear_attn_out-{il}", out)
        return out

    # ------------------------------------------------------------ attention

    def layer_attention(self, il: int, x: np.ndarray, pos: np.ndarray, cap: Captures | None) -> np.ndarray:
        hp = self.hp
        m = self.m
        p = f"blk.{il}."
        T = x.shape[0]
        hd, nh, nkv = hp.head_dim, hp.n_head, hp.n_head_kv
        rep = nh // nkv

        Wq = m.matrix(p + "attn_q.weight")                 # (nh*2*hd, n_embd)
        Wk = m.matrix(p + "attn_k.weight")                 # (nkv*hd, n_embd)
        Wv = m.matrix(p + "attn_v.weight")
        Wo = m.matrix(p + "attn_output.weight")            # (n_embd, nh*hd)
        qn = m.vector(p + "attn_q_norm.weight")            # (hd,)
        kn = m.vector(p + "attn_k_norm.weight")

        qfull = (x @ Wq.T).reshape(T, nh, 2 * hd)
        q = qfull[:, :, :hd]
        gate = qfull[:, :, hd:]
        q = rms_norm(q, qn, hp.eps)
        k = rms_norm((x @ Wk.T).reshape(T, nkv, hd), kn, hp.eps)
        v = (x @ Wv.T).reshape(T, nkv, hd)

        freq = self.rope_cache()
        q = self.apply_rope(q, pos, freq)
        k = self.apply_rope(k, pos, freq)

        k_cache = self.kv_k[il]
        v_cache = self.kv_v[il]
        if k_cache is None:
            k_cache = np.zeros((0, nkv, hd), dtype=F32)
            v_cache = np.zeros((0, nkv, hd), dtype=F32)
        k_all = np.concatenate([k_cache, k], axis=0)        # (S, nkv, hd)
        v_all = np.concatenate([v_cache, v], axis=0)
        self.kv_k[il], self.kv_v[il] = k_all, v_all

        S = k_all.shape[0]
        kx = np.repeat(k_all, rep, axis=1)                  # (S, nh, hd)
        vx = np.repeat(v_all, rep, axis=1)
        scale = F32(1.0 / np.sqrt(hd))
        scores = np.einsum("thd,shd->hts", q, kx) * scale   # (nh, T, S)
        # causal mask: query position (S - T + t) attends to s <= S - T + t
        qpos = np.arange(S - T, S)
        mask = np.arange(S)[None, :] > qpos[:, None]        # (T, S)
        scores = np.where(mask[None, :, :], F32(-np.inf), scores)
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        attn = np.einsum("hts,shd->thd", probs, vx)         # (T, nh, hd)

        attn = attn * sigmoid(gate)
        return (attn.reshape(T, nh * hd)) @ Wo.T            # (T, 5120)

    # ------------------------------------------------------------------ ffn

    def layer_ffn(self, il: int, x: np.ndarray) -> np.ndarray:
        p = f"blk.{il}."
        Wg = self.m.matrix(p + "ffn_gate.weight")
        Wu = self.m.matrix(p + "ffn_up.weight")
        Wd = self.m.matrix(p + "ffn_down.weight")
        h = silu(x @ Wg.T) * (x @ Wu.T)
        return h @ Wd.T

    # -------------------------------------------------------------- forward

    def forward(self, tokens: list[int], cap: Captures | None = None,
                start_pos: int = 0, max_layers: int | None = None) -> np.ndarray:
        """Returns logits (T, vocab) and fills `cap` if given."""
        hp = self.hp
        m = self.m
        T = len(tokens)
        x = np.stack([m.embedding_row(t) for t in tokens]).astype(F32)
        if cap is not None:
            cap.put("model.input_embed", x)
        pos = np.arange(start_pos, start_pos + T, dtype=F32)
        n_layers = hp.n_layer if max_layers is None else min(max_layers, hp.n_layer)

        for il in range(n_layers):
            inp = x.copy()
            norm_w = m.vector(f"blk.{il}.attn_norm.weight")
            cur = rms_norm(x, norm_w, hp.eps)
            if cap is not None:
                cap.put(f"attn_norm-{il}", cur)
            if m.is_recurrent(il):
                cur = self.layer_deltanet(il, cur, cap)
            else:
                cur = self.layer_attention(il, cur, pos, cap)
            cur = cur + inp
            if cap is not None:
                cap.put(f"attn_residual-{il}", cur)
            ffn_res = cur
            post_w = m.vector(f"blk.{il}.post_attention_norm.weight")
            cur = rms_norm(cur, post_w, hp.eps)
            if cap is not None:
                cap.put(f"attn_post_norm-{il}", cur)
            cur = self.layer_ffn(il, cur)
            if cap is not None:
                cap.put(f"ffn_out-{il}", cur)
            x = cur + ffn_res
            if cap is not None:
                cap.put(f"l_out-{il}", x)

        x = rms_norm(x, m.vector("output_norm.weight"), hp.eps)
        if cap is not None:
            cap.put("result_norm", x)
        logits = x @ m.matrix("output.weight").T
        return logits
