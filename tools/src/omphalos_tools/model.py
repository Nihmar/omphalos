"""GGUF model loader for the omphalos reference forward pass (PLAN.md §9).

Tensors stay quantized (raw ggml bytes) and are dequantized on demand with
`omphalos_tools.quant`, so the whole 27B model fits comfortably in RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gguf import GGUFReader

from . import gguf_meta as meta
from . import quant


@dataclass(frozen=True)
class HParams:
    n_layer: int
    n_embd: int
    n_ffn: int
    n_head: int
    n_head_kv: int
    head_dim: int
    n_rot: int
    rope_sections: tuple[int, ...]
    rope_freq_base: float
    eps: float
    full_attention_interval: int
    ssm_conv_kernel: int
    ssm_inner: int
    ssm_state: int
    ssm_dt_rank: int
    ssm_group: int
    vocab_size: int

    @property
    def n_rot_pairs(self) -> int:
        return self.n_rot // 2


class Model:
    """Lazy dequantizing GGUF reader (ggml tensor order: ne0 contiguous)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.reader = GGUFReader(self.path)
        self.fields = self.reader.fields
        self.tensors = {t.name: t for t in self.reader.tensors}
        self.hp = self._load_hparams()

    def _load_hparams(self) -> HParams:
        f = self.fields
        arch = meta.text(f, "general.architecture")
        n_all = meta.get_int(f, f"{arch}.block_count", 0)
        n_nextn = meta.get_int(f, f"{arch}.nextn_predict_layers", 0)
        embd_shape = [int(v) for v in self.tensors["token_embd.weight"].shape]
        return HParams(
            n_layer=n_all - n_nextn,
            n_embd=meta.get_int(f, f"{arch}.embedding_length", embd_shape[0]),
            n_ffn=meta.get_int(f, f"{arch}.feed_forward_length", 0),
            n_head=meta.get_int(f, f"{arch}.attention.head_count", 0),
            n_head_kv=meta.get_int(f, f"{arch}.attention.head_count_kv", 0),
            head_dim=meta.get_int(f, f"{arch}.attention.key_length", 0),
            n_rot=meta.get_int(f, f"{arch}.rope.dimension_count", 0),
            rope_sections=tuple(meta.get_list(f, f"{arch}.rope.dimension_sections", [0, 0, 0, 0])),
            rope_freq_base=meta.get_float(f, f"{arch}.rope.freq_base", 10000.0),
            eps=meta.get_float(f, f"{arch}.attention.layer_norm_rms_epsilon", 1e-6),
            full_attention_interval=meta.get_int(f, f"{arch}.full_attention_interval", 4),
            ssm_conv_kernel=meta.get_int(f, f"{arch}.ssm.conv_kernel", 0),
            ssm_inner=meta.get_int(f, f"{arch}.ssm.inner_size", 0),
            ssm_state=meta.get_int(f, f"{arch}.ssm.state_size", 0),
            ssm_dt_rank=meta.get_int(f, f"{arch}.ssm.time_step_rank", 0),
            ssm_group=meta.get_int(f, f"{arch}.ssm.group_count", 0),
            vocab_size=embd_shape[1],
        )

    def is_recurrent(self, il: int) -> bool:
        """Linear-attention (gated delta net) layers: (il+1) % interval != 0."""
        return il < self.hp.n_layer and (il + 1) % self.hp.full_attention_interval != 0

    def raw(self, name: str) -> np.ndarray:
        # gguf-py returns typed arrays (float32 for F32, uint8 for quantized):
        # reinterpret as raw bytes for the dequantizers.
        data = np.ascontiguousarray(self.tensors[name].data)
        return data.view(np.uint8).reshape(-1)

    def dequant(self, name: str) -> np.ndarray:
        """Dequantized tensor, flat in ggml order (ne0 contiguous)."""
        t = self.tensors[name]
        return quant.dequantize(self.raw(name), t.tensor_type)

    def shape(self, name: str) -> list[int]:
        return [int(v) for v in self.tensors[name].shape]

    def vector(self, name: str) -> np.ndarray:
        """1D tensor as (ne0,)."""
        return self.dequant(name)

    def matrix(self, name: str) -> np.ndarray:
        """2D tensor as (out, in) — ggml stores [in=ne0, out=ne1]."""
        ne = self.shape(name)
        if len(ne) == 1:
            return self.dequant(name)
        flat = self.dequant(name)
        return flat.reshape(tuple(reversed(ne)))

    def embedding_row(self, token: int) -> np.ndarray:
        """Row `token` of token_embd.weight (ggml [n_embd, n_vocab])."""
        t = self.tensors["token_embd.weight"]
        block, type_size = quant.GGML_QUANT_SIZES[t.tensor_type]
        row_bytes = self.hp.n_embd // block * type_size
        raw = self.raw("token_embd.weight")
        off = token * row_bytes
        return quant.dequantize(raw[off:off + row_bytes], t.tensor_type, self.hp.n_embd)
