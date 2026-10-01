"""omphalos_tools.quant against gguf-py's own dequantization, bit for bit.

The NumPy reference (and through it every golden check) rests on these
decoders; validate_dequant.py checks them against libggml, which needs a
llama.cpp build. This runs anywhere (#105).
"""

from __future__ import annotations

import numpy as np
import pytest
from gguf import GGMLQuantizationType as GT
from gguf.quants import GGML_QUANT_SIZES
from gguf.quants import dequantize as gguf_dequantize

from omphalos_tools import quant

# Every type the model uses.
TYPES = [GT.F32, GT.BF16, GT.Q2_K, GT.Q4_K, GT.Q6_K, GT.IQ1_M, GT.IQ2_XXS, GT.IQ2_XS,
         GT.IQ2_S, GT.IQ3_XXS, GT.IQ3_S, GT.IQ4_XS]


@pytest.mark.parametrize("ggml_type", TYPES, ids=lambda t: t.name)
def test_dequantize_matches_gguf(ggml_type: GT) -> None:
    block, type_size = GGML_QUANT_SIZES[ggml_type]
    n_blocks = 512
    rng = np.random.default_rng(int(ggml_type))
    raw = rng.integers(0, 256, size=n_blocks * type_size, dtype=np.uint8)
    # Random f16 / bf16 scales are sometimes inf or NaN: the products are too,
    # in both implementations; compare them position for position.
    with np.errstate(invalid="ignore", over="ignore"):
        want = gguf_dequantize(raw.reshape(1, -1), ggml_type).reshape(-1).astype(np.float32)
        got = np.asarray(quant.dequantize(raw, ggml_type, n_blocks * block),
                         dtype=np.float32).reshape(-1)
    assert got.shape == want.shape
    assert np.isfinite(want).mean() > 0.9  # the comparison is about real numbers
    np.testing.assert_array_equal(got, want)


def test_rel_diff() -> None:
    from omphalos_tools.golden import rel_diff

    ref = np.array([1.0, -4.0, 2.0], dtype=np.float32)
    assert rel_diff(ref, ref) == 0.0
    assert rel_diff(ref + np.array([0.0, 0.0, 1.0], dtype=np.float32), ref) == pytest.approx(0.25)
