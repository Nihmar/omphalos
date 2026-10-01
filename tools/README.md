# omphalos tools

Python tooling for the engine, run through `uv` only (`uv run python <script>`,
from this directory). The package `omphalos_tools` (`src/`) holds:

- `gguf_meta`: GGUF metadata;
- `quant` / `quant_tables`: the quantization formats, bit-exact with ggml;
- `model`: a loader that keeps the tensors quantized;
- `reference`: the NumPy reference forward pass (PLAN.md §9);
- `golden`: the helpers the check scripts share (golden-dump loading, running
  the engine tools).

## Scripts

| script | checks |
|---|---|
| `inspect_gguf.py` | architecture sanity pass; writes `model-facts/` |
| `validate_dequant.py` | NumPy dequantization vs ggml, bit-exact |
| `validate_reference.py` | NumPy reference vs the golden dump, layer by layer |
| `validate_gpu_dequant.py` | an `omph-dequant` dump vs NumPy, bit-exact |
| `check_gpu_linear.py` | `omph-linear` (dequant → f16 → hipBLASLt) vs NumPy |
| `check_gpu_attn.py` | `omph-attn` (attention block, the engine's kernels) vs the dump |
| `check_gpu_gdn.py` | `omph-gdn` (delta-net block, `gdn_step`) vs the dump |
| `check_gpu_run.py` | `omph-run` (64-layer stack) vs the dump |
| `check_gpu_decode.py` | greedy decode vs the NumPy reference (cached) |
| `compare_logits.py` | KL / top-1 between two `omph-run` logits files |
| `extract_quant_tables.py` | regenerates the ggml tables (Python and C++) |
| `check_doc_math.py` | Markdown math that GitHub would mangle (CI) |
| `tests/` (pytest) | the NumPy decoders vs gguf-py, bit for bit (CI) |

The exact commands and the layers each check uses are in
[AGENTS.md](../AGENTS.md) ("Entry points"). The golden dump lives in
`models/golden/cpu` (local, git-ignored) and is regenerated with `native/`
(`build.sh` + `dump_tensors`) against a llama.cpp build.
