# AGENTS.md

Working instructions for AI coding agents in **omphalos**. These rules are mandatory.
[PLAN.md](./PLAN.md) is the source of truth for the design and its rationale; this file is the source of truth for *how work is done* in this repository.

## What this project is

**omphalos** is a from-scratch, highly optimized inference engine for a single model on a single GPU: `Qwen3.8-27B` (GSQ-RCO GGUF) on an AMD Radeon RX 9060 XT 16 GB (RDNA4, `gfx1200`, ROCm, Linux/CachyOS).

Priorities, in order: **1) VRAM savings — 2) decode speed — 3) prefill speed**. Vision runs on CPU and its latency is irrelevant. The engine is deliberately not generic: do not add generality it does not need.

### Non-negotiable constraints

- Weights stay **bit-exact** with the source GGUF: lossless reorder/re-layout only. **Never re-quantize.**
- Vision (`mmproj`) runs on **CPU**, never in VRAM.
- Python runs **only through `uv`** (`uv run` / `uvx`). Never `sudo pip`, never `--break-system-packages`, never the system Python.
- MTP must **not** slow down prefill.
- ROCm / HIP (`gfx1200`) is the reference backend.

## Language policy

- **Everything in the repository is written in English**: code, identifiers, comments, commit messages, branch names, issue and PR titles/bodies, documentation, file names.
- Conversation with the maintainer may be in Italian. The conversational language never leaks into artifacts: if a discussion happens in Italian, every commit, issue and document that comes out of it is still written in English.

## Git workflow

- **One branch per feature/fix; never commit directly to `main`.**
  - Naming: `<type>/<issue-number>-<short-slug>` — e.g. `feat/12-wmma-prefill-gemm`, `fix/31-deltanet-snapshot-offset`. If no issue exists yet, create one first (see below).
  - Types: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `chore`.
- **Commit frequently.** Many small, focused commits, each leaving the tree buildable; never batch unrelated changes into one commit.
- **Conventional Commits** in the imperative mood, in English: `feat: ...`, `fix: ...`, `docs: ...`, `perf: ...`.
- Reference the issue in commits: `Refs #<n>` while in progress, `Closes #<n>` when the work completes. Merge back to `main` through a PR (`gh pr create`) that closes the issue.
- Every AI-authored commit ends with the authoring agent's co-author trailer — for Command Code:

  ```
  Co-authored-by: CommandCodeBot <noreply@commandcode.ai>
  ```

  Pass the message via HEREDOC: `git commit -F - <<'EOF' ... EOF`.
- **Never commit large artifacts** (model files, converted weights, profiler dumps). `models/` is local-only and git-ignored. Benchmark results are the exception: CSV files under `bench/`, tagged with the engine commit hash (PLAN.md §17).

## GitHub issues — work tracking and decision log

`gh` is installed and authenticated (account **Nihmar**; repo `Nihmar/omphalos`). Use it directly — no additional setup is needed.

- **Track all work in issues.** Before starting anything non-trivial, create or find its issue: `gh issue create --title "..." --body "..."`. One issue per feature, fix, or milestone-sized unit; mention the relevant PLAN.md milestone ( §18) in the body.
- **Document decisions in the issue, not only in code.** When a design or implementation decision is made, add a comment with: context, alternatives considered, chosen option, rationale (`gh issue comment <n>`). The issue thread is the project's decision log; code comments and PR descriptions point back to it.
- When a decision changes the design, **update PLAN.md in the same change** and say so in the issue.
- When a `[verify]` item from PLAN.md ( §19) is resolved, record the finding in the relevant issue and update PLAN.md.

## Toolchain

- Engine: **C++20 + HIP**, CMake ≥ 3.21 (`CMAKE_HIP_ARCHITECTURES=gfx1200`).
- Tooling, reference implementation, validation: **Python via `uv`** + NumPy.
- Profiling and reference builds: `rocprofv3`, `hipblaslt-bench`, `rocminfo` / `amd-smi`, llama.cpp (HIP/Vulkan/CPU), `gguf-dump` (PLAN.md §6–§7).
- Once build/test/bench entry points exist, their exact commands are documented here.

## Entry points

Model path used below: `models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (local, git-ignored).

```sh
# build
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build -j

# full forward pass: prefill + greedy decode (options in the table below)
engine/build/omph-run <model.gguf> models/golden/cpu/tokens.txt <out-logits.f32> \
    --gemv --generate 3 --gen-out /tmp/gen.txt [--trace-dir DIR] [--tokens N]

# decode speed (PLAN.md §17): "step gpu" / "step wall" per token
OMPH_TIMING=1 engine/build/omph-run <model> <tokens.txt> /tmp/x.f32 --last-logits \
    --gemv --generate 65 --gen-out /tmp/gen.txt

# per-block checks against the golden dump (models/golden/cpu, local)
cd tools
../engine/build/omph-dequant <model> <tensor> /tmp/dq.raw
uv run python validate_gpu_dequant.py <model> <tensor> /tmp/dq.raw  # dequant kernel, bit-exact
uv run python check_gpu_linear.py <model> <tensor>  # matmul path
uv run python check_gpu_attn.py <model> <layer>   # attention layers 3 / 7 / 63 (attn_prep +
    # attention_gqa); [--kv q8q4|q4q4 --window 0] the quantized cache, [--chunk 1] the decode path
uv run python check_gpu_gdn.py  <model> <layer>   # delta-net layers 0 / 1 / 20 (gdn_step);
    # [--chunk 1|3] the decode path / the conv state carried across calls
uv run python check_gpu_run.py  <model> --layers  # 64-layer stack vs the dump
uv run python check_gpu_decode.py [--gemv]        # greedy decode vs the NumPy reference
                                                  # (cached in models/golden/cpu; --refresh)
uv run python compare_logits.py ref.f32 test.f32  # KL + top-1 agreement over every
                                                  # position (e.g. OMPH_KV_F32=1 vs default)

# long-context KV validation (exact f32 KV kept in host RAM as the reference)
OMPH_KV_HOST=1 engine/build/omph-run <model> <tokens.txt> ref.f32 --logits-tail 512
engine/build/omph-run <model> <tokens.txt> q8.f32 --logits-tail 512   # default: Q8/Q4 KV
bench/m5_llama_kv_kl.sh <llama.cpp-bin-dir> <ctx>   # llama.cpp's own KV-quant KL (budget)

# CPU-only tests (no GPU, no ROCm; what CI runs, .github/workflows/ci.yml)
cmake -S engine/tests -B engine/build-tests && cmake --build engine/build-tests -j
ctest --test-dir engine/build-tests --output-on-failure   # repack round trip, GGUF parser
cd tools && uv run ruff check . && uv run python -m pytest tests -q   # decoders vs gguf-py
uv run python check_doc_math.py ../docs/*.md ../PLAN.md        # math GitHub would mangle

# regenerating the golden dump (CPU backend, needs a llama.cpp build)
tools/native/build.sh <llama.cpp-dir> && tools/native/dump_tensors ...
```

### `omph-run` options and environment switches

| option | effect |
|---|---|
| `--gemv` | decode with the fused GEMVs on repacked weights: the fast path. Only effective with `--generate`; a prefill-only run stays on the f16 + hipBLASLt path |
| `--generate N --gen-out FILE` | greedy-decode N tokens after the prompt, ids to FILE |
| `--last-logits` / `--logits-tail N` | write only the last row / the last N rows of logits (long prompts) |
| `--tokens N` | use only the first N prompt tokens |
| `--trace-dir DIR` | dump every layer's output (one-chunk prompts only) |

The `OMPH_*` switches are parsed once, in `engine/src/runtime/options.{hh,cc}` (the authoritative list). Defaults are what the engine runs; ablations give wrong results with valid timings.

| variable | kind | effect |
|---|---|---|
| `OMPH_TIMING` | diagnostics | VRAM after load, per-step `step gpu` / `step wall` |
| `OMPH_PHASES` | diagnostics | per-phase GPU totals; its ~460 events per step add ~2.6 ms, so never measure the step with it (#100) |
| `OMPH_TRACE_ALLOC` / `OMPH_TRACE_F16` / `OMPH_TRACE_STAGE` | diagnostics | f16-scratch allocations / matmuls falling back to the f16 path / f16 staging and cache hits |
| `OMPH_KV_F32` | KV | exact f32 cache in VRAM (reference) |
| `OMPH_KV_HOST` | KV | exact f32 cache in pinned host RAM (long-context reference) |
| `OMPH_KV_K4` | KV | K in V's Q4 format too (#81, experiment) |
| `OMPH_KV_WINDOW=N` | KV | FP16 ring of the last N tokens (default 128, 0 = off) |
| `OMPH_NO_OVERLAP` | A/B | no side stream for sibling GEMVs (#71) |
| `OMPH_NO_B4` | A/B | no four-token GEMVs in a `--gemv` prefill |
| `OMPH_NO_BF16_GEMV` | A/B | BF16 weights through the f16 path |
| `OMPH_NO_F16_CACHE` | A/B | re-convert f16-path weights on every call |
| `OMPH_HOST_ARGMAX` | A/B | greedy argmax on the host instead of the device (#102) |
| `OMPH_SKIP_ATTN` / `OMPH_SKIP_FFN` / `OMPH_SKIP_BLOCKS` | ablation | no attention / no FFN / no blocks at all |
| `OMPH_SKIP_GEMV` / `OMPH_SKIP_GEMV_TYPE=T` / `OMPH_SKIP_STAGE` | ablation | no fused GEMVs / none of GGUF type T / no f16 + hipBLASLt matmuls |

`omph-gemv-bench` reads two of its own: `OMPH_BENCH_STREAMS=N` (alternate launches over N streams) and `OMPH_OCCUPANCY` (print the occupancy probe).

## Working agreements

- PLAN.md is the source of truth for architecture; deviations are proposed in an issue first.
- Every performance claim comes with a measurement following PLAN.md §17 (fixed conditions, median of ≥5 runs, results in `bench/`).
- "Done" means: tree builds, the CPU tests and CI pass, the relevant GPU checks of "Entry points" pass, results reported as measured — never claimed untested.
- Keep each change scoped to its issue; keep refactors separate from features.
