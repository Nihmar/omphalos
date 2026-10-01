# Custom Inference Engine — Qwen3.8-27B (GSQ-RCO GGUF) on AMD RX 9060 XT 16 GB (ROCm, Linux)

*Design notes and summary of the discussion — 30 September 2026*

> **Conventions used in this document**
> - `~` marks an estimate. Estimates are there to reason about orders of magnitude, not as promises.
> - **[verify]** marks something that must be checked against GGUF metadata, the reference implementation, AMD specs, or a measurement before relying on it.
> - Model dimensions used in examples (hidden size 5120, 16 full-attention layers, 4 KV heads, head_dim 256, …) are **illustrative placeholders**. Read the real values with `gguf-dump`.

---

## Table of contents

1. [Goals and constraints](#1-goals-and-constraints)
2. [Hardware facts](#2-hardware-facts)
3. [Model facts](#3-model-facts)
4. [Performance model](#4-performance-model)
5. [Engine architecture overview](#5-engine-architecture-overview)
6. [Toolchain, languages and tools](#6-toolchain-languages-and-tools)
7. [Step 0 — Baseline with llama.cpp](#7-step-0--baseline-with-llamacpp)
8. [Custom weight format (lossless)](#8-custom-weight-format-lossless)
9. [Loader, reference implementation, validation](#9-loader-reference-implementation-validation)
10. [GPU kernels](#10-gpu-kernels)
11. [Runtime and decode loop](#11-runtime-and-decode-loop)
12. [MTP speculative decoding without hurting prefill](#12-mtp-speculative-decoding-without-hurting-prefill)
13. [KV cache quantization](#13-kv-cache-quantization)
14. [Vision encoder on CPU](#14-vision-encoder-on-cpu)
15. [VRAM budget](#15-vram-budget)
16. [Performance "breadcrumbs"](#16-performance-breadcrumbs)
17. [Measurement methodology](#17-measurement-methodology)
18. [Milestones](#18-milestones)
19. [Open questions / verification checklist](#19-open-questions--verification-checklist)
20. [References](#20-references)

---

## 1. Goals and constraints

**Scope.** Single user, batch size 1 (plus small batches of 2–5 tokens for speculative verification), one model, one GPU, Linux (CachyOS). The engine is deliberately *not* generic: it only has to run this model, with this quantization allocation, on this card.

**Priorities, in order:**

1. **VRAM savings** — the card has 16 GB, shared with the desktop (no iGPU, see §2).
2. **Decode speed** (tokens/s during generation).
3. **Prefill speed** (prompt processing tokens/s).
4. Vision latency is **irrelevant**; vision must work but can be slow.

**Hard constraints:**

| Constraint | Consequence |
|---|---|
| Weights must stay **bit-exact** with the GSQ-RCO GGUF | Any custom format may only *reorder / re-layout* bits. No re-quantization. Verified by per-tensor dequantization comparison. |
| **Vision must be kept**, running on **CPU** | `mmproj` never touches VRAM; image embeddings are uploaded to the GPU (a few MB). |
| **KV cache quantizable** to Q8 and Q4, with **mixed** types | Per-K/V and per-layer types; rotation + FP16 window to make Q4 viable. |
| **MTP must not slow down prefill** | KV-only "shadow" pass or deferred fill; no `lm_head` over the prompt. |
| **ROCm / HIP** is the reference backend | C++/HIP engine, gfx1200 target. |
| **Python only through `uv`** | Never touch the CachyOS system Python (no `sudo pip`, no `--break-system-packages`). See §6.3. |

---

## 2. Hardware facts

### GPU — AMD Radeon RX 9060 XT 16 GB

| Property | Value | Notes |
|---|---|---|
| Architecture | RDNA4, `gfx1200` | ROCm target string |
| Compute units | 32 | wave32 execution |
| VRAM | 16 GB GDDR6, 128-bit bus | |
| Memory bandwidth (spec) | ~320 GB/s | **Measured (M0): 318.3 GB/s** streaming read — 99.5% of spec, not the usual 85–90% (`bench/bw_membench.hip`) |
| FP16 matrix (WMMA) throughput | ~100 TFLOPS dense (spec) | **Measured (M0): ~46 TFLOPS** at the model's M=512 shapes via hipBLASLt; INT8 measured **~26 TOPS (0.57×)**, i.e. *not* ~2× |
| LDS | 64 KB per workgroup | Enough for IQ codebook tables + GEMM tiles |
| Bus | PCIe 5.0 x16 | Host↔device transfers of embeddings/logits are cheap |

### CPU — Intel Core i5-13400F

| Property | Value | Consequence |
|---|---|---|
| Cores | 6 P-cores + 4 E-cores, 16 threads | Pin vision encoder to P-cores, keep one P-core for the GPU driver thread |
| SIMD | AVX2, FMA, AVX-VNNI | Good int8 dot products on CPU |
| Missing | **No AVX-512, no native BF16** | Vision weights converted BF16 → FP32 at load (~1.8 GB RAM) |
| Graphics | **None ("F" SKU)** | **The desktop runs on the RX 9060 XT** → compositor, browser, editor eat VRAM |

### OS

- **CachyOS** (Arch-based). ROCm from the distro repos; Python tooling exclusively through `uv`.

---

## 3. Model facts

Source: `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` model card.

### Files

| File | bpw | Size | Notes from model card |
|---|---|---|---|
| `Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf` | 2.50 | 8.4 GB | Smallest |
| `Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf` | 2.75 | 9.3 GB | Matches base on AIME25 |
| `Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf` | 3.00 | 10.1 GB | Strong all-round point |
| `Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf` | 3.50 | 11.8 GB | Recommended, "task-lossless" |
| `*-mtp` variants | | +~0.35 GB | Same weights + MTP head (15 extra tensors) |
| `mmproj-Qwen3.8-27B-BF16.gguf` | 16 | 0.9 GB | Vision encoder + projector, BF16 |
| `tensor-allocation/<model>.rco-allocation.txt` | | | Quant type of **every** tensor + histogram |
| `imatrix-qwen3.8-27b.gguf` | | | Importance matrix used for quantization |

**Measured facts (M0, `gguf-dump` + allocation file, IQ3_S-mtp, 2026-09-30):**

- 65 blocks (64 main + `blk.64` MTP), hidden 5120, FFN 17408, 24 heads / 4 KV heads, head_dim 256 (K = V = 256), vocab 248320.
- Layer pattern: **48 Gated DeltaNet layers** (fused `attn_qkv` + `ssm_*`, output via `ssm_out`) and **16 full-attention layers** at indices 3, 7, …, 63 (separate `attn_q`/`attn_k`/`attn_v` + `attn_q_norm`/`attn_k_norm` + `attn_output`). `blk.64` (MTP) is a 17th attention-style layer.
- **Untied embeddings**: separate `output.weight` (Q4_K); `token_embd` is IQ2_S → can go to host RAM (§8.3).
- RoPE: `freq_base` 1e7, `rope.dimension_count` 64 (partial), M-RoPE `rope.dimension_sections = [11, 11, 10, 0]`.
- Gated DeltaNet: `ssm.state_size` 128, `ssm.group_count` 16, `ssm.inner_size` 6144, `ssm.time_step_rank` 48, `conv_kernel` 4.
- Quant mix (866 tensors): BF16 96, F32 360, IQ1_M 1, IQ2_S 17, IQ2_XS 9, IQ2_XXS 5, IQ3_S 144, IQ3_XXS 78, IQ4_XS 96, Q2_K 13, Q4_K 39, Q6_K 8. `lm_head` = Q4_K, MTP-layer weights all Q6_K.
- **`gate`/`up` do *not* share the quant type in 40 of 65 layers** (e.g. blk.0: IQ2_XS vs IQ2_XXS) → interleaving is only free where the types match (§8.3 ⚠️).

### What matters for the engine

1. **Non-uniform, per-tensor quantization (RCO).** Each tensor has its own GGUF quant type. The allocation file is the authoritative list of which dequantization kernels the engine needs (IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S plus whatever higher-precision types sensitive tensors received, e.g. Q4_K/Q5_K/Q6_K/Q8_0/F16 **[verify]**).
2. **GSQ grid assignments are the value of this model.** They were learned to minimize error inside the standard IQ codebooks. Re-quantizing destroys this → everything must be lossless.
3. **Architecture `qwen35`** = Qwen3.5-style hybrid: most layers are **Gated DeltaNet** (linear attention with a recurrent state + short causal conv1d), a minority are **full gated attention** layers with a KV cache. **[verify]** the exact layer pattern, head counts, head dims, RoPE configuration (possibly partial rotary and M-RoPE sections), norm variants, and whether embeddings are tied to `lm_head`.
4. **MTP head** (in `-mtp` builds): one extra decoder layer + `fc` projection, sharing final norm and `lm_head` with the main model. Enables self-speculative decoding.
5. **Multimodal** via `mmproj` (ViT encoder + projector), image tokens use multi-dimensional rotary positions (M-RoPE) in the attention layers **[verify]**.

---

## 4. Performance model

### 4.1 Decode is memory-bandwidth-bound

At batch 1, every generated token reads (almost) all weights once. Bytes read per token ≈ file size − token embedding table (only one row of the embedding is read per token) − MTP tensors (if not used) − `mmproj` (not on GPU).

Assuming the token embedding table is ~0.5–1 GB **[verify]**:

| File | ~Bytes/token | Ceiling @ 320 GB/s (spec) | Measured llama.cpp: 19–20 t/s is… |
|---|---|---|---|
| IQ3_S | ~11.1 GB | ~29 t/s | ~68% of spec |
| IQ3_XXS | ~9.4 GB | ~34 t/s | ~57% |
| IQ2_S | ~8.6 GB | ~37 t/s | ~52% |

**Measured baseline (llama.cpp, M0):** decode **20.6 t/s** without MTP (IQ3_S-mtp, f16 KV, d0); the original 19–20 t/s figure is confirmed.

**Interpretation.** The real 100% is the bandwidth a streaming-read kernel achieves. **Measured (M0): 318.3 GB/s — 99.5% of spec** (`bench/bw_membench.hip`), so for this file (~11.28 GiB of weights per token) the decode ceiling is ~26 t/s. The 19–20 t/s baseline is ~73% of that; a custom engine could plausibly reach ~24–26 t/s from kernel work alone. The larger lever is MTP (§4.3).

Other per-token traffic (small but not zero):

- Gated DeltaNet recurrent states: read + write every token (state stays FP32). Size per layer = heads × d_k × d_v × 4 bytes (+ conv tail). Likely ~1–3% of weight traffic **[verify]**.
- KV cache reads in full-attention layers: grows with context; this is where KV quantization also *speeds up* decode at long contexts.

### 4.2 Prefill is compute-bound

**Where milestone 2/3 actually stand (measured, 2026-09-30).** The current prefill
is the *correctness* path, not a prefill design. A 10-token pass costs 2.0 s wall
clock, ~0.56 s of which is the one-time weight upload over PCIe; the pass itself
is ~1.45 s = **~7 t/s**. `OMPH_TIMING=1` splits it:

| phase | time | share |
|---|---:|---:|
| dequant each layer's weights to f16 (`stage_w`) | 830 ms | 57 % |
| f16 GEMMs (hipBLASLt) | 339 ms | 23 % |
| attention / gated delta net kernels | 43 ms | 3 % |
| small ops, casts, launches | ~200 ms | 14 % |

80 % of the pass is therefore exactly what the M8 design replaces, and the block
kernels are 3 % *at T = 10* — they are what grows with T, since the delta net is
sequential today, which is why §10.3's chunked form matters for long prompts.
With `--gemv` the pass is 8.7 s instead (the f16 fallback re-stages raw bytes):
use `--gemv` for decode measurements only.

Against llama.cpp's measured **622.7 t/s** (pp512, M0) and the ~700-800 t/s
target, four things separate us:

1. **f16 materialization.** The M2 path dequantizes every weight to f16 in global
   memory per pass: 27.5 B params x 2 B = **54 GB of writes** plus 11.28 GB of
   reads. Dequantizing into LDS inside the GEMM removes almost all of it.
2. **Small/skinny GEMMs.** hipBLASLt at M = 10 is nowhere near the measured
   **46 TFLOPS fp16 at M = 512** — and fp16 WMMA is the only matrix path this GPU
   has (the int8 tile does not exist on gfx1200, see §10.5 and
   `bench/results/m3-isa-probe.txt`).
3. **Sequential DeltaNet.** The decode-shaped gated delta net runs one token per
   step: O(T) dependent launches (~120k for a 512-token prompt). The chunked
   WY/UT form (§10.3) is what removes that, and it is the hardest piece in the
   project.
4. **Naive attention and no fusions** (no flash-style tiles, no RMSNorm/epilogue
   fusion).

**The target is reachable, and the arithmetic says why:** 512 tokens x 27.5 B
params x 2 = ~28 TFLOP; at the *measured* 46 TFLOPS that is ~0.6 s, i.e.
**~850 t/s**. So the ~700-900 t/s goal is exactly this GPU's fp16 WMMA compute
limit — there is no headroom to buy with cleverness, only with dequant-into-LDS,
large M, the chunked DeltaNet and tiled attention. That is milestone 8, which is
why prefill is priority 3.


During prefill, weights are read once per micro-batch (e.g. 512 tokens) and reused for every token in it. At 622.7 t/s with ~11.28 GiB of weights and ubatch 512, the card reads only ~14 GB/s (~4% of bandwidth).

- Dense FLOPs ≈ 2 × params per token ≈ 2 × 27 B = **~54 GFLOP/token**.
- **Measured baseline (llama.cpp, M0):** **622.7 t/s** prefill at empty context (pp512, f16 KV, `-t 6`) → ≈ **34 TFLOPS achieved**. The historical "~750 t/s" figure did not reproduce.
- Against the measured fp16 GEMM ceiling (~46 TFLOPS, §2) → **~74% of it**; against the ~100 TFLOPS spec peak → ~34%.
- Target for a well-tuned dequant + WMMA path: originally **~50–60% of peak → ~900–1100 t/s**; the M0 measurement revises this to **~700–800 t/s**. Hard: IQ decoding (codebook lookups, sign unpacking, scales) competes with the matrix units; DeltaNet chunked prefill and attention add FLOPs that are less matrix-friendly.
- **M0 update (measured):** the fp16 GEMM ceiling at the model's M=512 shapes is **~46 TFLOPS** (hipBLASLt; §2) — the prefill ceiling for a dequant+WMMA path is ~**850 t/s** before dequant overhead. The INT8 route measured *slower* than fp16 (~26 TOPS, 0.57×) — do not count on a ~2× INT8 speedup until a raw WMMA check confirms it (§10.5).
- The real ceiling is measured (M0) with `bench/hipblaslt_gemm_bench.hip` at the model's projection shapes (M = 512); `hipblaslt-bench` itself is not packaged on this distro.

### 4.3 What MTP can give on decode

With `k` draft tokens and per-token acceptance rate `α`, one verification step yields on average:

```
E[tokens/step] = (1 − α^(k+1)) / (1 − α)
```

Step cost = verification pass (batch k+1, only slightly more expensive than batch 1 because decode is memory-bound) + k draft passes.

**Hidden cost of drafting:** each draft pass runs the MTP layer **and the full `lm_head`** (vocab × hidden, ~0.7–1.3 GB depending on its quant type **[verify]** in allocation file) → each draft re-reads ~10% of the model. **Truncated-vocab drafting** (only the ~32k most frequent tokens, see §12.5) makes the draft `lm_head` ~5–8× cheaper.

Illustrative, α = 0.7, baseline 19.5 t/s:

| Setup | Tokens/step | Relative step cost | ~Decode |
|---|---|---|---|
| k=2, full `lm_head` in drafts | 2.19 | ~1.35 | ~32 t/s |
| k=2, truncated-vocab drafts | 2.19 | ~1.20 | ~36 t/s |
| k=3, truncated-vocab drafts | 2.53 | ~1.30 | ~38 t/s |

α depends heavily on content: code and structured text accept well, creative prose less. Measure per workload.

### 4.4 Why llama.cpp prefill drops from ~750 to ~500 t/s with MTP

> **M0 update (2026-09-30): not reproduced.** With the current build (upstream code @ `6c7a87f7e`, `draft-mtp` n=2, prompt ~2.9k tokens) prefill costs **~3–4%** (client −3/−4%; kernel time 5.268 → 5.427 s under `rocprofv3`). The "750 t/s" figure itself did not reproduce either (622.7 t/s pp512, f16 KV, `-t 6`). The analysis below is kept as historical context.

Prefill is compute-bound, so extra MTP work costs time roughly proportional to its FLOPs. Rough cost of MTP work per prompt token, relative to the main model:

| MTP work per prompt token | ~Relative cost |
|---|---|
| `fc` + K/V projections only (what is actually *needed*) | ~0.2–0.3% |
| Full MTP layer (attention + MLP) | ~1.5–2% |
| Full MTP layer + `lm_head` on every token | ~6–7% |

Even the most naive variant should give ~700 t/s, not 500. The remaining ~25–30% is **scheduling overhead** in the current implementation. Candidates (to confirm with `rocprofv3 --kernel-trace`, with vs. without the `-mtp` file):

- Outputs requested for **all** tokens → the *main* `lm_head` also runs over the whole prompt.
- MTP forces smaller micro-batches.
- MTP graph runs as a separate pass with host syncs / copies of hidden states.
- MTP invoked per token or per ubatch instead of batched.

Our design (§12) keeps prefill within ~1–2% of the non-MTP number.

---

## 5. Engine architecture overview

```
                        ┌──────────────────────────── CPU (i5-13400F) ─────────────────────────────┐
  prompt text ─────────►│ tokenizer ─┐                                                            │
  image(s) ────────────►│ preprocess ─► ViT encoder + projector (FP32, P-cores) ─► image embeds   │
                        │            │                                   (cache by image hash)   │
                        │ host thread (pinned P-core): graph replay, sampling result readback,   │
                        │   detokenize/stream, token_embd rows (pinned RAM, if untied)            │
                        └──────────────┬───────────────────────────────────────────┬─────────────┘
                                       │ token ids / embeddings (KB–MB)             │ token id (4 B)
                                       ▼                                            │
┌────────────────────────────────── GPU (RX 9060 XT, HIP) ─────────────────────────┴────────────────┐
│ Weights: custom lossless layout (SoA, tile-major, rows reordered for fusion)                      │
│                                                                                                   │
│  per layer:  [RMSNorm⊕GEMV-in] → DeltaNet (conv1d + gated delta rule, FP32 state, snapshots)      │
│              or                  Gated full attention (RoPE/M-RoPE, Hadamard, quantized KV)      │
│              → [GEMV-out ⊕ residual] → [RMSNorm⊕gate/up interleaved ⊕ SwiGLU] → [down ⊕ residual]│
│                                                                                                   │
│  head:       final norm ⊕ lm_head (row-sorted by token frequency) ⊕ argmax / sampling on GPU     │
│  MTP:        fc ⊕ MTP layer (own quantized KV) ⊕ truncated lm_head prefix → k drafts             │
│  verify:     same kernels with N = k+1 (small-batch GEMV) + DeltaNet per-position snapshots      │
│  prefill:    dequant→WMMA GEMM, chunked DeltaNet, flash attention, KV-only MTP shadow pass       │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Toolchain, languages and tools

### 6.1 Languages

| Component | Language | Why |
|---|---|---|
| Engine runtime + kernels | **C++20 + HIP** | Direct access to RDNA4 intrinsics (`v_dot4_i32_i8`, gfx12 WMMA builtins), HIP graphs, rocprof tooling; CUDA knowledge transfers directly |
| Build | **CMake** (≥ 3.21, native HIP language support) | `CMAKE_HIP_ARCHITECTURES=gfx1200` |
| Offline converter, inspection, reference model, validation, frequency tables | **Python via `uv`** + NumPy | Runs rarely; NumPy is fast enough for bit repacking of 12 GB if vectorized |
| Vision encoder | C/C++ on CPU: llama.cpp's `libmtmd` (CPU-only build) first, optionally own ViT later | Preprocessing is the easiest thing to get silently wrong |
| Public interface | **C ABI** shared library (`libomphalos.so`) + small **OpenAI-compatible HTTP server** | C ABI is callable from anything (Python `ctypes`, Delphi on Linux64, …); HTTP server lets existing clients/agents work unchanged |

Optional prototyping language for kernels: Triton on ROCm, via `uv` (**[verify]** gfx12 support in the Triton version you get). Useful to try tiling ideas quickly; final kernels stay in HIP.

### 6.2 ROCm on CachyOS

```bash
# ROCm HIP SDK + tools (package names per Arch repos — [verify] on CachyOS)
sudo pacman -S rocm-hip-sdk rocminfo rocprofiler-sdk amdsmi cmake ninja

rocminfo | grep -i gfx          # expect gfx1200
amd-smi static                  # card info
```

ROCm has official RDNA4 support since the 6.4.x series **[verify]** exact minimum version for the 9060 XT; ROCm 7.x recommended if available.

### 6.3 Python — strictly through `uv`

Rules:

- **Never** `sudo pip`, never `pip install --break-system-packages`, never install Python packages with `pacman` for this project.
- Every Python command runs via `uv run …` (project env) or `uvx …` (one-off tools).
- The interpreter itself is managed by `uv` (`uv python install`), independent of the system Python.

```bash
sudo pacman -S uv                       # uv itself from the repos is fine (it's a static binary)

uv python install 3.12
mkdir -p ~/Projects/omphalos/tools && cd ~/Projects/omphalos/tools
uv init --name omphalos-tools --python 3.12
uv add numpy gguf huggingface_hub tokenizers

# inspect GGUF metadata and tensor list
uv run gguf-dump ../models/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf | less

# download only what is needed
uv run hf download ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF \
    --include "*IQ3_S*" "mmproj*" "tensor-allocation/*" \
    --local-dir ../models

# one-off tool without adding it to the project
uvx --from gguf gguf-dump --help
```

Optional: PyTorch (ROCm wheels) for a GPU-side reference or GEMM experiments. The ROCm wheels bundle their own ROCm runtime and do not touch the system ROCm. In `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm6.4"   # [verify] current ROCm wheel index on pytorch.org
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-rocm" }
```

```bash
uv add torch
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

A PyTorch **CPU** reference is often enough (and simpler) for correctness work; keep the GPU wheel optional.

### 6.4 Profiling and inspection tools

| Tool | Use |
|---|---|
| `rocprofv3` (rocprofiler-sdk) | Kernel traces, per-kernel timings, counters. First tool to reach for. |
| `rocprof-compute` (ex Omniperf) | Roofline / detailed counter analysis — **[verify]** RDNA4 support level |
| Radeon GPU Profiler (RGP) | Very detailed wave-level analysis, mainly for Vulkan/DX12 |
| Radeon GPU Analyzer (RGA) / `--save-temps` | Inspect generated ISA: check vectorized loads (`global_load_b128`), VGPR count, spills |
| `-Rpass-analysis=kernel-resource-usage` | Compiler report of VGPR/SGPR/LDS/occupancy per kernel |
| `amd-smi` / sysfs `mem_info_vram_used` | VRAM usage, clocks, power, temperature |
| LACT | GPU clocks, power limit, fan curve, (memory) overclocking on Linux — **[verify]** RDNA4 memory OC support |
| `perf`, `htop` | CPU side (vision encoder, host thread) |
| llama.cpp (`llama-bench`, `llama-perplexity --kl-divergence`, `llama-eval-callback`) | Baseline speed, quality reference, intermediate tensor dumps |

### 6.5 Suggested repository layout

```
omphalos/
├── engine/                 # C++/HIP, CMake
│   ├── include/omphalos.h   # C ABI
│   ├── src/
│   │   ├── format/         # custom format reader (+ load-time repacker during development)
│   │   ├── kernels/        # gemv_iq3s.hip, gemv_iq2s.hip, deltanet_decode.hip, attn_decode.hip, ...
│   │   ├── runtime/        # graphs, streams, memory arena, sampling, MTP scheduler
│   │   └── server/         # OpenAI-compatible HTTP server
│   └── tests/              # per-kernel tests vs golden tensors
├── tools/                  # uv project: converter, gguf inspection, reference model, validation
├── third_party/llama.cpp   # CPU-only build for libmtmd + reference builds (HIP/Vulkan)
├── models/                 # GGUF + converted files (git-ignored)
└── bench/                  # benchmark scripts + results (CSV)
```

Minimal CMake skeleton:

```cmake
cmake_minimum_required(VERSION 3.21)
project(omphalos LANGUAGES CXX HIP)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_ARCHITECTURES gfx1200)
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -Rpass-analysis=kernel-resource-usage")

add_library(omphalos SHARED
  src/format/reader.cpp
  src/kernels/gemv_iq3s.hip
  # ...
)
target_include_directories(omphalos PUBLIC include)
```

---

## 7. Step 0 — Baseline with llama.cpp

Before writing any engine code, establish the bar and the numerical reference.

> **Status (M0, 2026-09-30): done.** Scripts and results under `bench/` (issue #3): HIP/Vulkan llama-bench suites, MTP A/B, bandwidth ceiling, GEMM ceilings, prefill traces, perplexity/KL baseline.

```bash
cd ~/Projects/omphalos/third_party
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp

# HIP / ROCm build
cmake -B build-hip -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1200 -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip

# Vulkan (RADV) build — sometimes faster than HIP on RDNA, worth comparing
cmake -B build-vk -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk

# CPU-only build (for libmtmd / vision encoder and CPU references)
cmake -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu
```

Benchmarks to record (headless or at least with a minimal desktop, see §17):

```bash
M=../../models/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf

# decode + prefill, flash attention on
./build-hip/bin/llama-bench -m $M -ngl 99 -fa 1 -p 512 -n 128
./build-vk/bin/llama-bench  -m $M -ngl 99 -fa 1 -p 512 -n 128

# decode at longer contexts
./build-hip/bin/llama-bench -m $M -ngl 99 -fa 1 -n 128 -d 4096,16384

# KV cache quantization baselines
./build-hip/bin/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -n 128 -d 16384
./build-hip/bin/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q4_0 -n 128 -d 16384
```

Also record:

- The same runs with the `-mtp` file (decode t/s → lets us back out llama.cpp's acceptance rate; prefill t/s → the 750 → 500 drop).
- `rocprofv3 --kernel-trace` of one prefill with and without MTP (§4.4).
- **Achievable bandwidth**: a trivial HIP kernel that sums a ~10 GB buffer with 128-bit loads. That number is the real 100% for decode.
- **Achievable FP16 GEMM throughput**: `hipblaslt-bench` with the model's projection shapes at M = 512. That is the real 100% for prefill (minus dequant overhead).
- Reference logits / KL baseline: `llama-perplexity --kl-divergence-base` on a fixed text, used later to compare the engine.

---

## 8. Custom weight format (lossless)

### 8.1 Principle

**Reorder and re-layout bits freely; never change a dequantized value.** Every converted tensor is verified by dequantizing both the original GGUF tensor and the converted tensor and comparing **bit-exactly**. This gives a mathematical guarantee that GSQ-RCO quality is untouched.

Rejected ideas:

- **Re-quantizing** into a GPU-"native" format → destroys GSQ grid assignments and RCO allocation.
- **Expanding** IQ indices into wider, easier-to-decode encodings → more bytes per weight → directly slower decode (bandwidth-bound).

### 8.2 Why the GGUF block layout is GPU-unfriendly

GGUF stores each block as an array-of-structs. Block sizes for 256 weights (**[verify]** against `ggml-common.h`):

| Type | Block bytes | Fields |
|---|---|---|
| IQ2_XS | 74 | `d` (fp16), `qs[32]` (uint16: grid idx + signs), `scales[8]` |
| IQ2_S | 82 | `d`, `qs[64]`, `qh[8]`, `scales[8]` |
| IQ3_XXS | 98 | `d`, `qs[96]` (grid indices + packed sign/scale words) |
| IQ3_S | 110 | `d`, `qs[64]`, `qh[8]`, `signs[32]`, `scales[4]` |

None of these is a multiple of 16 bytes → misaligned loads, no clean 128-bit vector loads, fields interleaved so a wave's lanes don't read contiguous streams.

Codebook tables (**[verify]** sizes): IQ2_XS grid 512 × 8 B, IQ2_S grid 1024 × 8 B, IQ3_XXS grid 256 × 4 B, IQ3_S grid 512 × 4 B, plus sign tables → all fit comfortably in LDS.

### 8.3 Transformations (all lossless)

1. **Array-of-structs → struct-of-arrays.** Separate streams per field: all `qs`, all `qh`, all `signs`, all `scales`, all `d`. Each stream aligned to 256 B (or more). Lanes issue aligned `global_load_b128`, perfectly coalesced.
2. **Tile-major / kernel-order layout.**
   - For GEMV: interleave rows so a workgroup reads one contiguous strip; each lane's chunk is 16 B-aligned; the order matches exactly the loop order of the kernel.
   - For prefill GEMM: tiles laid out as they are loaded into LDS for WMMA.
   - Possibly two layouts are not affordable (VRAM) → pick the GEMV-optimal layout and make the GEMM kernel adapt (decode is the priority).
3. **Row reordering (free: blocks run along K, the input dimension).** Permuting *rows* (outputs) never touches a block.
   - **Interleave `gate` and `up`** row by row → one workgroup produces `gate_i` and `up_i` and applies SwiGLU in its epilogue. No separate activation pass, no intermediate write.
   - **Group Q/K/V by GQA group** (and the output gate of gated attention) → one workgroup computes the query heads and their KV head together → fuse RoPE, QK-norm, Hadamard rotation, KV quantization + cache write.
   - Same idea for the DeltaNet input projections (q/k/v/gates/β grouped per head).
   - **Sort `lm_head` rows by token frequency** (store the permutation): truncated-vocab MTP drafting becomes "read the first N rows of the same matrix". Zero extra VRAM. If embeddings are tied to `lm_head`, the embedding lookup uses the same permutation.
   - ⚠️ If RCO assigned **different quant types** to `gate` and `up` (or to Q/K/V), they cannot become one homogeneous matrix. Then the interleaved kernel must handle two types in one launch (e.g. per-row-group type tag). Check the allocation file first **[verify]**.
4. **Placement decisions baked into the file.**
   - `token_embd` → **host pinned RAM** (only one row per token is needed) if not tied **[verify]**. Saves ~0.5–1 GB VRAM.
   - MTP tensors in their own section → not loaded when MTP is off.
   - Vision tensors in a host-only section (or keep reading `mmproj` GGUF via libmtmd).
5. **Execution order and alignment.** Tensors stored in the order they are used, sections aligned to 2 MB → streaming load, simple mmap.

### 8.4 Proposed file structure

```
[Header]
  magic "OMPH", format version, layout version
  source GGUF sha256 (+ mtp/mmproj sha256)
  model hyperparameters (copied from GGUF metadata)
  tokenizer + chat template (or pointer to source GGUF)
[Tensor table]   one entry per tensor:
  name, GGUF quant type, logical shape, layout id, row-permutation id,
  placement (GPU / host / MTP / vision), section, offset, size, xxhash of converted bytes,
  "verified bit-exact" flag + hash of dequantized values
[Permutation tables]   e.g. lm_head frequency order, gate/up interleave maps
[Constant tables]      IQ grids and sign tables (or compiled into the engine)
[Section: GPU weights]        2 MB aligned, execution order
[Section: host weights]       token_embd (if untied)
[Section: MTP weights]
[Section: vision weights]     optional
```

### 8.5 Offline vs. load-time conversion

- Load-time speed is **not** the argument: a GPU repack kernel transforms 12 GB in well under a second, negligible versus reading from NVMe.
- Real benefits of offline: bit-exact verification done **once**; expensive transformations (frequency statistics, per-tensor layout choices) outside the engine; a simpler engine that reads exactly what the kernels expect.
- Cost: one more component with its own versioning.

**Plan:** implement **load-time repacking first** (fast iteration on layouts while writing kernels), then move the same logic into an offline converter (Python + NumPy via `uv`, or C++) once layouts stabilize. Don't freeze a format you'll change ten times while optimizing kernels.

---

## 9. Loader, reference implementation, validation

1. **GGUF reader** (engine side, C++): header → KV metadata → tensor infos → aligned data. `mmap`, register as pinned (`hipHostRegister`) or copy in chunks with `hipMemcpyAsync`. Only one model to support → hardcode what can be hardcoded.
2. **Architecture sanity pass** (Python, `uv run gguf-dump`): record layer pattern (which layers are DeltaNet vs full attention), dims, head counts, RoPE params and sections, norm eps, tied embeddings, MTP tensor names, chat template, special tokens.
3. **Reference forward pass** (Python, NumPy or PyTorch CPU):
   - Dequantization ported **exactly** from `ggml-quants.c` (grids, sign tables, scale formulas).
   - Classic silent bugs to watch: RMSNorm variant (e.g. `(1 + w)` zero-centered weights vs `w`), partial rotary dimension, M-RoPE section layout, QK-norm placement, gated attention's sigmoid gate, DeltaNet gate parametrization (α/β activations, L2-norm of q/k), conv1d padding and activation.
4. **Golden tensors**: dump intermediate tensors from llama.cpp (`llama-eval-callback`) for a few fixed prompts; compare layer by layer with the reference, then with each GPU kernel.
5. **Validation tiers:**
   - Dequantization: **bit-exact**.
   - Single kernels: tolerance vs FP32 reference (accumulation order differs).
   - End-to-end: top-1 agreement and **KL divergence** of logits vs llama.cpp over a fixed text corpus; perplexity on a small wikitext slice.
   - Speculative decoding: greedy output with MTP must be **identical** to greedy output without MTP.

---

## 10. GPU kernels

Listed in order of how much runtime they account for.

### 10.1 Quantized GEMV (decode) — ~85–90% of decode time

- One kernel per quant type present in the allocation (templated), **fused dequant + dot**; dequantized weights never touch memory.
- ⚠️ **No int8 *matrix* path on RDNA4.** `v_wmma_i32_16x16x16_iu8` is not in the gfx1200 ISA (it is in gfx1100's); gfx1200 gained the fp16/f32 WMMA tile instead — that is where the measured 46 TFLOPS fp16 comes from. The *vector* int8 dot `v_dot4_i32_i8` **does** exist and computes correctly on gfx1200, but LLVM does not expose `__builtin_amdgcn_sdot4` for this target (`dot1-insts` is not advertised) and it runs at ~14 TOPS (measured, `bench/results/m3-isa-probe.txt`) — well under the fp16 matrix path. So: **dequantize in registers and accumulate with f16/f32 FMA**. Decode is bandwidth-bound (measured 0.88 TFLOPS while reading 701 MiB in 2.90 ms), so the ALU has ~5× headroom and only the bytes per weight matter.
- Activations stay f16 (the milestone-2 kernels already produce f16 activations), so no activation quantization step is needed.
- Scales applied per sub-block in FP32 (a per-32-weight sub-block term); codebook/grid values are dequantized in registers.
- Codebooks + sign tables in LDS (loaded once per workgroup).
- **Small-batch variant N = 1…5 from day one** (templated on N): MTP verification reads the weights once for all N tokens → almost free when memory-bound.
- Fusions: RMSNorm in the prologue, SwiGLU in the epilogue of interleaved gate/up, residual add in the epilogue of output/down projections.
- Enough loads in flight: several outstanding 128-bit loads per lane (unrolling) rather than relying only on occupancy.
- Grid sizing: 32 CUs → make sure the number of workgroups divides evenly (avoid a partial last wave, "tail effect"); use split-K for small matrices so all CUs are busy.

Measured so far (M3.1, `bench/results/m3-gemv-q4k.txt`): fused Q4_K on `output.weight`
= 253.2 GB/s = **79.6 %** of the 318.3 GB/s ceiling (target ≥ 60 %), 28.6× the
f16 dequant + hipBLASLt path.

### 10.2 Gated DeltaNet — decode

Per head, recurrent state `S` (d_k × d_v), per token (Gated Delta rule):

```
S_t = α_t · S_{t-1} · (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
o_t = S_t q_t
```

plus short causal conv1d on q/k/v (keep the conv tail as state), gates/normalization per the reference implementation **[verify]**.

- One fused kernel per layer: conv1d update + gate computation + state update + output (+ output gate/norm).
- State stays **FP32** (errors would accumulate recurrently).
- **Snapshot output mode** for MTP verification: when processing N = k+1 tokens, write the state (and conv tail) **after each position** to a small ring of k+1 snapshot buffers. On acceptance of j tokens, just point to snapshot j. Design this in from the start; retrofitting is painful.

### 10.3 Gated DeltaNet — prefill (chunked)

- Chunkwise-parallel formulation (WY / UT-transform representation), chunks of 64 tokens: intra-chunk work as small matmuls (WMMA), inter-chunk recurrence sequential over chunks.
- Hardest algorithm of the project. Reference implementations: `flash-linear-attention` (fla-org, Triton) and llama.cpp's implementation; read them side by side with the Gated DeltaNet paper.
- Can start with a slow but correct sequential version (prefill speed is priority 3).

### 10.4 Full (gated) attention

- Decode: **flash-decoding** with split-K over the sequence, GQA-aware (one workgroup per KV head serving its query heads), online softmax, dequantize quantized KV on the fly (§13). With int8 K, quantize Q to int8 and use `dot4`.
- Prefill: flash-attention style tiles with WMMA; the tokens of the current ubatch can use their FP16 values directly before being quantized into the cache.
- RoPE / M-RoPE, QK-norm, Hadamard rotation fused into the Q/K projection epilogue (§8.3, §13).

### 10.5 Prefill GEMM (dequant → WMMA)

Two options:

- **FP16 path**: dequantize weight tiles to FP16 in LDS, `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` (or rocWMMA). Numerically closest to reference.
- **INT8 path: dead on this hardware [resolved, M3].** There is no int8 WMMA tile on gfx1200 (`v_wmma_i32_16x16x16_iu8` assembles for gfx1100, not for gfx1200), and the vector int8 dot `v_dot4_i32_i8` runs at ~14 TOPS (measured, `bench/results/m3-isa-probe.txt`), far under the 46 TFLOPS fp16 matrix path. That fully explains the M0 measurement of INT8 hipBLASLt at 0.57× FP16: the library has no int8 matrix hardware to use. **Use the FP16 WMMA path.**
- Double-buffer LDS tiles; overlap global loads, dequant and WMMA; tune tile sizes for 32 CUs. Start slow and correct.

### 10.6 Small ops

RMSNorm, residuals, SwiGLU, gating, RoPE: never standalone kernels in the final engine — fused into prologues/epilogues of the GEMVs/GEMMs.

Measured on the decode step (M4, issue #41):

- **HIP graphs are worth ~1 ms, not the 25 ms a kernel trace suggests.** `omph-graph-probe`
  launches a chain of 1861 tiny dependent kernels — a decode step's length — one by one
  (6.755 ms) and replayed from a captured graph (5.773 ms): a graph recovers 15 % of the
  3.63 us per-kernel cost. The device-side token/position refactor that capture needs is
  therefore not worth doing, and §11's "graphs for the per-token forward pass" is dropped
  for the decode.
- **Fusions into the prologue/epilogue of the GEMV are the right shape** (this section), but
  fusing two or three small kernels at a time measures as noise: three rounds gave 0-1 ms.
- **The sublayer boundary stays a separate launch (#103).** Removing all 128 `add_rms_norm_f16`
  launches of a step (ablation, wrong results) saves at most 0.50 ms; removing the 64
  `swiglu_f16` launches saves nothing.
  - **Folding the add + norm into the output GEMV's last workgroup**: bit-exact, but the step
    is 0.2 ms *slower* than with the separate kernel. On top of that, the epilogue code alone,
    never taken, cost ~1.1 ms through the GEMVs' codegen.
  - **Normalizing in the consumers' prologue**: not pursued. The ceiling is 0.5 ms, minus a
    redundant 5120-element reduction in every workgroup of every consumer.
- **The non-GEMV quarter of the step is dominated by per-launch latency, not work.** The
  GEMVs are 82 % of the step and run near the achievable rate; the rest is ~165 calls of
  ~15 us each. Two of those were bugs, not tuning: a 178 MB f16 conversion of one IQ1_M
  tensor repeated every token (1.83 ms per step, now cached once at load) and a
  shared-memory tree reduction with eight barriers in `rms_norm_f16`, which the decode
  calls with a single row.

### 10.7 Head and sampling

- Final RMSNorm ⊕ `lm_head` GEMV ⊕ per-workgroup partial argmax / top-k in one kernel, final reduction in a tiny second kernel.
- **Sampling on GPU**: only the chosen token id (4 bytes) goes back to the host, never 150–250k logits per token.
- For temperature > 0: top-k/top-p/temperature on GPU; with speculative decoding, the acceptance rule of §12.6.

---

## 11. Runtime and decode loop

- **Preallocate everything** at load (memory arena): weights, KV, states, snapshots, scratch. No allocations in the loop; peak VRAM is deterministic.
- **HIP graphs: measured and dropped for the decode** (see §10.6). A graph is worth ~1 ms of a
  57 ms step, so the device-side position/token refactor the original plan called for is not
  worth its complexity. Revisit only if a future step becomes launch-bound — the probe
  (`omph-graph-probe`) is the instrument that decides it.
- **Two streams**: main compute stream + auxiliary stream for deferred MTP KV fill (§12.3) and uploads (image embeddings, prompt-cache restores).
- Host thread pinned to a P-core; the only per-token host work is reading back the token id and detokenizing/streaming (can be on another thread). **Done for the greedy decode (#102)**: the argmax runs on the device and 8 bytes come back instead of 1 MB of logits (wall 47.99 → 47.84 ms/token). Re-measured in #102 on the 789-kernel step: wall − GPU time is now 0.02 ms, and a graph would save ~0.47 µs × 789 ≈ 0.37 ms. Graphs stay dropped: they would need device-side positions and a fixed attention grid (its split count follows `seq`).
- Advanced (later): **persistent "megakernel"** for decode — one long-running kernel that walks the layers, removing inter-kernel bubbles and tail effects. High complexity; only after everything else.

---

## 12. MTP speculative decoding without hurting prefill

### 12.1 What the MTP head needs

In the Qwen3-Next / Qwen3.5 style (**[verify]** details against the reference implementation for `qwen35`, e.g. HF transformers / vLLM):

1. At position *i*, inputs are the main model's final hidden state `h_i` and the embedding of the **next** token `t_{i+1}`. Both are normalized, concatenated, projected 2d → d by `fc`.
2. The result goes through **one decoder layer with its own attention and its own KV cache**.
3. Shared final norm + shared `lm_head` → draft logits.

To draft well at position *n*, the MTP layer's attention needs KV entries for positions 0…n−1 → some prompt work is **inherent** if full-context drafting is wanted.

What is **not** needed for prompt positions: Q projection, the attention computation itself, output projection, MLP, final norm and `lm_head` (logits only needed at the last position).

### 12.2 Option A — KV-only "shadow" pass fused per ubatch (default)

```
for each prefill ubatch [a, b):
    h[a..b) = main_model_forward(tokens[a..b))             # normal prefill
    # MTP shadow pass, reusing h while it is still in VRAM:
    x   = fc( concat( norm(embed(tokens[a+1..b+1))), norm(h[a..b)) ) )
    x   = mtp_input_norm(x)
    K,V = mtp_kv_proj(x) → rope(K) → (hadamard, quantize) → append to mtp_kv_cache
    # no Q, no attention, no O-proj, no MLP, no lm_head

# last prompt position only:
logits = lm_head(final_norm(h[n-1]))  → sample t_n
full MTP layer at position n-1 with (h[n-1], embed(t_n)) → first draft(s)
```

Details:

- **Off-by-one:** position *i* pairs `h_i` with `t_{i+1}`. The last prompt position has no next token until sampling, so its MTP entry is written during the first draft step.
- **Ubatch boundaries:** the last position of ubatch *k* needs the first token of ubatch *k+1* → index into the full prompt token array (already known).
- **No extra memory:** `h` of the current ubatch is already in VRAM; hidden states for the whole prompt are never stored.
- `fc` + K/V projection fused into one kernel → two small GEMMs per ubatch, ~0.2–0.3% of prefill cost.

### 12.3 Option B — deferred fill on a second stream (zero impact on prefill and TTFT)

- Prefill runs **without** the shadow pass but keeps final-layer `h` for the prompt (n × d × 2 bytes, e.g. ~80 MB for 8k tokens at FP16 — costs VRAM temporarily; can also be staged in host RAM).
- The MTP KV fill runs on the auxiliary stream while the first tokens decode **without** speculation. Decode GEMV is memory-bound and leaves compute units idle; the fill is compute-bound → they overlap well.
- Speculation turns on once the fill has caught up.
- Given the VRAM priority, Option A is the default; B is an experiment.

### 12.4 Option C — windowed MTP context

- Fill MTP KV only for the last **W** prompt positions (e.g. 1–2k). Cost becomes constant regardless of prompt length, and the MTP KV cache becomes tiny (VRAM!).
- Drafting quality probably depends mostly on local context → acceptance may barely drop.
- **Experiment:** plot acceptance rate vs W ∈ {0, 256, 1k, 4k, full} on real workloads (code, chat, prose). W = 0 (MTP attends only to generated tokens) is the extreme and might be surprisingly usable.
- Combines with A (shadow pass only for the last W positions) and with aggressive MTP KV quantization (§13.6).

### 12.5 Drafting

- `k` drafts per step by chaining the MTP layer on its own output **[verify]** exactly which hidden state is fed back (pre- or post-norm) in the reference.
- **Measured and not done (#126):** a frequency-truncated draft head. The top 32k / 64k tokens of a wikitext + C++ corpus cover 94.9 % / 97.5 % of generated tokens, so the ~2 ms a draft saves is about what the lost acceptance costs, and the ranking is corpus-dependent (other languages fall outside it). Adaptive k: k = 3 is best or tied on every prompt measured (`bench/results/m6-speculative.txt`).
- **Truncated-vocab draft head:** score only the top ~32k most frequent tokens = first rows of the frequency-sorted `lm_head` (§8.3). Build the frequency table from a corpus representative of the real workload (for code-heavy use, a code corpus including the languages you actually use), tokenized with the model's tokenizer (`uv run` + `tokenizers`).
- **Adaptive k:** track acceptance over the last few steps and pick k ∈ {1, 2, 3, 4} dynamically.
- Tree drafting (several candidates per position) is attractive for memory-bound verification, but the DeltaNet recurrence needs a separate state per branch → not worth it initially.

### 12.6 Verification and acceptance rule

- Verification = forward pass with N = k+1 tokens using the small-batch GEMV kernels (§10.1), DeltaNet in snapshot mode (§10.2), attention writing KV for all N positions.
- **Greedy:** accept drafts while `draft_j == argmax(p_j)`; the first mismatch is replaced by the verifier's token; plus one bonus token if all are accepted.
- **Sampling (temperature > 0):** standard speculative sampling (Leviathan et al. / Chen et al.): accept draft token *x* with probability `min(1, p(x)/q(x))`; on rejection sample from `normalize(max(0, p − q))`. This preserves the target distribution exactly. With a truncated-vocab draft, `q` is zero outside the subset → still valid (those tokens are never drafted and remain reachable via the residual distribution).

### 12.7 Rollback — the hybrid-architecture problem

- Full-attention KV: rollback = truncate the KV length — true for the Q8/Q4 and f32 caches (rows past the kept length are rewritten before they are read), **not** for the FP16 ring: a verification overwrites the slots of positions `seq − window ..`, still inside the window after a rollback. **Done (#122, #98):** the k+1 slots a verification writes are saved before it and those of the rejected positions restored.
- **DeltaNet recurrent state cannot be "un-applied".** Options considered:
  - Snapshot per position: k+1 copies of all recurrent states (151 MB each with the real dims), i.e. +604 MB at k = 3 — rejected for VRAM.
  - Checkpoint + replay through the DeltaNet layers: re-runs the projections.
  - **Chosen (#104, done in #122): two state buffers + rank-1 replay.** The verification reads A and writes B; every token records its rank-1 factors (decay, normalized k, d: ~33 KB per token per layer) and the first one the conv input history. All accepted → swap A/B; j accepted → a replay kernel applies `S ← decay·S; S ← S + k dᵀ` to A for the j tokens with `gdn_step`'s float operations in its order, **bit-exact** (checked on every layer by `OMPH_SPEC_CHECK=1`), and the conv tail is rebuilt from the history. +151 MB, one state read + write only on a partial acceptance.
- MTP layer KV: after verification, write entries for accepted positions using the **true** `h` from the verification pass (not the draft-time hidden states).
- Image positions: the "next token embedding" input of the MTP head at image positions is not a vocabulary embedding → check how the reference handles it, or simply skip image positions in the MTP shadow pass if acceptance does not suffer.

---

## 13. KV cache quantization

### 13.1 Scope

- KV cache exists **only in full-attention layers** (hybrid model) + the MTP layer.
- Per token: `2 (K,V) × n_attn_layers × n_kv_heads × head_dim × bytes_per_element`.

Illustrative (16 attention layers, 4 KV heads, head_dim 256 — **placeholders**):

| Format | Bytes/token | 32k tokens |
|---|---|---|
| FP16 | 64 KB | ~2.0 GB |
| Q8 (8.5 bpw) | 34 KB | ~1.1 GB |
| K Q8 + V Q4 | 25 KB | ~0.8 GB |
| Q4 (4.5 bpw) | 18 KB | ~0.55 GB |

At long contexts attention decode is bound by KV reads → smaller KV is also **faster**.

### 13.2 K is more sensitive than V

Errors in K change the pre-softmax scores (i.e. *where* the model looks); errors in V are averaged by the weighted sum. K also tends to have outlier channels that break block quantization. Default: **K Q8, V Q4**.

### 13.3 Hadamard rotation (makes K at Q4 viable)

- After RoPE (and QK-norm), apply a fast Walsh–Hadamard transform along head_dim to **both Q and K**. H is orthogonal → `(HQ)·(HK) = Q·K`, scores unchanged; outliers get spread across channels → much better quantization.
- Cost O(d log d) per head, fused into the kernel that writes the cache (head_dim is a power of two **[verify]**).
- V can be rotated too, applying Hᵀ to the attention output at runtime. **Cannot** be folded into `o_proj` offline — that would require re-quantizing its weights (violates the lossless rule).

### 13.4 FP16 windows

- **Recent window:** last N tokens (e.g. 64–128) kept in FP16, quantized in blocks as the window fills. **Measured (M5, #61):** 128 tokens take the KL from 0.00174 to 0.00090 nats at 16k (and remove a 0.27-nat outlier) for 8.4 MB; on by default in the Q8/Q4 mode. Better quality where it matters most, and allows **per-channel K quantization over groups of tokens** (KIVI-style), more accurate than per-token for K.
- **Initial tokens:** first few positions in FP16 (they often receive a lot of attention; gated attention should reduce this "sink" effect, but keeping a few in FP16 is almost free).

### 13.5 Mixed precision per layer

- With few attention layers, measure sensitivity **one layer at a time**: quantize a single layer to Q4, measure KL increase vs FP16 KV at long context. Assign Q8/Q4 per layer (K and V separately) under a VRAM budget — RCO spirit in miniature.

### 13.6 MTP layer KV can be aggressive

Errors in the MTP path only lower acceptance, never correctness (verification guarantees exactness) → Q4 or lower + windowing (§12.4). Tune by measuring acceptance.

### 13.7 DeltaNet states are **not** quantized

Recurrently read and rewritten every token → errors accumulate. Small, fixed-size → keep FP32 (BF16 only after measuring), snapshots included.

### 13.8 Layout, kernels, validation

- Struct-of-arrays: quantized values contiguous per (layer, KV head, token block), scales in a separate stream, aligned for 128-bit loads.
- Flash-decoding dequantizes on the fly; int8 K × int8 Q with `dot4`.
- Validation at **long context** (16–32k; short prompts hide KV errors): KL divergence vs FP16 KV, needle-in-a-haystack style retrieval tests. Compare against llama.cpp's `-ctk/-ctv` options as a baseline. Our margin over llama.cpp: rotation, FP16 windows, per-layer choice.

---

## 14. Vision encoder on CPU

### 14.1 Why

- `mmproj` is ~0.9 GB BF16 (~450 M params) → **0.9 GB VRAM saved**, plus the ViT activations (attention over thousands of patches) which can be large.
- The encoder runs once per image; output is small: e.g. 1024 image tokens × 5120 × 2 B ≈ 10 MB → negligible over PCIe 5.0.
- Latency is irrelevant for this project.

### 14.2 Flow

1. CPU: preprocessing (resize, patching, normalization) → ViT → projector → embeddings in the LLM hidden dimension.
2. Upload embeddings to the GPU; they replace the image placeholder tokens in the sequence.
3. GPU prefill as usual (image tokens count as prompt tokens: ~1024 tokens ≈ ~1.4 s at 750 t/s).

### 14.3 On the i5-13400F

- No AVX-512, no native BF16 → convert weights BF16 → FP32 at load (~1.8 GB RAM).
- AVX2/FMA GEMMs (and AVX-VNNI if an int8 path is ever wanted — avoid quantizing the vision encoder: vision encoders are sensitive and RAM is not a constraint).
- Pin encoder threads to the **6 P-cores**, leave one free for the GPU host thread (prevents jitter in kernel launches during concurrent decode). E-cores optional.
- Rough cost: ~2 × params × patches + attention ≈ ~4 TFLOP for a ~1 MP image with 16×16 patches → seconds to tens of seconds on this CPU. Acceptable.

### 14.4 Resolution

- The main lever for both encoder time and GPU prefill tokens: halving the side → ~4× fewer patches.
- For UI screenshots (e.g. forms to be converted to code/DFM), small text and dense controls need resolution → measure quality at several resolutions on the real use case before fixing a limit.

### 14.5 Implementation

- **Pragmatic:** llama.cpp `libmtmd` / `clip` built **CPU-only**, used only for encoding; take its output embeddings and feed them to the engine. It already implements the correct preprocessing for this architecture (the part most likely to go silently wrong) and doubles as numerical reference. **[verify]** current `mtmd.h` API for encode + output-embedding access.
- **Custom (later, optional):** own ViT in C++ with AOCL-BLIS / OpenBLAS / oneDNN GEMMs in FP32.

### 14.6 Engine-side details

- **M-RoPE positions** for image tokens (temporal/height/width sections) in the full-attention layers' RoPE kernel **[verify]** sections in metadata. DeltaNet layers have no positional encoding.
- **Overlap:** the GPU can prefill the text *before* the image while the CPU encodes; text *after* the image waits (its positions depend on the number of image tokens, known right after preprocessing).
- **Embedding cache** keyed by image hash (multi-turn conversations don't re-encode).
- MTP at image positions: see §12.7.

---

## 15. VRAM budget

Illustrative, with IQ3_S + MTP and 32k context (**all rows [verify]** with real dims and measurements):

| Item | ~Size |
|---|---|
| Weights IQ3_S-mtp | 12.1 GB |
| − `token_embd` moved to host RAM (if untied) | −0.5 … −1.0 GB |
| KV cache 32k, K Q8 / V Q4 (placeholder dims) | ~0.8 GB |
| MTP layer KV (Q4, windowed) | < 0.05 GB |
| DeltaNet states + (k+1) snapshots | ~0.1–0.5 GB |
| Prefill scratch / activations (ubatch 512) | ~0.3–0.6 GB |
| HIP runtime / context | ~0.2–0.4 GB |
| Desktop (no iGPU!) | ~0.3–1.0 GB |
| Vision | **0** (CPU) |
| **Total** | **~13.3–15.4 GB** |

Levers if tight:

- **IQ3_XXS** instead of IQ3_S: −1.7 GB (quality still close to base per the model card).
- Smaller prefill ubatch (256): less scratch, slightly slower prefill.
- KV fully Q4 with Hadamard + windows; shorter max context.
- Headless benchmarking / lightweight desktop session.
- Prompt-prefix cache kept in **host RAM** instead of VRAM (§16.6).

---

## 16. Performance "breadcrumbs"

Small gains, a few percent each at most, but they add up. Rough expected impact in brackets.

### 16.1 System / hardware

- **VRAM overclock** (LACT, requires `amdgpu.ppfeaturemask=0xffffffff`): decode scales ~linearly with memory clock → often the single biggest "free" gain [+3–8% decode]. GDDR6 errors can be *silent* (error correction/retry masks them as lower performance, or worse, corrupt values) → validate with the KL test after any OC, and benchmark to confirm speed actually went up. **[verify]** RDNA4 memory OC support in LACT.
- **Undervolt / raise power limit** so the core keeps boosting during prefill (compute-bound) [prefill +x%].
- **Power profile COMPUTE** (`pp_power_profile_mode`) and/or `power_dpm_force_performance_level=high` during benchmarks; check that memory clock stays at max in decode.
- **Thermals:** fan curve so the card never throttles during long runs.
- **Headless / minimal desktop** while running: frees VRAM *and* removes compositor contention.
- Host thread pinned to a P-core; vision threads kept off that core.

### 16.2 Memory access (decode)

- Struct-of-arrays, 128-bit aligned loads, fully coalesced (§8.3) [several %].
- Several independent 128-bit loads in flight per lane (unrolling / software pipelining) [latency hiding].
- **Non-temporal loads for weights** (`__builtin_nontemporal_load`, streaming cache policy): weights are read once per token and should not evict KV/state/activations from caches **[verify]** effect on RDNA4 [small].
- Keep activations in FP16/int8, accumulate in FP32.
- Codebook tables in LDS, loaded once per workgroup.
- Put all tiny per-token state (norm weights, gates, biases) contiguous so they come in few cache lines.

### 16.3 Kernel launch / scheduling

- **HIP graphs** with device-side position counters (§11) [removes a few % of launch overhead].
- **Fusion everywhere** (norm in prologues, SwiGLU/residual in epilogues, RoPE/Hadamard/KV-write in QKV epilogue) → fewer kernels, fewer intermediate writes.
- **Tail effects**: workgroup counts that divide evenly across 32 CUs; split-K for small matrices (DeltaNet projections, MTP `fc`).
- **Per-shape autotuning table**: the model is fixed, so autotune tile/unroll/workgroup size per (tensor shape, quant type) offline once, and store the best config (can even live in the custom file format).
- Check ISA for register spills and VGPR usage (`-Rpass-analysis=kernel-resource-usage`, `--save-temps`); `__launch_bounds__` where useful.
- Later: persistent decode megakernel (no inter-kernel bubbles).

### 16.4 Head, sampling, host

- Sampling on GPU, only 4 bytes back per token [avoids ~1 MB logits copy + sync per token].
- Fused final norm + `lm_head` + partial argmax.
- Detokenization and streaming on a separate host thread.
- Tokenizer work for the next request overlapped with GPU work.

### 16.5 MTP-specific

- Truncated-vocab drafts via frequency-sorted `lm_head` prefix [big: −5–8× draft `lm_head` cost].
- Adaptive k based on recent acceptance.
- Small-batch GEMV kernels specialized for N = 2…5 (not generic GEMM).
- Snapshot-based DeltaNet rollback (no replay pass).
- Aggressive MTP KV quantization + windowing (VRAM + speed).
- Frequency table built from the *actual* workload's text distribution.

### 16.6 VRAM-specific

- `token_embd` in host pinned RAM (if untied). **Done (#92)**: the dequant kernel reads the row over PCIe; −388 MiB, decode step unchanged.
- Vision on CPU.
- MTP section loaded only when used. **Done for now (#92)**: `blk.<n_layer>.*` is not uploaded; M6 gates it behind the MTP option.
- DeltaNet state allocated for the 48 recurrent layers only (#92).
- **Prompt-prefix cache in host RAM**: store quantized KV + DeltaNet state checkpoint at the end of a system prompt / common prefix; restore with an upload (PCIe 5.0) instead of re-running prefill.
- Preallocated arena → no fragmentation, predictable peak.
- Extreme option (slow): offload old KV blocks to host RAM for very long contexts.

### 16.7 Prefill-specific

- INT8 WMMA path (IQ values fit int8) vs FP16 path — measure speed vs KL.
- Double-buffered LDS tiles, overlapping load / dequant / WMMA.
- Tune ubatch size (512 vs 1024 vs 2048) against VRAM scratch.
- KV-only MTP shadow pass instead of full MTP layer (§12.2).

---

## 17. Measurement methodology

- Fixed conditions: headless (or fixed minimal desktop), same power profile, card warmed up, fixed fan curve.
- Warm-up runs discarded; report **median of ≥5 runs**.
- Decode t/s at context depths **0 / 4k / 16k / 32k**; prefill t/s at **512 / 4k** prompt lengths; **peak VRAM**; time to first token.
- Report efficiency, not just speed: decode as % of *measured* achievable bandwidth; prefill as % of *measured* hipBLASLt FP16 GEMM throughput.
- MTP: acceptance rate per workload type (code / chat / prose), tokens per step, effective t/s.
- Quality: KL divergence vs llama.cpp reference (and vs FP16 KV for KV experiments), top-1 agreement, perplexity on a fixed slice; greedy-output identity with/without MTP.
- Keep results as CSV in `bench/` with git commit hash of the engine.

---

## 18. Milestones

| # | Milestone | Exit criterion |
|---|---|---|
| 0 | Baselines | llama.cpp HIP/Vulkan numbers (with/without MTP, KV types), measured bandwidth and GEMM ceilings, kernel traces of the MTP prefill drop |
| 1 | Inspection + CPU reference | `gguf-dump` facts recorded; NumPy/PyTorch-CPU reference matches llama.cpp logits (KL ≈ 0) |
| 2 | Naive GPU path | One kernel per op, dequant to FP16, correct tokens, greedy output = reference |
| 3 | Custom layout + fused int8-dot GEMV | Bit-exact load-time repacking; decode ≥ 60% of measured bandwidth |
| 4 | Remove the non-GEMV waste | Decode ≥ 75% of measured bandwidth (55.7 ms/token over 12.38 GiB). Graphs measured at ~2 % and dropped, see §10.6 |
| 5 | Quantized KV | K Q8/V Q4 with Hadamard + windows; KL within budget at 32k |
| 6 | MTP | KV-only shadow prefill (prefill within ~2% of non-MTP), snapshots, truncated-vocab drafts, adaptive k; greedy identity |
| 7 | Vision on CPU | libmtmd CPU encoding, M-RoPE, embedding cache |
| 8 | Fast prefill | Dequant→WMMA GEMM, chunked DeltaNet; target ~900+ t/s |
| 9 | Offline converter + final format | Layouts frozen, bit-exact verification in the converter |
| 10 | Polish | C ABI, OpenAI-compatible server, prompt-prefix cache in host RAM, breadcrumbs |

Status: **M0-M5 are complete; M6 is functionally complete (#121).** Greedy speculative decoding with the model's MTP head (`--draft-mtp 3`): output identical to plain greedy; 18.9 ms/token on wikitext, 19.4 on prose, 24-25 on code / repetitive text, against 45.5 ms plain (1.8-2.4x). Rollback is bit-exact (A/B delta-net states + rank-1 replay, FP16-ring restore, #122), the MTP KV fill costs +1.1-1.5 % of a `--gemv` prefill, verifications run on NT-token GEMVs (#126). Not done: truncated-vocab drafts (measured, not worth it), adaptive k (k = 3 always best or tied), sampling (temperature > 0). Measurements: `bench/results/m6-speculative.txt`.

Status: **M0-M5 are complete.** M5 (quantized KV, issues #43, #58-#61): K Q8 + V Q4
with the Hadamard rotation and a 128-token FP16 window, 4.92x less KV VRAM (4.29 GB ->
0.87 GB at 32k). Validated against the exact f32 KV (kept in host RAM for the purpose,
`OMPH_KV_HOST=1`) up to 32k: KL 0.00084 / 0.00090 / 0.00071 nats at 8k / 16k / 32k,
top-1 >= 98.6 %, under llama.cpp's own q8_0/q4_0 KV (0.00162 / 0.00149 / 0.00135) at
every length — the budget chosen in #58 (`bench/results/m5-kv-kl-long.txt`). The
GQA-grouped flash attention (#59) makes the decode nearly flat in context: 58.4 ms at
512, 61.7 ms at 8k, 65.0 ms at 16k with the quantized KV (was 99.0 ms at 8k).
Since #69 the Q8/Q4 KV is the engine's default (`OMPH_KV_F32=1` for the f32 reference).
K at Q4 (#81, `OMPH_KV_K4=1`, -0.32 GB at 32k): KL 0.0017 / 0.0035 / 0.0021 at 8k / 16k /
32k — about llama.cpp's q4_0/q4_0 (0.0025 / 0.0028 / 0.0026) but over the q8_0/q4_0
budget, so K stays Q8 by default; a per-layer K choice (§13.5) is the open follow-up.

Decode after M5 (#63, #66; `bench/results/decode-step-20261001.txt`): **55.2 ms/token** at
a 10-token prompt (58.0 on main measured the same day), 54.8 ms at 512 and 58.2 ms at 8k
with the quantized KV. Measured by ablation (`OMPH_SKIP_GEMV_TYPE`), the GEMVs run at
~78 % of the bandwidth in the model and take ~47 ms; everything else is 3.9 ms of kernel
time. `rocprofv3` per-kernel times inflate some GEMVs by up to 40 % and the `OMPH_TIMING`
phase totals were wrong before #64 — use the ablation. Per-kernel GEMV tuning measured
at diminishing returns (#63); the next decode lever is MTP (M6). The gap to llama.cpp
(48.5 ms) is GEMV efficiency.
Before that, M4's status: the decode was at
73.3 % of the ceiling (57 ms/token) when M3 closed, and the two defects above put it at
~75.6 % (55.2 ms/token) — at the milestone's criterion. What is left of the criterion is
whatever the remaining ~15 us-per-launch kernels cost.

- **M3 (custom layout + fused GEMV).** Every quant type in the allocation except
  IQ1_M (0.02 GiB) has a fused dequant+dot kernel, each with a repacked layout
  verified byte-identical against the source block stream. Decode: **59 ms/token**
  — 14.1x the naive f16 path — reading 11.19 GiB of weights, i.e.
  **~204 GB/s = 64 % of the measured 318.3 GB/s ceiling** (the GEMV launches alone:
  250 GB/s = 79 %). The milestone criterion was >= 60 %. Greedy output is still
  identical to the NumPy reference on the optimized path (prefill logits
  rel 4.9e-04).
- The prefill is *not* part of M3 and is untouched: ~2 s per 10-token pass
  (~7 t/s), to be rebuilt in M8 (see §4.2 for where its time goes). The naive GPU path (`omph-run`) reproduces the NumPy reference: prefill logits rel 3.7e-04, greedy decode identical over 3 generated tokens with persistent KV / conv / delta-net state (see the M2 section of AGENTS.md for the commands). M4 (graphs + fusions) is next.

---

## 19. Open questions / verification checklist

- [x] Which quant file produced the 19–20 t/s decode and 750 t/s prefill baselines? → **M0: IQ3_S-mtp** reproduced (tg128 20.6 t/s f16 @d0); the 750 t/s prefill figure did not reproduce (622.7 t/s pp512).
- [x] llama.cpp decode t/s **with** the `-mtp` build → back out acceptance rate. → **M0: acceptance 77–86%** (`draft-mtp` n=2), decode ~20 → ~35 t/s.
- [x] Root cause of the 750 → 500 t/s MTP prefill drop (kernel trace). → **M0: not reproduced**; ~3% cost, kernels unchanged (§4.4).
- [x] Real model dims: hidden size, layer count, DeltaNet vs attention layer pattern, head counts, head dims, vocab size. → measured, see §3.
- [x] RoPE: partial rotary factor, M-RoPE sections, theta. → `freq_base` 1e7, rotary dim 64 (partial), sections [11, 11, 10, 0].
- [x] Norm variants (zero-centered RMSNorm?), QK-norm, gated attention gate, DeltaNet gate/β parametrization, conv1d kernel size. → **M1/M2**: plain RMSNorm (eps 1e-6) for every norm; QK-norm per head (head_dim 256 for the attention layers, 128 for the delta net); the attention output gate comes fused in `attn_q` (2·head_dim per head) and is applied as `sigmoid(g)`; DeltaNet gate = `softplus(alpha + dt_bias)·ssm_a`, β = `sigmoid(beta)`; conv kernel 4 with `silu`. All confirmed against the dump.
- [x] Tied embeddings? (decides whether `token_embd` can go to host RAM separately). → **untied** (separate `output.weight`, Q4_K).
- [x] Allocation file: all quant types present; do `gate`/`up` and Q/K/V share types? `lm_head` type? → see §3: mix listed; `gate`/`up` differ in 40/65 layers; DeltaNet `attn_qkv` is fused, full-attention layers have separate Q/K/V; `lm_head` = Q4_K.
- [x] MTP: exact inputs (`h` pre- or post-norm), chaining for k > 1. → **#121/#124** (llama.cpp `graph_mtp` + draft-mtp): `h` is the target's hidden **after `output_norm`**; `x = eh_proj([enorm(embed(t)); hnorm(h)])` (embedding first), then a full-attention block with its own KV, `shared_head_norm` and the shared `output.weight`; position p pairs (h_{p-1}, t_p), h = 0 at p = 0; a chained draft feeds back the block's own `shared_head_norm` output. GPU vs NumPy (fed llama.cpp's `h_nextn`): rel 4.5e-2 on both drafts (`check_gpu_mtp.py`). Image positions: open, M7.
- [x] RDNA4 specs: FP16/INT8 matrix peak, LDS size, cache sizes; ROCm version on CachyOS. → ROCm 7.2.4; measured fp16 ~46 TFLOPS / INT8 ~26 TOPS at M=512; LDS 64 KB; vendor peak still [verify].
- [x] llama.cpp `new_state` buffer layout vs the engine kernels → **settled in M2**: the engine keeps the delta-net state as `(n_vh, state_size, state_size)` with the ggml k-head tiling (`h % n_kh`); the per-token outputs match the reference, so the fused kernel's internal (dumped) buffer layout is irrelevant for us.
- [ ] `mtmd.h` API for extracting image embeddings. → M7.
- [ ] RDNA4 memory OC support in LACT.

---

## 20. References

**Model and quantization**

- Model card: `https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`
- GSQ: *Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling*, arXiv:2604.18556 — code: `IST-DASLab/GSQ`
- RCO: *Model Compression with Exact Budget Constraints via Riemannian Manifolds*, arXiv:2605.00649 — code: `IST-DASLab/RCO`

**Architecture**

- Gated Delta Networks: *Improving Mamba2 with Delta Rule* (Yang, Kautz, Hatamizadeh), ICLR 2025
- `flash-linear-attention` (fla-org) — reference kernels for (Gated) DeltaNet
- Qwen3-Next / Qwen3.5 technical material and HF transformers / vLLM implementations of `qwen35`

**Speculative decoding / MTP**

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023)
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling* (2023)
- DeepSeek-V3 technical report (MTP modules)
- EAGLE-3; FR-Spec (frequency-ranked vocabulary for draft heads)

**KV cache quantization**

- KIVI (per-channel K, per-token V quantization)
- QuaRot / SpinQuant (Hadamard / rotation-based outlier removal)
- KVQuant

**Kernels**

- Flash-Decoding (Dao et al., 2023)
- llama.cpp: `ggml/src/ggml-common.h` (block structs), `ggml-quants.c` (dequantization), `ggml-cuda` (HIP build: `mmvq`, `mmq`, flash attention), `tools/mtmd` (vision)
- AMD: RDNA4 ISA reference, rocWMMA, hipBLASLt, rocprofiler-sdk documentation
