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

Model paths used below: `models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (local, git-ignored) and the `.omph` file `omph-convert` writes from it (#178). The engine binaries that run the model (`omph-run`, `omph-generate`, `omph-server`, `omph-capi-demo`, the C ABI) load only `<model.omph>`; the validation tools and scripts take the GGUF (they derive the `.omph` next to it, `tools/omph_model.py`).

```sh
# build
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build -j

# build with vision (#160): llama.cpp's mtmd from a CPU-only llama.cpp build (out of tree;
# every GPU backend off: the encoder must never touch VRAM), then point the engine at it
cmake -S <llama.cpp> -B /var/tmp/omphalos-llama-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=OFF -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
cmake --build /var/tmp/omphalos-llama-cpu -j --target mtmd llama
cmake -S engine -B engine/build -DOMPH_LLAMA_DIR=<llama.cpp> -DOMPH_LLAMA_LIB=/var/tmp/omphalos-llama-cpu/bin

# convert the GGUF once (#178): weights repacked into the kernels' layouts and verified bit for bit,
# the metadata (tokenizer, chat template) copied, the source SHA-256 recorded; ~1 min, 11.3 GiB.
# The engine refuses an .omph of another format version (IQ3_S tiles: 2; IQ3_XXS tiles: 3;
# IQ4_XS and Q4_K tiles: 4, #244; IQ2_S, Q2_K, IQ2_XS, IQ2_XXS tiles: 5, #253;
# Q6_K tiles: 6, #258): reconvert (the DFlash2 drafter .omph too)
engine/build/omph-convert <model.gguf> [<model.omph>]

# tokenizer (#148): text on stdin -> ids, or --decode ids -> text (reads a .gguf or an .omph)
engine/build/omph-tokenize <model.gguf> [--no-parse-special] < prompt.txt > tokens.txt
# chat template (#150): a JSON request {messages, tools?, add_generation_prompt?,
# enable_thinking?, ...} on stdin -> the prompt (--chat) or its ids (--chat-ids)
engine/build/omph-tokenize <model.gguf> --chat-ids < request.json > tokens.txt

# generation (#152): text, a chat request (--chat) or ids (--prompt-ids) in, streamed text out;
# greedy uses MTP speculation, --temp/--top-k/--top-p/--min-p/--seed sample, --then FILE a next turn,
# --repeat N the same request again (resumes from a checkpoint), --cache-mib N (0: no checkpoints),
# --mmproj FILE --image FILE (one per image item / <|image_pad|>), --force IDS --logits-out FILE
# (teacher-forced logits, validation)
engine/build/omph-generate <model.omph> --chat --max 256 < request.json

# the C ABI (include/omphalos.h, #154) from plain C: load, chat, tokenize, generate; with an mmproj
# and an image, one more turn about it (images in omph_generate_params, #180)
engine/build/omph-capi-demo <model.omph> [<mmproj.gguf> <image>]

# OpenAI-compatible server (#156): /v1/chat/completions, /v1/completions, /v1/models, /health;
# streamed or not, reasoning_content / tool_calls; one request at a time on 127.0.0.1:8080.
# Logs to stderr a progress line every 3 s of decoding (tokens, t/s, drafts accepted) and a summary
# per request (#231). --dflash FILE: draft with the DFlash2 drafter (#245) instead of the MTP block
# (convert z-lab/Qwen3.8-27B-DFlash2-GGUF's Q4_K_M with omph-convert; also for omph-generate, omph-run).
# Options: --host --port --ctx --cache-ram MIB (sequence checkpoints in host RAM, #158; default
# 2048) --kv-ram MIB (whole conversations in host RAM: a prompt that leaves the cached one saves
# it, one that continues a saved one restores it, #179; default 8192) --mmproj FILE (images as base64 data: URLs, #160) --alias --api-key --cors ORIGIN, request defaults --temp
# --top-k --top-p --min-p --max-tokens (default greedy: speculative MTP decoding)
engine/build/omph-server <model.omph> [--port 8080]

# full forward pass: prefill + greedy decode (options in the table below)
engine/build/omph-run <model.omph> models/golden/cpu/tokens.txt <out-logits.f32> \
    --gemv --generate 3 --gen-out /tmp/gen.txt [--trace-dir DIR] [--tokens N]

# the decode / verification attention alone on a synthetic cache (#169): kernel time and bandwidth
# of the WMMA and scalar kernels per length and token count, plus the row identity check; seconds.
# --k4: the K4/V4 cache (OMPH_KV_K4)
engine/build/omph-attn-bench [--seq 4096,32768,100000] [--tokens 1,4] [--ctx N] [--chunk K] [--k4]

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
uv run python check_gpu_mtp.py                    # MTP draft head vs NumPy fed llama.cpp's h
uv run python check_gpu_decode.py [--gemv]        # greedy decode vs the NumPy reference
                                                  # (cached in models/golden/cpu; --refresh)
uv run python check_tokenizer.py <model> <llama.cpp>/bin/llama-tokenize [--fuzz N]
                                                  # tokenizer vs llama.cpp, token for token
uv run python check_chat_template.py <model>      # chat template vs jinja2, byte for byte
uv run python check_conversations.py [--spec mtp,dflash]  # #179: back to a saved conversation:
                                                  # restored, not prefilled, same answer
uv run python check_server.py [--url URL]         # omph-server end to end with the openai
                                                  # client (starts ../engine/build/omph-server)
    [--image <llama.cpp>/tools/mtmd/test-1.jpeg]  # ... and images (the server gets --mmproj)
uv run python check_vision.py --image <llama.cpp>/tools/mtmd/test-1.jpeg
                                                  # images vs llama.cpp, teacher-forced logits
                                                  # (needs tools/native/dump_mtmd_logits)
uv run python compare_logits.py ref.f32 test.f32  # KL + top-1 agreement over every
                                                  # position (e.g. OMPH_KV_F32=1 vs default;
                                                  # OMPH_GDN_EXACT=1 vs default for the WY prefill,
                                                  # #240, which check_gpu_gdn's 10 tokens never reach)

# long-context KV validation (exact f32 KV kept in host RAM as the reference)
OMPH_KV_HOST=1 engine/build/omph-run <model> <tokens.txt> ref.f32 --logits-tail 512
engine/build/omph-run <model> <tokens.txt> q8.f32 --logits-tail 512   # default: Q8/Q4 KV
bench/m5_llama_kv_kl.sh <llama.cpp-bin-dir> <ctx>   # llama.cpp's own KV-quant KL (budget)
# the same for decode steps: the f32-KV run's greedy tokens forced into the others (#138)
OMPH_KV_F32=1 engine/build/omph-run <model> <tokens.txt> /tmp/p.f32 --last-logits --gemv \
    --generate 64 --gen-out ref.txt --gen-logits ref.f32
engine/build/omph-run <model> <tokens.txt> /tmp/p.f32 --last-logits --gemv --generate 64 \
    --gen-out q.txt --gen-logits q.f32 --gen-force ref.txt   # then compare_logits.py ref.f32 q.f32

# CPU-only tests (no GPU, no ROCm; what CI runs, .github/workflows/ci.yml)
cmake -S engine/tests -B engine/build-tests && cmake --build engine/build-tests -j
ctest --test-dir engine/build-tests --output-on-failure   # repack round trip, GGUF parser, tokenizer, JSON + chat
cd tools && uv run ruff check . && uv run python -m pytest tests -q   # decoders vs gguf-py
uv run python check_doc_math.py ../docs/*.md ../PLAN.md        # math GitHub would mangle

# regenerating the golden dump (CPU backend, needs a llama.cpp build)
tools/native/build.sh <llama.cpp-dir> && tools/native/dump_tensors ...   # also builds dump_mtmd_logits
```

### `omph-run` options and environment switches

| option | effect |
|---|---|
| `--gemv` | decode with the fused GEMVs on repacked weights: the fast path. Only effective with `--generate`; a prefill-only run stays on the f16 + GEMM path |
| `--generate N --gen-out FILE` | greedy-decode N tokens after the prompt, ids to FILE |
| `--gen-logits FILE` / `--gen-force FILE` | write each decode step's logits (N - 1 rows) / decode the tokens of FILE instead of the greedy ones: decode-step comparisons between KV settings or kernels (#138) |
| `--last-logits` / `--logits-tail N` | write only the last row / the last N rows of logits (long prompts) |
| `--tokens N` | use only the first N prompt tokens |
| `--trace-dir DIR` | dump every layer's output (one-chunk prompts only) |
| `--dflash FILE` | speculative greedy decode with the DFlash2 drafter .omph (#245): 7 drafts per step, the MTP block not loaded; output identical to plain greedy |
| `--draft-mtp K` | speculative greedy decode with K MTP drafts per step (#124, #126); output identical to plain greedy, bit for bit (#161). K = 3 is the measured best (1.8-2.4x the plain decode); costs +0.5 GB VRAM (MTP block + alternate delta-net state) |
| `--mtp` / `--mtp-out FILE` | load the MTP block (needs `--gemv --generate`) / write two chained drafts' logits after the prompt (validation) |
| `--draft-oracle FILE [--draft-k K] [--draft-corrupt N]` | speculative greedy decode with drafts from a token file (a plain greedy run's `--gen-out`), every N-th draft corrupted: validates the verification + rollback (#122); the output must equal the plain greedy run |

The `OMPH_*` switches are parsed once, in `engine/src/runtime/options.{hh,cc}` (the authoritative list). Defaults are what the engine runs; ablations give wrong results with valid timings.

| variable | kind | effect |
|---|---|---|
| `OMPH_TIMING` | diagnostics | VRAM after load, per-step `step gpu` / `step wall` |
| `OMPH_PHASES` | diagnostics | per-phase GPU totals; its ~460 events per step add ~2.6 ms, so never measure the step with it (#100) |
| `OMPH_TEST_BAD_SIDE=N` | diagnostics | swap the side stream the calibration rejected back in after N decode steps: exercises the #144 watchdog |
| `OMPH_TEST_MROPE=swap\|flat` | ablation | image positions with h and w exchanged, or 1D: must score worse in `check_vision.py` (#160) |
| `OMPH_TEST_Q8ACT=B` | ablation | every prefill GEMM's input quantized to int8 in blocks of B along k (one amax / 127 scale each, llama.cpp's q8_1 at B = 32) and back: an int8 GEMM's numerics, for its KL (#213) |
| `OMPH_SPEC_CHECK` | diagnostics | check every speculative rollback: the replay bit-exact against `gdn_step`, the FP16 ring restored (slow) |
| `OMPH_TRACE_ALLOC` / `OMPH_TRACE_F16` / `OMPH_TRACE_STAGE` | diagnostics | f16-scratch allocations / matmuls falling back to the f16 path / f16 staging and cache hits |
| `OMPH_KV_F32` | KV | exact f32 cache in VRAM (reference) |
| `OMPH_KV_HOST` | KV | exact f32 cache in pinned host RAM (long-context reference) |
| `OMPH_KV_K4` | KV | K in V's Q4 format too (#81, experiment) |
| `OMPH_KV_K4_LAYERS=i,j,...\|none` | KV | K4 on those attention layers (0-15), the rest Q8. Default: 2, 4, 5, 6, 7, 9, 11, 15, the least sensitive ones (#175, PLAN.md §13.5, `bench/results/k4-per-layer-175.txt`); `none`: K8 everywhere (the pre-#175 cache) |
| `OMPH_KV_WINDOW=N` | KV | every query reads its last N keys exactly from an FP16 ring of N + 15 slots (default 128, 0 = off; #161) |
| `OMPH_OVERLAP` | A/B | a side stream for sibling GEMVs (#71); off by default since #189 (the persistent-warp GEMVs fill the GPU alone: 44.2 vs 44.9 ms per step, and a side stream the calibration keeps could cost ~9 ms). With it on, the runner times the overlap at load and drops a side stream that loses to running in order (#132), and re-times it when single-token steps turn 1.35x slower during the run (#144; `OMPH_TIMING` prints both) |
| `OMPH_NO_B4` | A/B | no NT = 2..4-token GEMVs (verifications, short `--gemv` prefills): one launch per token |
| `OMPH_NO_GROUP` | A/B | the sibling GEMVs of one input (gate + up, qkv + gate, q + k + v) as separate launches instead of one grouped launch (#214; 1..16 tokens since #255) |
| `OMPH_DRAFT_VOCAB=N` | decode | MTP drafts take their argmax, and DFlash2 its top-16 candidates (#245), over the first N token ids (default 98304; 0 = the whole head) while the recent text stays inside them, the whole head otherwise (#217): output unchanged, ~3-7 % faster speculative decoding in English and code |
| `OMPH_NGRAM=0` | decode | no n-gram (prompt lookup) drafts: by default, when the 4..8-gram ending at the next token occurred before, the tokens that followed it (up to 15) are the step's drafts instead of MTP / DFlash2's (#199): output unchanged, 1.3-2.6x faster on edits and quotes of the context |
| `OMPH_NGRAM_MIN=N` | decode | the fewest n-gram drafts that replace the model drafter's (default 4) |
| `OMPH_DFLASH_KEEP=P` | decode | DFlash2 drafts position n only while the measured chance that drafts 1..n are all kept is >= P (default 0.12; 0: always 7; every 8th step drafts all, #245) |
| `OMPH_DFLASH_PMIN=P` | decode | DFlash2 drafts stop where the selector's best candidate has softmax probability < P (default 0: off; llama.cpp's p_min, measured useless here) |
| `OMPH_NO_SWIGLU_GEMM` | A/B | prefill: the FFN's up GEMM writes f32 and `swiglu_f16` runs after it, instead of SwiGLU in the up GEMM's epilogue (#221; bit-identical either way) |
| `OMPH_GEMM_MIN=T` | A/B | `--gemv` runs of T+ tokens (prefill chunks) take the GEMM path (default 16, the measured crossover with the fused GEMM; #129, #141) |
| `OMPH_NO_FUSED_GEMM` | A/B | dequantize each weight to f16, then the GEMM, instead of the fused dequant + WMMA GEMM (#141) |
| `OMPH_STAGE_MIB=N` | A/B | f16 weights staged for the GEMM in row slices of ~N MiB (default 20, cache-resident; 0 = whole tensors; #129, #132) |
| `OMPH_ATTN_DEC_SCALAR` | A/B | decode / verification attention on the scalar kernel instead of the WMMA one (#169) |
| `OMPH_ATTN_SCALAR` | A/B | prefill attention on the scalar kernel instead of the WMMA one (#97) |
| `OMPH_GDN_EXACT` | A/B | prefill delta rule per token (bit-identical to decode) instead of the chunked WY form (#240: f16 WMMAs, KL 0.0002-0.0005 vs exact, ~3 % faster prefill); verifications always run the exact rule |
| `OMPH_GDN_SERIAL` | A/B | a multi-token delta rule in one launch (one workgroup per head) instead of the token-parallel form (#96) |
| `OMPH_NO_BF16_GEMV` | A/B | BF16 weights through the f16 path |
| `OMPH_NO_F16_CACHE` | A/B | re-convert f16-path weights on every call |
| `OMPH_HOST_ARGMAX` | A/B | greedy argmax on the host instead of the device (#102) |
| `OMPH_SKIP_ATTN` / `OMPH_SKIP_FFN` / `OMPH_SKIP_BLOCKS` | ablation | no attention / no FFN / no blocks at all |
| `OMPH_SKIP_GEMV` / `OMPH_SKIP_GEMV_TYPE=T` / `OMPH_SKIP_STAGE` | ablation | no fused GEMVs / none of GGUF type T / no f16 + GEMM matmuls |

`omph-gemv-bench <model> --all-of-type T [--nt N] [--gemm T]`, `--merge-type T` (#203: the sibling GEMVs of type T as separate launches vs one launch on their concatenated rows), `--group [--nt N]` (#214, #255: every sibling group as separate launches vs one `gemv_group` launch at N = 1..16 tokens, outputs compared) times every tensor of a GGUF type back to back (1 s warm-up; `--nt N`: the N-token verification kernels, #63, 2..16 (one pass for the tile types IQ3_S, IQ3_XXS, IQ4_XS, Q4_K, IQ2_S, Q2_K, IQ2_XS, IQ2_XXS, Q6_K, #178, #244, #253, #258; the 1..4-token bodies back to back for the others); `--multi` (without `--nt`): each token of an N-token call against the 1-token call; `--gemm T`: the prefill's fused dequant + WMMA GEMM on T tokens, with its TFLOPS, #208); it reads two switches of its own: `OMPH_BENCH_STREAMS=N` (alternate launches over N streams) and `OMPH_OCCUPANCY` (print the occupancy probe).

## Working agreements

- PLAN.md is the source of truth for architecture; deviations are proposed in an issue first.
- Every performance claim comes with a measurement following PLAN.md §17 (fixed conditions, median of ≥5 runs, results in `bench/`).
- "Done" means: tree builds, the CPU tests and CI pass, the relevant GPU checks of "Entry points" pass, results reported as measured — never claimed untested.
- Keep each change scoped to its issue; keep refactors separate from features.
