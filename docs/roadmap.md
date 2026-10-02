# Performance roadmap after M7 / M10

Written 2026-10-02 (#181), after the server (#156), sequence checkpoints
(#158), vision (#160), the speculative identity fix (#161) and the first
long-context measurements (#166). Priorities stay those of PLAN.md: VRAM,
then decode, then prefill. Every item has an issue; every gain claimed below
is an estimate until its issue reports a measurement (PLAN.md §17).

## 1. Where we are

Measured on the RX 9060 XT. Short-context figures come from `bench/results/`;
long-context ones from `bench/results/long-context-vram.txt` and the partial
needle-in-a-haystack runs of #166 (single runs; the NIAH decode figures are
short code answers, decoded speculatively).

| | omphalos | llama.cpp |
|---|---|---|
| decode, plain, short context | 45.8 ms/token (21.8 t/s) | 48.5 ms/token |
| decode, speculative (MTP, k = 3), short context | 19-25 ms/token (40-52 t/s) | ~35 t/s with its MTP |
| prefill at 512 / 16k tokens | ~860 / ~745 t/s | 622 t/s (pp512) |
| prefill at 25k / 50k / 75k / 100k (NIAH) | 669 / 545 / 459 / 395 t/s | 537 / 480 / 438 / not run |
| decode at 25k / 50k / 75k / 100k (NIAH) | 46.4 / 40.6 / 33.7 / 27.3 t/s | 18.3 / 16.0 / 14.0 / not run |
| plain decode at 81k / 112k | 60.1 / 66.1 ms/token | |
| speculative decode at 81k / 112k | 51.7 / 80.3 ms/token | |
| prefill at 81k / 112k | 445 / 373 t/s | |
| VRAM after load, 8k context, MTP on / off | 12.0 / 11.7 GiB | |
| VRAM at 110k, filled: MTP / no MTP / K4 + MTP | 15.2 / 14.5 / 14.4 GiB peak | 15.3-15.5 GiB peak (K8/V4, 25-75k) |
| NIAH, 4 needles + 12 distractors, 25-100k | 16 / 16 found (K8/V4 and K4/V4) | 12 / 12 so far (K8/V4) |

VRAM figures are the device total (the desktop's ~0.15-0.2 GiB included).

## 2. What the measurements say

1. **The weights are tuned, long-context attention is not.** The plain
   decode grows ~0.18 ms per 1k tokens of context; reading the KV at the
   318 GB/s ceiling costs ~0.084 ms (26.6 KB per position over the 16
   attention layers). The decode attention kernel therefore runs at about
   half the bandwidth there. llama.cpp loses more (~0.33 ms per 1k tokens),
   so we are still ahead at every length, but the margin shrinks.
2. **Speculation turns into a loss at long context.** At 112k a
   verification step costs ~150 ms, 2.3x a plain step, and the speculative
   decode (80 ms/token) is slower than plain greedy (66). The verification
   tile computes 24 rows (4 tokens x 6 query heads) with scalar FMAs: at
   short context attention is negligible and this does not show.
3. **MTP acceptance does not drop with the context length** (#168): on the
   same text, the full 60k / 90k / 112k context accepts 53 / 60 / 51 % of the
   drafts, the last 1000 tokens alone 58 / 49 / 67 %. The content decides;
   the step cost is what grows (45-60 vs 23-28 ms/token).
4. **The MTP prefill pass is cheap but not free:** +0.3 % of the prefill at
   112k, +0.7 % at 81k, +1.1-1.5 % at short context (a KV-only pass).
5. **Prefill attention at long context** runs at ~15 TFLOPS effective vs
   ~60 for the GEMM: every 16-query tile dequantizes all the earlier keys
   again (32 times per 512-token chunk).
6. **K4/V4** found every needle, as K8/V4 did, and saves ~0.9 GB at 110k,
   but its KL measured in #81 is over the q8_0/q4_0 budget.
7. **Host <-> device copies** ran at 3.3 GB/s (the link trained at
   2.5 GT/s); after the maintainer's fix 27.7 GB/s at PCIe 4.0 x16 (#176):
   a checkpoint restores in 7.6 ms instead of 51.4.
8. **Measured where the time goes** (#167, `bench/results/phase0-long-context-167.txt`):
   the verification attention costs 3.8x the single-token attention on the
   same keys (77 ms per step at 100k, more than all the GEMVs together); the
   single-token attention reaches 40 % of the bandwidth; the prefill's WMMA
   attention is 25 / 40 / 57 % of the prefill at 25k / 50k / 100k.

## 3. The plan

### Phase 0: measure before changing anything

| issue | what | output |
|---|---|---|
| #166 | finish the NIAH runs (llama.cpp K8/V4 at 100k, K4/V4 at all lengths); save the table, speeds and VRAM | `bench/results/`, PR |
| #167 | per-kernel profile (rocprofv3 + ablations) of a decode step, a verification step and a prefill chunk at 25k / 50k / 100k, and of the MTP drafts | attention vs GEMV share, attention bandwidth and FLOP rate |
| #168 | MTP acceptance: the same continuation with the full 112k context vs the last ~1000 tokens, ~128 generated tokens | content or long-context degradation |
| #176 | re-run the host <-> device bandwidth probe | confirm or retire the 3.3 GB/s figure |

Hypotheses that phase 0 does not confirm are dropped before any code.
Done 2026-10-02: the three attention hypotheses are confirmed (#167); the
MTP acceptance hypothesis is not (#168: content, not context length); the
PCIe link is fixed (#176).

### Phase A: speculation at long context (the largest expected gain)

| issue | what | estimate | constraint |
|---|---|---|---|
| #169 | one WMMA attention kernel for decode and verification (T <= 8) | done: one layer at 100k 771 vs 1269 us (T = 1), 1430 vs 4927 (T = 4); at 50k plain step 51 vs 67 ms, speculative 36 vs 48 ms/token | rows bit-identical to the decode step (#161) |
| #170 | adaptive speculation: a cost model (step times vs context length, recent acceptance) picks k per step, k = 0 included | speculative never slower than plain | output unchanged for any k |
| #171 | MTP attention window: its KV only for the last N tokens (4-8k), drafts attend over those | MTP prefill cost to ~0, ~2 ms per step at 112k, ~0.2 GB less VRAM at 110k | may lower acceptance: measure with #168's test |

### Phase B: decode attention bandwidth at long context

| issue | what | estimate |
|---|---|---|
| #172 | 32-64-key blocks, the next block loaded while computing, one barrier per block, more key splits at long context (fixed per run) | done differently: V tile without padding (occupancy) and the softmax split across the waves; one layer at 100k 771 -> 626 us (T = 1), 1387 -> 819 (T = 4); speculative at 100k 31.3 -> 27.1 ms/token (`bench/results/decode-attention-172.txt`) |

If #169 lands first, #172 applies to the WMMA kernel.

### Phase D: decode weights (short context)

| issue | what | estimate |
|---|---|---|
| #63 | the low-bit GEMVs (IQ3_S, IQ3_XXS, IQ2_*): grid in LDS, dot2 inner loop, uint4 x staging, more rows per workgroup on small projections | -3-5 ms of 45.8 |
| #174 | the NT = 2..4 verification GEMVs on the single-token kernels' fast path | done: verification step 51.3 -> 49.7 ms at 512 tokens, 57.0 -> 55.3 at 25k; speculative -3.5 / -2.7 % per token (`bench/results/verify-gemv-174.txt`) |

### Phase C: prefill attention at long context

| issue | what | estimate |
|---|---|---|
| #173 | larger query tiles (32-64 per workgroup), or the earlier keys dequantized to f16 once per chunk in ~32 MB slices | -30-35 % prefill time at 100k |

### Phase E: VRAM and quality (maintainer decisions)

| issue | what |
|---|---|
| #175 | per-layer K precision (Q4 where the KL allows), or K4 by default above some context length; needs a harder retrieval test and the per-layer KL |

### Housekeeping and backlog

| issue | what |
|---|---|
| #177 | review and close the stale issues #3, #8, #98, #104 |
| #178 | M9: offline converter and final format, after the layout-changing work (#63, #174); load is ~4 s today, so the gain is simplicity and one-time verification |
| #179 | host-RAM KV to switch between conversations (depends on #176's bandwidth) |
| #180 | vision follow-ups: encode overlapped with the text prefill, pinned encoder threads, images in the C ABI |

## 4. Order

1. Phase 0 (#166, #167, #168, #176): cheap, needs the GPU for about an hour.
2. Phase A (#169, then #170; #171 if #168 allows): fixes the case where
   speculation makes decoding worse.
3. Phase B (#172) and phase D (#63, #174): decode, priority 2.
4. Phase C (#173): prefill, priority 3.
5. #178 (M9) once the layouts stop changing.

Phase E and the backlog can be picked up in between when the maintainer
decides.

## 5. Rules for every item

- An issue, a `<type>/<n>-slug` branch, small commits, a PR merged after CI.
- Measurements per PLAN.md §17: fixed conditions, median of 5 runs against
  main in the same session, results in `bench/results/`.
- Speculative output stays identical to plain greedy, bit for bit (#161):
  the verification rows are checked against the decode step.
- Text-only outputs stay identical to main (`ident.sh`-style), or, when the
  arithmetic changes on purpose, the KL / top-1 against the f32-KV reference
  is reported.
- The CPU tests, `check_server.py` and the relevant GPU checks pass.

## 6. Decisions for the maintainer

- K precision (#175): keep Q8 K, per-layer K4, or K4 above some length.
- M9 (#178): when, and whether the GGUF path stays next to the converted
  format.
- Whether the MTP window (#171) is acceptable if it costs some acceptance.
