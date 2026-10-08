# The llama.cpp reference and its refresh (#393)

`bench/llama.cpp.pin` is the single place that says which llama.cpp omphalos is
compared against. `tools/llama_pin.py` reads it; the tools and scripts that run
llama.cpp call it, so every result says which commit it measured, and a
checkout that drifted from the pin is at least warned about (`OMPH_LLAMA_STRICT=1`
makes it fatal).

```sh
uv run python tools/llama_pin.py                  # the pin
uv run python tools/llama_pin.py --check <llama.cpp>/build-hip/bin
```

The reference is **upstream `ggml-org/llama.cpp` at a release tag** (v0.6.0 =
b11429 at pinning time). The RX 9060 XT fork stays in use for HIP numbers — its
RDNA4 patches are what makes llama.cpp's throughput on gfx1200 what it is — and
is recorded in the pin too (`overlay_*`), including the plain upstream commit
its master is based on.

## Refresh procedure

Run it every 4-6 weeks, or sooner when upstream ships something relevant. The
scheduled workflow `.github/workflows/llama-pin.yml` opens (or bumps) an issue
when a newer upstream release exists. Each refresh is an issue; the findings
and decisions go in its thread.

1. **Move the pin.** Pick the new release (a `bNNNN`/`vX.Y.Z` tag), update
   `url`, `tag`, `release`, `commit`, `date` in `bench/llama.cpp.pin`, and rebase
   the RX 9060 XT overlay onto it (update `overlay_commit`/`overlay_base`).
   Rebuild the configurations the pin names: HIP `gfx1200` (overlay), Vulkan,
   and a CPU-only build for mtmd (`tools/native/build.sh`,
   `OMPH_LLAMA_DIR`/`OMPH_LLAMA_LIB`).
2. **Re-run the comparisons against llama.cpp**, with the new commit recorded:
   `llama-bench` pp/tg (`bench/m0_llama_bench.sh`), the KV-quant KL budget
   (`bench/m5_llama_kv_kl.sh`), the tokenizer / chat-template / vision checks,
   the TTFT and agent-turn matrices (`tools/ttft.py`, `tools/agent_turns.py`),
   the DFlash2 and loop-rate runs. Same conditions as the originals (PLAN.md
   §17: fixed conditions, warm-up discarded, median of ≥5 runs).
3. **Regenerate the golden dump** if the reference forward changed
   (`tools/native/build.sh` + the `dump_tensors` steps in AGENTS.md) and check
   that the engine still matches (KL ≈ 0).
4. **Review the changes between the old and the new pin** for:
   1. **the web UI and its server API** — `tools/ui` (or `tools/server/webui`)
      and the endpoints it calls (`/props`, `/slots`, `/tools`, timings,
      `prompt_progress`, new routes or fields). omph-server embeds llama.cpp's
      UI as is (#378, #380, #382): re-embed the new assets
      (`-DOMPH_WEBUI_DIR=...`), implement or 501 any new route, run
      `tools/check_server.py` and the UI by hand. `/props` reports the UI's
      commit as `webui_llama_commit`;
   2. **engine-side work that could help omphalos** — ggml-hip/ROCm kernels
      (MMQ, WMMA on RDNA4, flash attention, GEMV), quant types, KV-cache
      formats, speculative decoding (MTP, draft models, n-gram), DeltaNet /
      Qwen3.x support, prefill batching, sampling. Each relevant item becomes
      its own issue, measured against our numbers before anything is adopted
      (PLAN.md §17).
5. **Write the findings** in the refresh issue (the decision log), record the
   new results in `bench/results/` with the `llama_commit` of the pin, and
   update PLAN.md / AGENTS.md / README where a reference number or a command
   changes.

Results written before the pin (files that carry no `llama_commit`) were
measured against whatever checkout was on the machine, usually the RX 9060 XT
fork's `build-hip`: where a file cites a commit, that is what it used; the rest
are "pre-pin, unknown commit" and are not comparable with post-pin numbers.
