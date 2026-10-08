---
name: llama.cpp refresh
about: Move bench/llama.cpp.pin to a new release and re-validate against it (#393)
title: "llama.cpp refresh: "
labels: chore
---

The pin has moved (or a newer upstream release exists): follow the checklist in
[docs/llama-refresh.md](../blob/main/docs/llama-refresh.md). The findings of
this refresh go in this issue.

- [ ] Update `bench/llama.cpp.pin` (tag, release, commit, date) and rebase the
      RX 9060 XT overlay onto it (`overlay_commit` / `overlay_base`)
- [ ] Rebuild: HIP `gfx1200` (overlay), Vulkan, CPU-only mtmd
- [ ] Re-run the llama.cpp comparisons with the new commit recorded
      (`m0_llama_bench.sh`, `m5_llama_kv_kl.sh`, tokenizer / chat-template /
      vision checks, TTFT and agent turns, DFlash2, loop rate)
- [ ] Regenerate the golden dump if the reference forward changed; engine KL ≈ 0
- [ ] Review the web UI and its server API between the old and the new pin;
      re-embed the assets, implement or 501 new routes, run `check_server.py`
- [ ] Review the engine-side changes (HIP/ROCm kernels, quant types, KV
      formats, speculative decoding, DeltaNet/Qwen3.x, sampling); file one
      issue per relevant item
- [ ] Record the results in `bench/results/` with `llama_commit`, write the
      findings here, update PLAN.md / AGENTS.md / README
