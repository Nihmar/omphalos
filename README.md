# omphalos

A from-scratch inference engine for one model on one GPU: `Qwen3.8-27B`
(GSQ-RCO IQ3_S GGUF) on an AMD Radeon RX 9060 XT 16 GB (RDNA4, `gfx1200`,
ROCm/HIP). The priorities, in order: VRAM, decode speed, prefill speed. The
weights stay bit-exact with the GGUF (lossless re-layouts only).

- [PLAN.md](PLAN.md): the design and its rationale, the milestones, and the
  measured status;
- [AGENTS.md](AGENTS.md): how work is done here, plus the build, run and
  check commands (Entry points);
- [docs/technical-report.md](docs/technical-report.md): the kernels, layouts
  and measurements, with the math;
- [tools/README.md](tools/README.md): the Python tooling and validation
  scripts.

Build and run:

```sh
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release && cmake --build engine/build -j
engine/build/omph-run <model.gguf> <tokens.txt> logits.f32 --last-logits \
    --gemv --generate 64 --gen-out gen.txt
```
