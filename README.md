# omphalos

A from-scratch inference engine for one model on one GPU: `Qwen3.8-27B`
(GSQ-RCO IQ3_S GGUF) on an AMD Radeon RX 9060 XT 16 GB (RDNA4, `gfx1200`,
ROCm/HIP). The priorities, in order: VRAM, decode speed, prefill speed. The
weights stay bit-exact with the GGUF (lossless re-layouts only).

- [PLAN.md](PLAN.md): the design and its rationale, the milestones, and the
  measured status;
- [AGENTS.md](AGENTS.md): how work is done here, plus every build, run and
  check command (Entry points);
- [docs/technical-report.md](docs/technical-report.md): the kernels, layouts
  and measurements, with the math;
- [docs/roadmap.md](docs/roadmap.md): the measured status at short and long
  context and the planned performance work, one issue per item;
- [tools/README.md](tools/README.md): the Python tooling and validation
  scripts.

## What it does

| | |
|---|---|
| Decode | fused dequant + dot GEMVs on load-time repacked weights; greedy decoding drafts 3 tokens per step with the model's own MTP head and verifies them in one pass, with output bit-identical to plain greedy |
| Prefill | fused dequant + WMMA GEMM, WMMA flash attention, token-parallel delta net |
| KV cache | K in Q8, V in Q4, the last 128 tokens exact in an FP16 ring |
| Text | the GGUF's byte-level BPE tokenizer and its chat template (thinking, reasoning effort, tool calls), reimplemented in C++ and checked against llama.cpp and jinja2 |
| Sampling | greedy (speculative), or temperature / top-k / top-p / min-p with a seed |
| Cache reuse | a prompt that extends the cached sequence prefills only its new tokens; checkpoints in host RAM let a retried answer or an edited history resume from an earlier point |
| Vision | images encoded on the CPU by llama.cpp's mtmd (no VRAM), fed as embeddings with M-RoPE positions |
| Interfaces | `omph-generate` (CLI), `omph-server` (OpenAI-compatible HTTP), `libomphalos.so` with a C ABI (`engine/include/omphalos.h`) |

Measured on the RX 9060 XT (details and conditions in `bench/results/`):

| | |
|---|---|
| decode, speculative greedy (MTP, k = 3) | 19-25 ms/token (40-52 t/s), depending on the text |
| decode, plain | 45.8 ms/token (21.8 t/s; llama.cpp: 48.5 ms) |
| prefill | ~860 t/s at 512 tokens, ~875 t/s at 2k, ~745 t/s at 16k (llama.cpp pp512: 622 t/s) |
| VRAM | 12.0 GiB of the 16 in use after load with an 8k context and the MTP head, 11.7 without it (device total, ~0.15 GiB of desktop included) |
| model load | ~4 s (file in the page cache) |
| an image (640x488, 300 tokens) | 2.8 s to encode on the CPU, then a normal prefill |

## Build

```sh
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release && cmake --build engine/build -j
```

Vision needs llama.cpp's mtmd from a CPU-only llama.cpp build: AGENTS.md
("build with vision") has the commands; then add
`-DOMPH_LLAMA_DIR=<llama.cpp> -DOMPH_LLAMA_LIB=<its build>/bin` to the first
line.

## Use

```sh
M=models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf

# a chat request (OpenAI-style messages) in, streamed text out
echo '{"messages":[{"role":"user","content":"Hi!"}],"add_generation_prompt":true}' |
    engine/build/omph-generate $M --chat --max 256

# an image (with a vision build)
engine/build/omph-generate $M --chat --mmproj models/mmproj-Qwen3.8-27B-BF16.gguf \
    --image photo.jpg < request-with-an-image-item.json

# the OpenAI-compatible server on 127.0.0.1:8080
engine/build/omph-server $M [--mmproj models/mmproj-Qwen3.8-27B-BF16.gguf] [--ctx 8192]
```

`omph-server` serves `/v1/chat/completions` (streamed or not, with
`reasoning_content`, `tool_calls`, stop strings, usage with cached tokens,
images as base64 `data:` URLs), `/v1/completions`, `/v1/models` and
`/health`, one request at a time; existing OpenAI clients work unchanged.
Requests that leave the sampling out are greedy, the fastest path. Options:
`--port`, `--host`, `--ctx`, `--api-key`, `--cors`, `--cache-ram`, the
default sampling (`--temp`, `--top-k`, `--top-p`, `--min-p`, `--max-tokens`).

`engine/examples/capi_demo.c` shows the C ABI: load, tokenize, render a chat,
generate with a token callback.
