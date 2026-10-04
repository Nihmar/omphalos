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
| Decode | fused dequant + dot GEMVs on load-time repacked weights; every decode drafts 3 tokens per step with the model's own MTP head, 7 with the DFlash2 drafter (`--dflash`, #245) or up to 15 from an n-gram lookup when the context repeats (#199), and verifies them in one pass: greedy output bit-identical to plain greedy, sampled output with plain sampling's distribution (speculative sampling, #197) |
| Prefill | fused dequant + WMMA GEMM, WMMA flash attention, token-parallel delta net |
| KV cache | V in Q4; K in Q4 on the 8 least sensitive of the 16 attention layers and Q8 on the rest (#175); the last 128 tokens exact in an FP16 ring |
| Text | the GGUF's byte-level BPE tokenizer and its chat template (thinking, reasoning effort, tool calls), reimplemented in C++ and checked against llama.cpp and jinja2 |
| Sampling | greedy, or temperature / top-k / top-p / min-p with a seed; both decode speculatively (code at temperature 0.6: 18.5 vs 44.4 ms/token) |
| Cache reuse | a prompt that extends the cached sequence prefills only its new tokens; checkpoints in host RAM let a retried answer or an edited history resume from an earlier point; a prompt that leaves the cached sequence saves its whole conversation in host RAM (`--kv-ram`, default 8 GiB) and returns to it with an upload |
| Vision | images encoded on the CPU by llama.cpp's mtmd (no VRAM), fed as embeddings with M-RoPE positions |
| Interfaces | `omph-generate` (CLI), `omph-server` (OpenAI-compatible HTTP), `libomphalos.so` with a C ABI (`engine/include/omphalos.h`) |

Measured on the RX 9060 XT (details and conditions in `bench/results/`):

| | |
|---|---|
| decode, speculative greedy (MTP, k = 3) | 19-25 ms/token (40-52 t/s), depending on the text |
| decode, speculative greedy (DFlash2 + n-gram, #245) | 79-265 t/s on code, quotes and edits, 46 on prose |
| decode, plain | 45.8 ms/token (21.8 t/s; llama.cpp: 48.5 ms) |
| prefill | ~860 t/s at 512 tokens, ~875 t/s at 2k, ~745 t/s at 16k (llama.cpp pp512: 622 t/s) |
| VRAM | 12.0 GiB of the 16 in use after load with an 8k context and the MTP head, 11.7 without it; at a 128K context (K4/V4) a 100k-token prompt peaks at 15.0 GiB with DFlash2, 14.4 with MTP, 13.6 plain — it fits (device total, the desktop included) |
| a long reasoning task (#229: an animated SVG, 15-65k tokens of thinking, 64K context) | speculative 60.7 / 47.0 t/s at temperature 0 / 1 with MTP and 77.5 / 47.3 with DFlash2, vs llama.cpp's MTP 30.1 / 27.4, peak VRAM 13.8 vs 15.5 GB; greedy output byte-identical to plain greedy over 34k tokens ([bench/results/pelican-229](bench/results/pelican-229/README.md)) |
| model load | <1 s from the `.omph` file (in the page cache): a 3-token run takes 0.89 s |
| an image (640x488, 300 tokens) | 2.8 s to encode on the CPU, overlapping the prefill of the text before it (first token -22 % to -41 %, #180), then a normal prefill |

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
# once: the engine loads the .omph file omph-convert writes from the GGUF (#178)
engine/build/omph-convert models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
M=models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.omph  # the examples below use $M

# a chat request (OpenAI-style messages) in, streamed text out
echo '{"messages":[{"role":"user","content":"Hi!"}],"add_generation_prompt":true}' |
    engine/build/omph-generate $M --chat --max 256

# an image (with a vision build)
engine/build/omph-generate $M --chat --mmproj models/mmproj-Qwen3.8-27B-BF16.gguf \
    --image photo.jpg < request-with-an-image-item.json

# the OpenAI-compatible server on 127.0.0.1:8080
engine/build/omph-server $M [--mmproj models/mmproj-Qwen3.8-27B-BF16.gguf] [--ctx 8192]

# the DFlash2 drafter needs its own .omph too (7 drafts per step, #245)
engine/build/omph-convert models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf

# everything on: a 128K context, DFlash2 drafts, n-gram drafts (the default)
# and images (a vision build); ~15 GiB of VRAM, the desktop included. The KV
# stays on its measured default (K4 on the eight least sensitive attention
# layers, #175); OMPH_KV_K4=1 puts K4 everywhere and saves ~0.4 GB more at
# 106k, at KL 0.0028 against 0.0015 for the mix and 0.0009 for K8 at 16k
# (bench/results/k4-per-layer-175.txt): an experiment, not the default.
OMPH_NGRAM=1 engine/build/omph-server \
    models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.omph --ctx 131072 --port 7070 \
    --dflash models/Qwen3.8-27B-DFlash2-Q4_K_M.omph \
    --mmproj models/mmproj-Qwen3.8-27B-BF16.gguf
```

`omph-server` serves `/v1/chat/completions` (streamed or not, with
`reasoning_content`, `tool_calls`, stop strings, usage with cached tokens,
images as base64 `data:` URLs), `/v1/completions`, `/v1/models` and
`/health`, one request at a time; existing OpenAI clients work unchanged.
Requests that leave the sampling out are greedy, the fastest path; the
decode is speculative either way (MTP drafts, plus n-gram drafts when the
context repeats; `--dflash` switches to the DFlash2 drafter). Options:
`--port`, `--host`, `--ctx`, `--alias`, `--api-key`, `--cors`, `--cache-ram`
and `--kv-ram` (sequence checkpoints and whole conversations in pinned host
RAM), `--mmproj`, `--dflash`, the default sampling (`--temp`, `--top-k`,
`--top-p`, `--min-p`, `--max-tokens`).

### Serving a coding agent

An agent whose model entry has no sampling parameters does not send
`temperature`, so the server's default applies: greedy. On a reasoning model
at a 20k-100k context that is the fastest way to a repetition loop -- a whole
session with pi (`provider: omphalos`, 2026-10-04) ended in five
`pi-loop-police` truncations
([#285](https://github.com/Nihmar/omphalos/issues/285)). Serve an agent with
the sampler Qwen recommends for thinking mode, and with a cap on a runaway
turn:

```sh
engine/build/omph-server $M --ctx 131072 --temp 0.6 --top-p 0.95 --top-k 20 \
    --max-tokens 16384 --dflash models/Qwen3.8-27B-DFlash2-Q4_K_M.omph
```

- `--max-tokens` (default: until the context is full) bounds one turn: without
  it a runaway generation can decode the whole 131k.
- thinking is on by default, and `preserve_thinking` (the template's default,
  as in llama.cpp) keeps every earlier reasoning block in the prompt. A client
  turns thinking off with `"chat_template_kwargs": {"enable_thinking": false}`
  (or `"reasoning_effort": "none"`), and drops the history's reasoning with
  `{"preserve_thinking": false}`.
- `--temp`/`--top-p`/... only set what a request leaves out; a client that
  sends its own sampling wins. A sampling field the engine does not implement
  is refused, not ignored (#284).

pi (`~/.pi/agent/models.json`) does not know a custom provider is a reasoning
model, so it sends no temperature and its thinking level does nothing.
Declaring both makes it send them (`off` -> `enable_thinking: false`, so the
engine's sampler is the only variable left):

```json
{
  "providers": {
    "omphalos": {
      "baseUrl": "http://localhost:7070/v1",
      "api": "openai-completions",
      "apiKey": "unused",
      "piGuiCustomEndpoint": true,
      "models": [{
        "id": "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.omph",
        "name": "Qwen3.8-27B (omphalos)",
        "reasoning": true,
        "thinkingLevelMap": { "off": "off", "low": "low", "medium": "medium", "high": "xhigh" },
        "contextWindow": 131072,
        "maxTokens": 16384,
        "samplingParams": { "temperature": 0.6, "top_p": 0.95, "top_k": 20 },
        "compat": { "thinkingFormat": "qwen-chat-template", "supportsReasoningEffort": false,
                    "supportsDeveloperRole": false, "supportsStore": false,
                    "maxTokensField": "max_tokens" }
      }]
    }
  }
}
```

`engine/examples/capi_demo.c` shows the C ABI: load, tokenize, render a chat,
generate with a token callback.
