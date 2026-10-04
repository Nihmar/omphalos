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

# the OpenAI-compatible server on 127.0.0.1:8080 (every option in "omph-server in full" below)
engine/build/omph-server $M [--mmproj models/mmproj-Qwen3.8-27B-BF16.gguf] [--ctx 8192]

# the other binaries (omph-run, omph-tokenize, the validation and benchmark tools):
# every option is documented in docs/tools.md

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
context repeats; `--dflash` switches to the DFlash2 drafter). Sampling fields
the engine does not implement (`frequency_penalty`, `presence_penalty`,
`repetition_penalty`, `repeat_penalty`, `logit_bias`) are refused with a 400
unless they hold their no-op value (0, 1.0, `{}`) -- never dropped in silence.
The three llama.cpp penalties are implemented (#298): `repeat_penalty`
(`repetition_penalty` is accepted as the vLLM spelling), `frequency_penalty`
and `presence_penalty`, all over the same `repeat_last_n` window of the last
tokens of the sequence (64 by default, `--repeat-last-n`), applied to the raw
logits as llama.cpp's `penalties` sampler does. A special token written
literally in a message's text or in a tool result (`<|im_end|>` in a file an
agent reads) stays text: only the template's own structure is parsed for
special tokens (#292).

### `omph-server` in full

```
omph-server <model.omph> [options]
```

What it serves, one request at a time and in the OpenAI wire format:
`GET /health`, `GET /v1/models`, `GET /v1/models/<alias>`,
`POST /v1/chat/completions` and `POST /v1/completions` (the three `/v1/*` paths
are also served without the prefix). The body of a chat request is the
template's `messages`; a chat's next turn prefills only its new tokens (the
cache is the engine's,
[#158](https://github.com/Nihmar/omphalos/issues/158),
[#179](https://github.com/Nihmar/omphalos/issues/179)), so a client that keeps
the conversation growing stays fast.

#### Options

| option | default | effect |
|---|---|---|
| `--host H` | `127.0.0.1` | address to bind |
| `--port P` | `8080` | port |
| `--ctx N` | `8192` | KV capacity in tokens: the hard limit of one conversation. A longer prompt is a 400; VRAM grows with it (~26 KiB per token with the default K/Q8 + V/Q4 cache, 0.87 GB at 32k, [#58](https://github.com/Nihmar/omphalos/issues/58)) |
| `--chunk N` | `512` | tokens per prefill chunk, i.e. the size of the activation buffers |
| `--alias NAME` | the file name, minus a `.gguf` suffix | the model id in `/v1/models` and in every response |
| `--no-mtp` | off | do not load the MTP block: no speculative drafts, -352 MiB of VRAM |
| `--dflash FILE` | off | draft with a DFlash2 drafter `.omph` (7 drafts per step) instead of the MTP block |
| `--cache-ram MIB` | `2048` | pinned host RAM for **sequence checkpoints** (`0`: none): an answer that is retried, or a history whose reasoning was dropped, resumes from a checkpoint instead of prefilling again |
| `--kv-ram MIB` | `8192` | pinned host RAM for **whole conversations** (`0`: none): a prompt that leaves the cached conversation saves it first, one that continues a saved conversation restores it |
| `--mmproj FILE` | off | the vision encoder: images as base64 `data:` URLs, encoded on the CPU (needs a build with `OMPH_LLAMA_DIR`) |
| `--api-key KEY` | off | require `Authorization: Bearer KEY` on everything but `/health` |
| `--cors ORIGIN` | off | allow browser requests from `ORIGIN` (`*` for any), preflight included |
| `--temp T` | `0` | default temperature; `0` is greedy, the fastest path |
| `--top-k K` | `0` | default top-k; `0` is off |
| `--top-p P` | `1` | default top-p; `1` is off |
| `--min-p P` | `0` | default min-p; `0` is off |
| `--repeat-penalty P` | `1.0` | default `repeat_penalty`; `1` is off |
| `--repeat-last-n N` | `64` | the tokens the three penalties look at; `0` is off |
| `--frequency-penalty P` | `0` | default `frequency_penalty` |
| `--presence-penalty P` | `0` | default `presence_penalty` |
| `--max-tokens N` | `-1` | default cap per request; `-1` runs until the context is full |

A numeric option whose value is not entirely a number is refused, naming the
flag and the value it got (`--top-p 0.95--top-k`, a flag pasted without its
space, used to set `top_p` to 0.95 and ignore the rest; #306) -- a typo cannot
configure a run half-way.

Every sampling option is only a **default for requests that leave that field
out**: a client that sends its own wins. Greedy and sampled requests both
decode speculatively (MTP drafts, or DFlash2's, plus n-gram drafts when the
context repeats); the one exception is a *penalized greedy* request, whose
argmax has to run on the host and therefore gives up the drafts.

#### Request fields

The OpenAI fields a request may carry, and what the engine does with them.
A field the engine knows but does not implement (`logit_bias`), and any value
out of range or of the wrong type, is refused with a 400 naming it -- never
dropped in silence ([#284](https://github.com/Nihmar/omphalos/issues/284)).

| field | notes |
|---|---|
| `messages` | required for chat: `{role, content, reasoning_content?, tool_calls?}`, `content` a string or a list of parts (`text`, `image_url`). `developer` is read as `system` |
| `model` | ignored: the server serves the one model it was started with |
| `tools` | declarations in the OpenAI shape; the template renders them into the prompt, and `<tool_call>` blocks in the answer come back as `tool_calls` |
| `tool_choice` | only `"none"` (drop the tools from the prompt) |
| `chat_template_kwargs` | `enable_thinking` (bool: the engine's `<think>` block, on unless turned off), `reasoning_effort` (`xhigh`/`medium`/`low`, the template's own switch, default `xhigh`), `preserve_thinking` (bool, default true: keep the history's reasoning blocks in the prompt) |
| `reasoning_effort` | top level: `off`/`none`/`minimal` turn thinking off, `low`/`medium`/`xhigh` set it, `high` reads as `xhigh` |
| `temperature` | `0` (or absent, with no default set) is greedy. Range `[0, 2]` |
| `top_p`, `top_k`, `min_p` | applied in that order after the penalties; `top_p 1`, `top_k 0`, `min_p 0` are off |
| `seed` | integer; makes the sampled draw reproducible |
| `repeat_penalty` | llama.cpp's, `1.0` is off. `repetition_penalty` is accepted as the vLLM spelling |
| `repeat_last_n` | the window the three penalties look at; `0` is off |
| `frequency_penalty`, `presence_penalty` | OpenAI's, `[-2, 2]`, in the same window |
| `max_tokens`, `max_completion_tokens` | positive integer; absent means the server's `--max-tokens` |
| `stop` | a string or a list of strings; the text from the first hit on is dropped |
| `stream` | SSE `chat.completion.chunk` / `text_completion` events |
| `stream_options.include_usage` | adds a final chunk with the `usage` object |
| `n` | must be `1` (the engine serves one sequence) |
| `logit_bias` | refused: not implemented. Send `{}` or nothing |
| `image_url` parts | `data:` URLs only (`data:image/png;base64,...`), one per image item; `--mmproj` is required and its `<|image_pad|>` placeholder must line up with the images. Audio, video and file parts are refused |
| `prompt` | `/v1/completions`: a string, token ids, or a batch of one -- no chat template, so no reasoning split |
| `add_generation_prompt` | ignored for chat: the server always appends the assistant's turn (that is what a chat request means here) |
| `echo` | `/v1/completions`: prepend the prompt to `choices[0].text` |

#### Responses

A chat answer splits at `</think>`: everything before it is
`choices[0].message.reasoning_content`, everything after it is `.content`, and
`<tool_call>` blocks in the content become `.tool_calls` (arguments as a JSON
string, OpenAI's shape). `finish_reason` is `stop` (an end-of-generation token
or a stop string), `length` (the cap, or the context filling up) or
`tool_calls`. Streamed and whole answers carry the same text; the streams hold
back text that may still become a tag or a stop string, so a delta boundary is
not a token boundary.

The full response also carries llama.cpp's `timings` object (`prompt_n`,
`prompt_ms`, `predicted_n`, `predicted_ms`, `draft_n`, `draft_n_accepted`) and
an OpenAI `usage` with `prompt_tokens_details.cached_tokens`: how much of the
prompt the cache already held, which is what a client can use to tell a
resumed conversation from a fresh prefill.

Errors are OpenAI-shaped (`{"error": {"message", "type", "param",
"code"}}`): 400 for a malformed request, a prompt longer than `--ctx`, or
image placeholders that do not match the images, 401 without the API key, 405
for a wrong method.

#### Log

On stderr, per request: the ready line (the model, the context), a progress
line every 3 seconds while **prefilling** (`  prefill 12288 / 98255 tokens, 820
t/s`) and while **decoding** (`  2048 tokens, 39.1 t/s (last 3 s: 40.7 t/s),
drafts accepted 71 %`), then the summary: the sampling the request ran with,
the prompt tokens and how many of them the cache held or restored, the prefill
and decode times and rates, the drafts accepted and the stop reason
(`(client gone)` when the client left first). Nothing for a minute means the
model is loading or the prompt is prefilling, not that the server is hung --
with `--cache-ram`/`--kv-ram` a continued conversation resumes instead of
prefilling and says so (`restored`).

#### Environment

The server passes `OMPH_*` through to the engine. The ones worth knowing when
serving (the complete list, with the diagnostics and the ablations, is in
[AGENTS.md](AGENTS.md)):

| variable | effect |
|---|---|
| `OMPH_KV_K4_LAYERS=i,j,...\|none` | K in Q4 on those attention layers (0-15), Q8 on the rest; the default is the eight measured least sensitive ([#175](https://github.com/Nihmar/omphalos/issues/175)), `none` is K8 everywhere |
| `OMPH_KV_K4` | K4 on every layer: the most aggressive cache, measured over llama.cpp's q8_0/q4_0 KL budget (#175): an experiment |
| `OMPH_KV_F32`, `OMPH_KV_HOST` | exact f32 KV in VRAM / in pinned host RAM (references, much more memory) |
| `OMPH_KV_WINDOW` | every query reads its last N keys exactly from an FP16 ring (default 128, `0` off) |
| `OMPH_MTP_WINDOW` | the MTP block's KV window (default 16384: 16 sinks plus the last 16k-32k positions), -150 MiB of VRAM at 128k |
| `OMPH_DRAFT_VOCAB` | drafts over the first N token ids (default 98304), a few percent faster |
| `OMPH_NGRAM`, `OMPH_NGRAM_MIN` | n-gram (prompt lookup) drafts instead of the model's (default on; `OMPH_NGRAM=0` off) |
| `OMPH_DFLASH_KEEP`, `OMPH_DFLASH_PMIN` | how many of DFlash2's 7 drafts to keep |
| `OMPH_GEMM_MIN` | the token count from which `--gemv` runs take the prefill GEMM path (default 16) |
| `OMPH_OVERLAP` | a side stream for sibling GEMVs (off by default: the persistent-warp GEMVs fill the GPU alone) |
| `OMPH_HOST_ARGMAX` | greedy argmax on the host instead of the device |
| `OMPH_TIMING` | VRAM after load and a per-step timing line on stderr |

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
cd /path/to/omphalos
M=models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.omph   # or wherever your .omph is
engine/build/omph-server $M --ctx 131072 \
    --temp 0.6 --top-p 0.95 --top-k 20 \
    --repeat-penalty 1.1 --repeat-last-n 1024 \
    --max-tokens 16384 \
    --dflash models/Qwen3.8-27B-DFlash2-Q4_K_M.omph
```

`M=` is repeated here because this block is copied on its own: with an empty
`$M` the first argument becomes the model path and the engine answers
`unknown option 131072`.

- `--max-tokens` (default: until the context is full) bounds one turn: without
  it a runaway generation can decode the whole 131k.
- **repetition**: `--repeat-penalty 1.1 --repeat-last-n 1024` (llama.cpp's
  sampler, #298) is the cheap first thing to try: a window that long covers a
  repeated paragraph, while the default 64 only covers repeated phrases. It
  scales the logits of the window's tokens (sign-aware), so it also touches
  the identifiers a coding session legitimately reuses: lower it to 1.05 if
  the answers get worse, not the repetition away.
- thinking is on by default, and `preserve_thinking` (the template's default,
  as in llama.cpp) keeps every earlier reasoning block in the prompt. A client
  turns thinking off with `"chat_template_kwargs": {"enable_thinking": false}`
  (or `"reasoning_effort": "none"`), and drops the history's reasoning with
  `{"preserve_thinking": false}`.
- `--temp`/`--top-p`/... only set what a request leaves out; a client that
  sends its own sampling wins. A sampling field the engine does not implement
  is refused, not ignored (#284).
- the same options have a terminal UI, `uv run python omph_tui.py` in `tools/`
  (form, live log, one row per request, `--profile` presets; it works over ssh
  from a phone): [docs/tools.md](docs/tools.md#omph-tui).

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
        "samplingParams": { "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                            "repeat_penalty": 1.1, "repeat_last_n": 1024 },
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
