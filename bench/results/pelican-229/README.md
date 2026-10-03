# Pelican riding a bicycle (#229)

Prompt: "Write SVG code that displays a 2D animation of a pelican riding a bicycle. No additional testing is required."

Rendered with the model's chat template (thinking on, reasoning effort xhigh, the template's default), the same raw prompt to omphalos (`omph-server`) and llama.cpp (`llama-server`, HIP), context 64K, no thinking budget. KV cache k8q4: omphalos' default (K4 on 8 layers, V4), llama.cpp `-ctk q8_0 -ctv q4_0`; k4q4 runs: omphalos `OMPH_KV_K4=1`, llama.cpp `-ctk q4_0 -ctv q4_0`. Temperature 0 is greedy; temperature 1 samples with top_k 20, top_p 0.95, min_p 0, seed 229. MTP: omphalos' speculative decoding (3 drafts), llama.cpp's `--spec-type draft-mtp --spec-draft-n-max 3`. The images are each SVG's first frame (`rsvg-convert`); open the `.svg` files in a browser for the animation. `tools/pelican.py` produced everything here.

| run | KV | tokens (thinking + answer) | total time | prefill | decode | peak VRAM (idle) |
|---|---|---|---|---|---|---|
| llama-plain-t0 | k8q4 | 65396 (thinking not closed) | 3938.1 s | 244.5 t/s | 16.61 t/s | 14054 MiB (356) |
| llama-plain-t1 | k8q4 | 22829 | 1194.7 s | 268.5 t/s | 19.12 t/s | 14054 MiB (356) |
| llama-mtp-k4q4-t0 | k4q4 | 21944 | 705.8 s | 225.5 t/s | 31.11 t/s | 14768 MiB (356) |
| llama-mtp-k4q4-t1 | k4q4 | 15271 | 550.1 s | 250.5 t/s | 27.79 t/s | 14835 MiB (356) |
| llama-mtp-t0 | k8q4 | 41547 | 1382.9 s | 198.2 t/s | 30.05 t/s | 15345 MiB (356) |
| llama-mtp-t1 | k8q4 | 25389 | 928.7 s | 250.5 t/s | 27.35 t/s | 15456 MiB (356) |
| omphalos-plain-t0 | k8q4 | 34373 | 1551.3 s | 318.5 t/s | 22.16 t/s | 13162 MiB (356) |
| omphalos-plain-t1 | k8q4 | 30526 | 1412.4 s | 324.1 t/s | 21.62 t/s | 13162 MiB (356) |
| omphalos-mtp-k4q4-t0 | k4q4 | 65396 (thinking not closed) | 986.5 s | 321.8 t/s | 66.32 t/s | 13572 MiB (356) |
| omphalos-mtp-k4q4-t1 | k4q4 | 56473 | 1196.2 s | 319.1 t/s | 47.22 t/s | 13572 MiB (356) |
| omphalos-mtp-t0 | k8q4 | 34373 | 566.7 s | 332.9 t/s | 60.7 t/s | 13832 MiB (356) |
| omphalos-mtp-t1 | k8q4 | 20506 | 436.1 s | 319.9 t/s | 47.05 t/s | 13841 MiB (356) |

## llama-plain-t0

65396 tokens, 3938.1 s, prefill 244.5 t/s, decode 16.61 t/s, peak VRAM 14054 MiB

(no SVG in the answer)

## llama-plain-t1

22829 tokens, 1194.7 s, prefill 268.5 t/s, decode 19.12 t/s, peak VRAM 14054 MiB

![llama-plain-t1](llama-plain-t1.png)

## llama-mtp-k4q4-t0

21944 tokens, 705.8 s, prefill 225.5 t/s, decode 31.11 t/s, peak VRAM 14768 MiB

![llama-mtp-k4q4-t0](llama-mtp-k4q4-t0.png)

## llama-mtp-k4q4-t1

15271 tokens, 550.1 s, prefill 250.5 t/s, decode 27.79 t/s, peak VRAM 14835 MiB

![llama-mtp-k4q4-t1](llama-mtp-k4q4-t1.png)

## llama-mtp-t0

41547 tokens, 1382.9 s, prefill 198.2 t/s, decode 30.05 t/s, peak VRAM 15345 MiB

![llama-mtp-t0](llama-mtp-t0.png)

## llama-mtp-t1

25389 tokens, 928.7 s, prefill 250.5 t/s, decode 27.35 t/s, peak VRAM 15456 MiB

![llama-mtp-t1](llama-mtp-t1.png)

## omphalos-plain-t0

34373 tokens, 1551.3 s, prefill 318.5 t/s, decode 22.16 t/s, peak VRAM 13162 MiB

![omphalos-plain-t0](omphalos-plain-t0.png)

## omphalos-plain-t1

30526 tokens, 1412.4 s, prefill 324.1 t/s, decode 21.62 t/s, peak VRAM 13162 MiB

![omphalos-plain-t1](omphalos-plain-t1.png)

## omphalos-mtp-k4q4-t0

65396 tokens, 986.5 s, prefill 321.8 t/s, decode 66.32 t/s, peak VRAM 13572 MiB

(no SVG in the answer)

## omphalos-mtp-k4q4-t1

56473 tokens, 1196.2 s, prefill 319.1 t/s, decode 47.22 t/s, peak VRAM 13572 MiB

![omphalos-mtp-k4q4-t1](omphalos-mtp-k4q4-t1.png)

## omphalos-mtp-t0

34373 tokens, 566.7 s, prefill 332.9 t/s, decode 60.7 t/s, peak VRAM 13832 MiB

![omphalos-mtp-t0](omphalos-mtp-t0.png)

## omphalos-mtp-t1

20506 tokens, 436.1 s, prefill 319.9 t/s, decode 47.05 t/s, peak VRAM 13841 MiB

![omphalos-mtp-t1](omphalos-mtp-t1.png)

## Findings

Conditions: omphalos at 9744d08, llama.cpp: the RX9060XT fork's HIP build; 2026-10-03, on a card the maintainer had overclocked that morning (plain decode step 44.2 -> 43.3 ms). One sample per run: the token counts and pictures say little about quality, the speeds are what this test measures.

- **Speculative decoding (MTP)**: omphalos 60.7 / 47.0 t/s at temperature 0 / 1, llama.cpp 30.1 / 27.4 (2.0x / 1.7x). With K4/V4, omphalos 47.2 t/s at temperature 1 on a 56k-token run vs llama.cpp's 27.8 on 15k (1.7x).
- **Greedy MTP is exact**: omphalos-plain-t0 and omphalos-mtp-t0 are the same 34 373 tokens, byte for byte. On the same text, MTP decodes 2.74x faster (22.2 -> 60.7 t/s) for +670 MiB of VRAM.
- **Plain decode**: omphalos 21.6 t/s vs llama.cpp 19.1 at temperature 1. llama.cpp's 16.6 at temperature 0 is over a 65k-token context, so it does not compare.
- **VRAM**: omphalos peaks at 13.2-13.8 GB with a 64K context, llama.cpp at 14.1-15.5 GB. K4/V4 saves ~260 MiB on omphalos at 64K and ~600 MiB on llama.cpp.
- **Two greedy runs looped** until the context ran out, with no `</think>`: llama-plain-t0 (22 % unique lines) and omphalos-mtp-k4q4-t0 (14 %, the SVG rewritten over and over). The K4/V4 run matches the k8q4 one for its first 1874 characters, then a near tie flips one word ("simple" / "manageable"). The 66 t/s of the looping run come from easily drafted repeated text. Every sampled run (temperature 1) closed its thinking and drew a picture.
- **Prefill** is not measured here: the prompt is a few dozen tokens (#232-#234 measure it).
