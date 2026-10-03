# Pelican riding a bicycle (#229)

Prompt: "Write SVG code that displays a 2D animation of a pelican riding a bicycle. No additional testing is required."

Rendered with the model's chat template (thinking on, reasoning effort xhigh, the template's default), the same raw prompt to omphalos (`omph-server`) and llama.cpp (`llama-server`, HIP), context 64K, no thinking budget. KV cache k8q4: omphalos' default (K4 on 8 layers, V4), llama.cpp `-ctk q8_0 -ctv q4_0`; k4q4 runs: omphalos `OMPH_KV_K4=1`, llama.cpp `-ctk q4_0 -ctv q4_0`. Temperature 0 is greedy; 0.6 and 1 sample with top_k 20, top_p 0.95, min_p 0, seed 229. MTP: omphalos' speculative decoding (3 drafts), llama.cpp's `--spec-type draft-mtp --spec-draft-n-max 3`. DFlash2: the z-lab Q4_K_M drafter, 7 drafts (omphalos `--dflash`, llama.cpp `-md ... --spec-type draft-dflash --spec-draft-n-max 7`). The images are each SVG's first frame (`rsvg-convert`); open the `.svg` files in a browser for the animation. `tools/pelican.py` produced everything here. omphalos runs from 2026-10-04 on (the `dflash` runs, `mtp-k4q4-t0.6`) also draft with n-grams of the context (on by default since #199, sampled runs included); the older ones predate them (#262 separates the two).

| run | KV | tokens (thinking + answer) | total time | prefill | decode | peak VRAM (idle) |
|---|---|---|---|---|---|---|
| llama-mtp-k4q4-t0 | k4q4 | 21944 | 705.8 s | 225.5 t/s | 31.11 t/s | 14768 MiB (356) |
| llama-mtp-k4q4-t1 | k4q4 | 15271 | 550.1 s | 250.5 t/s | 27.79 t/s | 14835 MiB (356) |
| llama-mtp-t0 | k8q4 | 41547 | 1382.9 s | 198.2 t/s | 30.05 t/s | 15345 MiB (356) |
| llama-mtp-t1 | k8q4 | 25389 | 928.7 s | 250.5 t/s | 27.35 t/s | 15456 MiB (356) |
| llama-plain-t0 | k8q4 | 65396 (thinking not closed) | 3938.1 s | 244.5 t/s | 16.61 t/s | 14054 MiB (356) |
| llama-plain-t1 | k8q4 | 22829 | 1194.7 s | 268.5 t/s | 19.12 t/s | 14054 MiB (356) |
| omphalos-dflash-k4q4-t0 | k4q4 | 20121 | 260.1 s | 291.9 t/s | 77.47 t/s | 14195 MiB (210) |
| omphalos-dflash-k4q4-t0.5 | k4q4 | 31388 | 469.2 s | 289.2 t/s | 66.96 t/s | 14322 MiB (210) |
| omphalos-dflash-k4q4-t0.6 | k4q4 | 22305 | 357.0 s | 279.2 t/s | 62.54 t/s | 14195 MiB (210) |
| omphalos-dflash-k4q4-t1 | k4q4 | 22814 | 482.6 s | 281.8 t/s | 47.31 t/s | 14195 MiB (210) |
| omphalos-mtp-k4q4-t0 | k4q4 | 65396 (thinking not closed) | 986.5 s | 321.8 t/s | 66.32 t/s | 13572 MiB (356) |
| omphalos-mtp-k4q4-t0.6 | k4q4 | 30899 | 487.6 s | 288.7 t/s | 63.43 t/s | 13490 MiB (210) |
| omphalos-mtp-k4q4-t1 | k4q4 | 56473 | 1196.2 s | 319.1 t/s | 47.22 t/s | 13572 MiB (356) |
| omphalos-mtp-t0 | k8q4 | 34373 | 566.7 s | 332.9 t/s | 60.7 t/s | 13832 MiB (356) |
| omphalos-mtp-t1 | k8q4 | 20506 | 436.1 s | 319.9 t/s | 47.05 t/s | 13841 MiB (356) |
| omphalos-plain-t0 | k8q4 | 34373 | 1551.3 s | 318.5 t/s | 22.16 t/s | 13162 MiB (356) |
| omphalos-plain-t1 | k8q4 | 30526 | 1412.4 s | 324.1 t/s | 21.62 t/s | 13162 MiB (356) |

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

## llama-plain-t0

65396 tokens, 3938.1 s, prefill 244.5 t/s, decode 16.61 t/s, peak VRAM 14054 MiB

(no SVG in the answer)

## llama-plain-t1

22829 tokens, 1194.7 s, prefill 268.5 t/s, decode 19.12 t/s, peak VRAM 14054 MiB

![llama-plain-t1](llama-plain-t1.png)

## omphalos-dflash-k4q4-t0

20121 tokens, 260.1 s, prefill 291.9 t/s, decode 77.47 t/s, peak VRAM 14195 MiB

![omphalos-dflash-k4q4-t0](omphalos-dflash-k4q4-t0.png)

## omphalos-dflash-k4q4-t0.5

31388 tokens, 469.2 s, prefill 289.2 t/s, decode 66.96 t/s, peak VRAM 14322 MiB

![omphalos-dflash-k4q4-t0.5](omphalos-dflash-k4q4-t0.5.png)

## omphalos-dflash-k4q4-t0.6

22305 tokens, 357.0 s, prefill 279.2 t/s, decode 62.54 t/s, peak VRAM 14195 MiB

![omphalos-dflash-k4q4-t0.6](omphalos-dflash-k4q4-t0.6.png)

## omphalos-dflash-k4q4-t1

22814 tokens, 482.6 s, prefill 281.8 t/s, decode 47.31 t/s, peak VRAM 14195 MiB

![omphalos-dflash-k4q4-t1](omphalos-dflash-k4q4-t1.png)

## omphalos-mtp-k4q4-t0

65396 tokens, 986.5 s, prefill 321.8 t/s, decode 66.32 t/s, peak VRAM 13572 MiB

(no SVG in the answer)

## omphalos-mtp-k4q4-t0.6

30899 tokens, 487.6 s, prefill 288.7 t/s, decode 63.43 t/s, peak VRAM 13490 MiB

![omphalos-mtp-k4q4-t0.6](omphalos-mtp-k4q4-t0.6.png)

## omphalos-mtp-k4q4-t1

56473 tokens, 1196.2 s, prefill 319.1 t/s, decode 47.22 t/s, peak VRAM 13572 MiB

![omphalos-mtp-k4q4-t1](omphalos-mtp-k4q4-t1.png)

## omphalos-mtp-t0

34373 tokens, 566.7 s, prefill 332.9 t/s, decode 60.7 t/s, peak VRAM 13832 MiB

![omphalos-mtp-t0](omphalos-mtp-t0.png)

## omphalos-mtp-t1

20506 tokens, 436.1 s, prefill 319.9 t/s, decode 47.05 t/s, peak VRAM 13841 MiB

![omphalos-mtp-t1](omphalos-mtp-t1.png)

## omphalos-plain-t0

34373 tokens, 1551.3 s, prefill 318.5 t/s, decode 22.16 t/s, peak VRAM 13162 MiB

![omphalos-plain-t0](omphalos-plain-t0.png)

## omphalos-plain-t1

30526 tokens, 1412.4 s, prefill 324.1 t/s, decode 21.62 t/s, peak VRAM 13162 MiB

![omphalos-plain-t1](omphalos-plain-t1.png)
