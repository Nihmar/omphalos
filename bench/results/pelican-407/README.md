# Pelican A/B: Qwen Sharp vs the GGUF template (#407, from #392 and #229)

The pelican task (an animated SVG of a pelican on a bicycle, #229) run with
both chat templates on the same model and decoding path:

- engine: main with #392 merged (`ab04032`; the feature commit is `8284ed1`),
  `engine/build/omph-generate`, RX 9060 XT, 2026-10-08;
- `--chat-template original|sharp`, `--dflash models/Qwen3.8-27B-DFlash2-Q4_K_M.omph`,
  `--temp 0.6 --top-p 0.95 --seed 229`, context 64k, thinking on, the run stops
  by itself (`end of generation`);
- **same `reasoning_effort` in each pair**: the GGUF template defaults to
  xhigh, Sharp to medium, so the pairs are xhigh (original default vs
  `reasoning_effort: "xhigh"` on Sharp) and medium (Sharp default vs
  `reasoning_effort: "medium"` on original). Comparing the defaults would mix
  template and effort.
- one seed per cell: the token counts of a long sampled reasoning run vary a
  lot between seeds, so these are indicative, not medians. The SVGs are the
  point of the exercise; a median over seeds would need 3x the runtime.

## Results (`summary.csv`)

| run | total | thinking | answer | wall | ms/token | drafts accepted | SVG |
|---|---|---|---|---|---|---|---|
| original, xhigh | 41719 | 37774 | 3944 | 463.5 s | 10.03 | 56.3 % | 8295 B, 61 shapes, 15 animations |
| sharp, xhigh | 15292 | 12546 | 2745 | 237.3 s | 15.38 | 36.7 % | 5218 B, 39 shapes, 10 animations |
| sharp, medium | 6007 | 3502 | 2504 | 72.9 s | 11.90 | 41.7 % | 5147 B, 41 shapes, 6 animations |
| original, medium | 4016 | 907 | 3108 | 55.4 s | 13.47 | 41.9 % | 6309 B, 51 shapes, 9 animations |

Reading:

- at **xhigh**, Sharp's terseness block is visible in the numbers: the thinking
  drops from 37 774 to 12 546 tokens (-67 %), the total from 41 719 to 15 292
  (-63 %), the wall time from 463.5 s to 237.3 s (-49 %). The SVG is smaller and
  simpler (39 vs 61 shapes, 10 vs 15 animations) but valid, animated and a
  recognisable pelican (`original-xhigh.png`, `sharp-xhigh.png`);
- at **medium** the two runs are close in total (6016 vs 4016) and here the
  original thought *less* (907 vs 3502) and answered more (3108 vs 2504): one
  seed each, and this is exactly where the run-to-run variance of the task
  lives, so nothing should be read into the medium order. What holds in both
  pairs is the SVG: Sharp draws a plain scene (flat sky/ground or plain
  background), the GGUF template adds decorations (sun, clouds, bushes,
  dashed road);
- DFlash2 accepted fewer drafts under Sharp at xhigh (36.7 % vs 56.3 %), so its
  ms/token is worse; the total time still halves because there are far fewer
  tokens. The acceptance tracks the text being generated, not the template's
  overhead.

## Files

Per run: `.txt` (thinking + answer, as generated), `.svg` (the answer's SVG),
`.png` (its first frame, `rsvg-convert -w 800`). `run.log` is the runner's log;
`summary.csv` is the table above. The runner is `tools/check_chat_template_ab.py`
for the short-prompt A/B; this pelican A/B was driven by the same
`omph-generate` invocation with the pelican prompt and `--ctx 65536` (the
default 8192 is not enough thinking room, which is what the first attempt hit).
