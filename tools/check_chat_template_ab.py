#!/usr/bin/env python3
"""A/B the two chat templates end-to-end (#392).

The same prompts, seeds, temperature and speculative path, rendered once with
the GGUF's own template (--chat-template original) and once with the vendored
Qwen Sharp one (--chat-template sharp): the byte-for-byte comparison is
check_chat_template.py, this is what the engine generates with them.

    uv run python check_chat_template_ab.py --dflash ../models/Qwen3.8-27B-DFlash2-Q4_K_M.omph \\
        --temp 0.6 --seeds 1,2,3

Each run reports the tokens generated, ms/token and drafts accepted, and
whether the answer carries the prompt's expected marker (a sampling miss is a
warning, not a failure). The summary is the median ms/token and acceptance per
template, and the terse-ness (mean completion tokens) of the two prompts sets.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

PROMPTS = [
    {"name": "math", "user": "What is 17*19? Answer with just the number.", "expect": "323"},
    {"name": "exact", "user": "Reply with exactly this word and nothing else: WALRUS", "expect": "WALRUS"},
    {"name": "code", "user": "Write a Python function named is_even(n) that returns True for even n. "
                             "No explanation.", "expect": "def is_even"},
    {"name": "prose", "user": "In one short sentence, what does a lighthouse keeper do during a storm?",
     "expect": None},
    {"name": "long-code", "user": "Write a complete C function that reverses a singly linked list. "
                                       "Include the struct and short comments.", "expect": "next"},
    {"name": "long-prose", "user": "Explain in three paragraphs how a modern city water system works, "
                                        "from the source to the tap.", "expect": "water"},
    {"name": "tool", "user": "What is the weather in Rome? Use the tool.",
     "tools": [{"type": "function", "function": {"name": "get_weather",
                                                 "parameters": {"type": "object",
                                                                "properties": {"city": {"type": "string"}},
                                                                "required": ["city"]}}}],
     "expect": "get_weather"},
]
SUMMARY = re.compile(r"(\d+) tokens in ([0-9.]+) ms \(([0-9.]+) ms/token, ([0-9.]+) t/s\); "
                     r"drafts (\d+) / (\d+) accepted")


def run(gen: str, model: str, template: str, prompt: dict, temp: float, seed: int, max_tokens: int,
        dflash: str) -> dict:
    req = {"messages": [{"role": "user", "content": prompt["user"]}], "enable_thinking": False,
           "add_generation_prompt": True}
    if "tools" in prompt:
        req["tools"] = prompt["tools"]
    cmd = [gen, model, "--chat", "--chat-template", template, "--max", str(max_tokens), "--temp", str(temp),
           "--top-p", "0.95", "--seed", str(seed)]
    if dflash:
        cmd += ["--dflash", dflash]
    p = subprocess.run(cmd, input=json.dumps(req).encode(), capture_output=True, check=False)
    text = p.stdout.decode(errors="replace")
    lines = [ln for ln in p.stderr.decode(errors="replace").splitlines() if "ms/token" in ln]
    if p.returncode != 0 or not lines:
        sys.exit(f"omph-generate failed for {template}/{prompt['name']}/{seed}:\n{p.stderr.decode()[-800:]}")
    m = SUMMARY.search(lines[-1])
    stop = lines[-1].split("stop: ")[-1].split(" (client gone)")[0]
    expect = prompt["expect"]
    return {"template": template, "prompt": prompt["name"], "seed": seed,
            "tokens": int(m.group(1)), "ms_per_token": float(m.group(3)),
            "accepted": int(m.group(5)), "drafted": int(m.group(6)), "stop": stop,
            "ok": expect is None or expect in text, "text": text.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.omph"))
    ap.add_argument("--gen", default=str(ROOT / "engine/build/omph-generate"))
    ap.add_argument("--dflash", default="", help="a DFlash2 drafter .omph (default: the MTP block)")
    ap.add_argument("--templates", default="original,sharp")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--out", default="", help="CSV path")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for template in args.templates.split(","):
        for prompt in PROMPTS:
            for seed in seeds:
                r = run(args.gen, args.model, template, prompt, args.temp, seed, args.max_tokens, args.dflash)
                rows.append(r)
                print(f"  {template:8s} {r['prompt']:6s} seed {seed}: {r['tokens']:4d} tokens, "
                      f"{r['ms_per_token']:6.2f} ms/token, {r['accepted']:3d}/{r['drafted']:3d} drafts"
                      f"{'' if r['ok'] else '  MISSING ' + str(prompt['expect'])}", flush=True)
    print()
    for template in args.templates.split(","):
        got = [r for r in rows if r["template"] == template]
        acc = sum(r["accepted"] for r in got)
        dr = sum(r["drafted"] for r in got)
        print(f"{template:8s}: median {statistics.median(r['ms_per_token'] for r in got):6.2f} ms/token, "
              f"{100.0 * acc / dr if dr else 0:5.1f} % accepted, "
              f"mean {statistics.mean(r['tokens'] for r in got):5.1f} tokens, "
              f"{sum(r['ok'] for r in got)}/{len(got)} markers")
    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["template", "prompt", "seed", "tokens", "ms_per_token",
                                              "accepted", "drafted", "stop", "ok"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in w.fieldnames})


if __name__ == "__main__":
    main()
