#!/usr/bin/env python3
"""M0 MTP A/B client: run fixed prompts against a local llama-server and append
one CSV row per prompt (decode t/s, prefill t/s, draft acceptance).

usage: uv run --no-project m0_mtp_client.py <port> <config-name> <out.csv>
"""

import csv
import json
import sys
import urllib.request
from pathlib import Path

N_PREDICT = 192
WARMUP_N = 8
PROMPTS = (
    ("code", "continue-code.txt"),
    ("prose", "prose.txt"),
    ("repetitive", "repetitive.txt"),
)


def complete(port: int, prompt: str, n_predict: int) -> dict:
    body = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": False,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read())


def make_row(config: str, name: str, timings: dict) -> dict:
    drafts = timings.get("draft_n") or 0
    accepted = timings.get("draft_n_accepted") or 0
    return {
        "config": config,
        "prompt": name,
        "pred_t_s": round(timings.get("predicted_per_second", 0.0), 2),
        "prompt_t_s": round(timings.get("prompt_per_second", 0.0), 2),
        "prompt_ms": round(timings.get("prompt_ms", 0.0), 1),
        "predicted_n": timings.get("predicted_n", 0),
        "draft_n": drafts,
        "draft_accepted": accepted,
        "acceptance_pct": round(100.0 * accepted / drafts, 1) if drafts else 0.0,
    }


def main() -> None:
    port, config, out_path = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    prompts_dir = Path(__file__).resolve().parent / "prompts"

    warmup_prompt = (prompts_dir / PROMPTS[0][1]).read_text(encoding="utf-8")
    complete(port, warmup_prompt, WARMUP_N)  # warmup, discarded

    rows = []
    for name, fname in PROMPTS:
        text = (prompts_dir / fname).read_text(encoding="utf-8")
        rows.append(make_row(config, name, complete(port, text, N_PREDICT)["timings"]))

    new_file = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
