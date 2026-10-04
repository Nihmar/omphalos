"""Repeat rate of a serving configuration on fixed long-reasoning prompts (#287).

The measurement behind the loop-police post-mortem of 2026-10-04: the model
kept re-deriving the same reasoning in pi's thinking stream, so the question
"which serving configuration repeats, and how much" needs a number instead of
a transcript. This runs a fixed prompt set through omph-server once per
configuration and scores the streamed reasoning with loop-police's own
semantic detector, so the score is comparable with what the extension fired
on: paragraphs separated by blank lines and at least PARA_MIN_LEN chars long
are fingerprinted by their first FINGERPRINT_LEN chars (a leading ordered-list
counter is normalized away); a fingerprint seen SEMANTIC_THRESHOLD times is a
repetition. Paragraphs inside ``` fences are skipped, as loop-police does.
Also reported: the worst paragraph pair of the run (difflib ratio), so a
verbatim loop can be told from the model re-deriving the same plan, and the
generated tokens.

    uv run python loop_rate.py [--configs greedy,sampled,k8] [--max-tokens N]
        [--prompts FILE] [--out bench/results/loop-rate-287.csv]

The configurations are server options (`--temp 0.6 --top-p 0.95 --top-k 20`,
`OMPH_KV_K4_LAYERS=none`, `OMPH_NGRAM=0`); one server (and model load) each.
The prompts default to a built-in set, the CommonMark list-item-interruption
deliberation that looped in the session. A JSON list of {"system", "user"}
replaces it with --prompts.
"""

import argparse
import csv
import difflib
import json
import os
import re
import subprocess
from pathlib import Path

from niah import free_port, post, wait_health
from omph_model import omph_file

ROOT = Path(__file__).resolve().parent.parent

# The failure mode of the 2026-10-04 session: one underspecified rule, many
# plausible readings, and a model that re-checks them in the same words.
PROMPTS = [
    {
        "system": None,
        "user": (
            "A Markdown block scanner must report, for every line, its block kind (paragraph, list "
            "item, heading, quote, code) and its list depth, following CommonMark 0.31.2. Work "
            "through these five inputs line by line and state the expected kind and depth of every "
            "line, citing the rule that decides it: `- w\\n  2) w`, `w\\n2) w\\n---`, "
            "`- w\\n  ```\\n* w`, `- foo\\n   # bar\\nbaz`, `- w\\n   2) w`. Then say which of the "
            "five you are least sure about and why."
        ),
    },
    {
        "system": None,
        "user": (
            "Design a compaction policy for a coding agent that must summarise a 100k-token "
            "conversation without losing a decision: what is kept verbatim, what is summarised, "
            "where the boundaries go, and how the policy avoids re-deriving the same plan on the "
            "next turn. Enumerate the alternatives, pick one, and justify it."
        ),
    },
]

CONFIGS = {
    "greedy": {},
    "sampled": {"argv": ["--temp", "0.6", "--top-p", "0.95", "--top-k", "20"]},
    "k8": {"env": {"OMPH_KV_K4_LAYERS": "none"}},
    "k8-sampled": {"env": {"OMPH_KV_K4_LAYERS": "none"},
                   "argv": ["--temp", "0.6", "--top-p", "0.95", "--top-k", "20"]},
    "nothink": {"request": {"chat_template_kwargs": {"enable_thinking": False}}},
    "no-ngram": {"env": {"OMPH_NGRAM": "0"}},
}


def paragraphs(text: str, skip_fences: bool) -> list[str]:
    """Blank-line separated paragraphs, as loop-police's semantic detector sees
    them; `skip_fences` drops the ones inside or containing ``` (its rule, and
    an unbalanced fence then silences the rest of the stream)."""
    out: list[str] = []
    in_fence = False
    for block in re.split(r"\r?\n[ \t]*\r?\n", text):
        marks = block.count("```")
        if not (skip_fences and in_fence) and not (skip_fences and marks) and block.strip():
            out.append(block.strip())
        if marks % 2 == 1:
            in_fence = not in_fence
    return out


def score(text: str, para_min: int, fp_len: int, threshold: int,
          skip_fences: bool) -> tuple[int, int, int, float]:
    """Fingerprint repetitions, the worst paragraph pair similarity, and the
    paragraphs scored: how many, and how many were skipped by the fence rule."""
    paras = [p for p in paragraphs(text, skip_fences) if len(p) >= para_min]
    counts: dict[str, int] = {}
    repeats = 0
    for p in paras:
        fp = re.sub(r"^\d+([.)])\s+", r"#\1 ", p)[:fp_len]
        counts[fp] = counts.get(fp, 0) + 1
        if counts[fp] == threshold:
            repeats += 1
    worst = 0.0
    if len(paras) > 1:
        short = [p[:2000] for p in paras]
        matcher = difflib.SequenceMatcher(None)
        for i in range(len(short)):
            for j in range(i + 1, len(short)):
                if min(len(short[i]), len(short[j])) < para_min:
                    continue
                matcher.set_seqs(short[i], short[j])
                worst = max(worst, matcher.ratio())
    skipped = len([p for p in paragraphs(text, False) if len(p) >= para_min]) - len(paras)
    return repeats, worst, len(paras), skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="greedy,sampled,k8")
    ap.add_argument("--prompts", default=None, help="a JSON list of {system?, user}")
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--para-min-len", type=int, default=40)
    ap.add_argument("--fingerprint-len", type=int, default=60)
    ap.add_argument("--threshold", type=int, default=3)
    ap.add_argument("--out", default=None, help="CSV path (bench/results/...), tagged with the commit")
    ap.add_argument("--dump", default=None, help="directory for each run's reasoning (a .txt per run)")
    args = ap.parse_args()

    prompts = json.loads(Path(args.prompts).read_text()) if args.prompts else PROMPTS
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, check=False).stdout.strip()
    rows = []
    for name in args.configs.split(","):
        cfg = CONFIGS[name]
        port = free_port()
        cmd = [f"{args.omph}/omph-server", omph_file(args.model), "--port", str(port), "--ctx",
               str(args.ctx), *cfg.get("argv", [])]
        env = {**os.environ, **cfg["env"]} if cfg.get("env") else None
        log = Path(f"/tmp/loop-rate-{name}.log")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(log, "w"), env=env)  # noqa: SIM115
        url = f"http://127.0.0.1:{port}"
        try:
            wait_health(url, proc)
            for k, p in enumerate(prompts):
                msgs = ([{"role": "system", "content": p["system"]}] if p.get("system") else []) + \
                    [{"role": "user", "content": p["user"]}]
                body = {"messages": msgs, "max_tokens": args.max_tokens, "stream": False,
                        "chat_template_kwargs": {"enable_thinking": True}, **cfg.get("request", {})}
                r = post(url + "/v1/chat/completions", body)
                m = r["choices"][0]["message"]
                reasoning = m.get("reasoning_content") or ""
                content = m.get("content") or ""
                text = reasoning or content
                repeats, worst, paras, skipped = score(text, args.para_min_len, args.fingerprint_len,
                                                       args.threshold, True)
                repeats_any = score(text, args.para_min_len, args.fingerprint_len, args.threshold, False)[0]
                tokens = (r["usage"].get("completion_tokens") or 0)
                per_k = 1000.0 * repeats / (tokens / 1000.0) if tokens else 0.0
                rows.append({"config": name, "prompt": k, "reasoning_chars": len(reasoning),
                             "completion_tokens": tokens, "paragraphs": paras, "paragraphs_skipped": skipped,
                             "repetitions": repeats, "repetitions_anywhere": repeats_any,
                             "repeats_per_1k_tokens": round(per_k, 3), "worst_pair_similarity": round(worst, 3)})
                if args.dump:
                    d = Path(args.dump)
                    d.mkdir(parents=True, exist_ok=True)
                    (d / f"{name}-p{k}.txt").write_text(text)
                print(f"{name:12s} p{k}: {tokens:5d} tokens, {paras:3d} paragraphs ({skipped} fence-skipped), "
                      f"{repeats:2d}/{repeats_any:2d} repetitions, worst pair {worst:.3f}", flush=True)
        finally:
            proc.terminate()
            proc.wait()
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["commit", "config", "prompt", "reasoning_chars",
                                              "completion_tokens", "paragraphs", "paragraphs_skipped",
                                              "repetitions", "repetitions_anywhere",
                                              "repeats_per_1k_tokens", "worst_pair_similarity"])
            w.writeheader()
            for row in rows:
                w.writerow({"commit": commit, **row})
        print(f"wrote {out}")
    fail = [r for r in rows if r["repetitions"] > 0]
    print(f"{len(rows)} runs, {len(fail)} with a repetition" +
          (f" ({', '.join(sorted({r['config'] for r in fail}))})" if fail else ""))


if __name__ == "__main__":
    main()
