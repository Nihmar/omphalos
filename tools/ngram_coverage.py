"""Offline n-gram draft coverage (#199): how much of a generated token stream
an n-gram lookup over the context would have drafted correctly.

The decode is replayed step by step. At each step the last n tokens of the
context (prompt + generated so far) are looked up at their latest earlier
occurrence; the m tokens that followed it are the draft. The drafted tokens
that match the actual continuation are accepted, plus the verification's own
token. With no match the step is an MTP step (counted separately: its
acceptance is not modelled here).

    uv run python ngram_coverage.py humaneval ../he-omphalos.jsonl     # prompt + final answer
    uv run python ngram_coverage.py ids PROMPT_IDS GENERATED_IDS        # token id files
"""

import argparse
import gzip
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tokenize(tok: str, model: str, text: str, special: bool) -> list[int]:
    cmd = [tok, model] + ([] if special else ["--no-parse-special"])
    return [int(x) for x in subprocess.run(cmd, input=text, capture_output=True, text=True, check=True).stdout.split()]


def replay(prompt: list[int], gen: list[int], n: int, m: int) -> dict:
    ctx = list(prompt)
    last = {}  # n-gram -> end index (exclusive) of its latest occurrence in ctx
    for j in range(n, len(ctx) + 1):
        last[tuple(ctx[j - n:j])] = j

    def push(t: int) -> None:
        ctx.append(t)
        if len(ctx) >= n + 1:
            # the n-gram ending just before the new token can now be followed by it
            key = tuple(ctx[-n - 1:-1])
            last[key] = len(ctx) - 1

    i = 0
    steps = ngram_steps = drafted_ok = 0
    while i < len(gen):
        key = tuple(ctx[-n:]) if len(ctx) >= n else None
        end = last.get(key) if key is not None else None
        steps += 1
        if end is not None and end < len(ctx):
            draft = ctx[end:end + m]
            a = 0
            while a < len(draft) and i + a < len(gen) and draft[a] == gen[i + a]:
                a += 1
            ngram_steps += 1
            drafted_ok += a
            take = a + 1  # the accepted drafts and the verification's own token
        else:
            take = 1  # an MTP step: its drafts are not modelled
        for t in gen[i:i + take]:
            push(t)
        i += take
    return {"tokens": len(gen), "steps": steps, "ngram_steps": ngram_steps, "drafted_ok": drafted_ok}


def report(name: str, rows: list[dict], n: int, m: int) -> None:
    tok = sum(r["tokens"] for r in rows)
    ok = sum(r["drafted_ok"] for r in rows)
    ns = sum(r["ngram_steps"] for r in rows)
    st = sum(r["steps"] for r in rows)
    print(f"{name:10s} n={n} m={m:2d}: {tok:6d} tokens, {100 * ok / max(tok, 1):5.1f} % drafted correctly by n-grams, "
          f"{100 * ns / max(st, 1):5.1f} % of steps had a match, {(ok + ns) / max(ns, 1):5.2f} tokens per n-gram step")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["humaneval", "ids"])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    ap.add_argument("--data", default=str(ROOT / "models/datasets/human-eval/data/HumanEval.jsonl.gz"))
    args = ap.parse_args()
    tok = f"{args.omph}/omph-tokenize"
    pairs = []
    if args.mode == "humaneval":
        from humaneval import render
        with gzip.open(args.data, "rt") as f:
            problems = {p["task_id"]: p for p in map(json.loads, f)}
        with open(args.files[0]) as f:
            for row in map(json.loads, f):
                prompt = render(tok, args.model, problems[row["task_id"]]["prompt"])
                pairs.append((tokenize(tok, args.model, prompt, True), tokenize(tok, args.model, row["answer"], False)))
    else:
        read = [[int(x) for x in Path(p).read_text().split()] for p in args.files[:2]]
        pairs.append((read[0], read[1]))
    for n in (2, 3, 4):
        for m in (4, 8, 16):
            report(args.mode, [replay(p, g, n, m) for p, g in pairs], n, m)


if __name__ == "__main__":
    main()
