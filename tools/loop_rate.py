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
import sys
import urllib.error
from pathlib import Path

import pi_session
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
    "k4-all": {"env": {"OMPH_KV_K4": "1"}},  # the configuration the 2026-10-04 session ran (#81)

    "k8-sampled": {"env": {"OMPH_KV_K4_LAYERS": "none"},
                   "argv": ["--temp", "0.6", "--top-p", "0.95", "--top-k", "20"]},
    "k4-all-sampled": {"env": {"OMPH_KV_K4": "1"},
                       "argv": ["--temp", "0.6", "--top-p", "0.95", "--top-k", "20"]},
    "nothink": {"request": {"chat_template_kwargs": {"enable_thinking": False}}},
    "no-ngram": {"env": {"OMPH_NGRAM": "0"}},
    # llama.cpp's penalties sampler (#298), over the sampler recipe of #285
    "pen-llama": {"argv": ["--temp", "0.6", "--top-p", "0.95", "--top-k", "20",
                           "--repeat-penalty", "1.1", "--repeat-last-n", "1024"]},
    "pen-short": {"argv": ["--temp", "0.6", "--top-p", "0.95", "--top-k", "20",
                           "--repeat-penalty", "1.1", "--repeat-last-n", "64"]},
    "pen-openai": {"argv": ["--temp", "0.6", "--top-p", "0.95", "--top-k", "20",
                            "--frequency-penalty", "0.2", "--presence-penalty", "0.5",
                            "--repeat-last-n", "1024"]},
}


def paragraph_spans(text: str, skip_fences: bool) -> list[tuple[int, str]]:
    """Blank-line separated paragraphs and their offsets, as loop-police's
    semantic detector sees them; `skip_fences` drops the ones inside or
    containing ``` (its rule, and an unbalanced fence then silences the rest of
    the stream)."""
    out: list[tuple[int, str]] = []
    in_fence = False
    at = 0
    for block in re.split(r"\r?\n[ \t]*\r?\n", text):
        marks = block.count("```")
        if not (skip_fences and in_fence) and not (skip_fences and marks) and block.strip():
            out.append((at + len(block) - len(block.lstrip()), block.strip()))
        at += len(block) + 2
        if marks % 2 == 1:
            in_fence = not in_fence
    return out


def paragraphs(text: str, skip_fences: bool) -> list[str]:
    return [p for _, p in paragraph_spans(text, skip_fences)]


def pair_counts(text: str, para_min: int, floor: float = 0.85) -> tuple[int, int, float]:
    """(paragraph pairs at or above `floor`, of them verbatim, the worst ratio):
    what a fingerprint rule misses when the model re-derives the same plan in
    slightly different words."""
    paras = [p[:1500] for p in paragraphs(text, True) if len(p) >= para_min]
    matcher = difflib.SequenceMatcher(None)
    pairs = verbatim = 0
    worst = 0.0
    for i in range(len(paras)):
        for j in range(i + 1, len(paras)):
            matcher.set_seqs(paras[i], paras[j])
            ratio = matcher.ratio()
            worst = max(worst, ratio)
            if ratio >= floor:
                pairs += 1
                verbatim += ratio >= 0.999
    return pairs, verbatim, worst


def first_firing(text: str, para_min: int, fp_len: int, threshold: int,
                 skip_fences: bool = True) -> int | None:
    """Where loop-police would truncate: the offset of the paragraph whose
    fingerprint reaches `threshold`, or None. Its escalation aids are ignored
    (the truncation is what a violation costs)."""
    counts: dict[str, int] = {}
    for at, block in paragraph_spans(text, skip_fences):
        if len(block) < para_min:
            continue
        fp = re.sub(r"^\d+([.)])\s+", r"#\1 ", block)[:fp_len]
        counts[fp] = counts.get(fp, 0) + 1
        if counts[fp] >= threshold:
            return at
    return None


# loop-police's defaults and the settings the postmortem suggests (a longer
# paragraph floor and fingerprint, a higher threshold): the sweep reports, per
# stream, how much of it each setting would have let through (#287).
SWEEP = [(3, 40, 60), (3, 160, 60), (5, 40, 60), (5, 160, 60), (3, 40, 120), (5, 160, 120)]


def sweep(paths: list[Path], out: Path | None = None) -> None:
    """Per stream: how much the model repeated and where loop-police's semantic
    detector would have truncated it under each setting of SWEEP."""
    head = f"{'stream':44s} {'pairs':>6s} {'verb':>5s} {'worst':>6s} " + \
        " ".join(f"t{t}/p{p}/f{f}".rjust(11) for t, p, f in SWEEP)
    print(head)
    rows = []
    for path in sorted(paths):
        text = path.read_text(errors="replace")
        pairs, verbatim, worst = pair_counts(text, 40)
        cells = []
        row = {"stream": path.name, "chars": len(text), "pairs_085": pairs, "verbatim": verbatim,
               "worst_pair": round(worst, 3)}
        for threshold, para_min, fp_len in SWEEP:
            at = first_firing(text, para_min, fp_len, threshold)
            cells.append("--" if at is None else f"{100.0 * at / max(len(text), 1):5.0f}%")
            row[f"first_firing_t{threshold}_p{para_min}_f{fp_len}"] = "" if at is None else at
        rows.append(row)
        print(f"{path.name:44s} {pairs:6d} {verbatim:5d} {worst:6.3f} " + " ".join(c.rjust(11) for c in cells))
    print("cells: the offset of the first truncation as a share of the stream (--: none); "
          "pairs: paragraphs at 0.85 similarity or more")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["stream"])
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"wrote {out}")


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


def write_csv(out: Path, rows: list[dict], commit: str) -> None:
    """The CSV accumulates: one file per experiment, the commit in every row,
    rewritten after every run so a killed matrix keeps what it measured."""
    out.parent.mkdir(parents=True, exist_ok=True)
    previous = []
    if out.exists():
        with out.open(newline="") as f:
            fresh = {(row["config"], row["prompt"]) for row in rows}
            previous = [r for r in csv.DictReader(f) if r and (r.get("config"), r.get("prompt")) not in fresh]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["commit", "config", "prompt", "prompt_tokens", "reasoning_chars",
                                          "completion_tokens", "paragraphs", "paragraphs_skipped",
                                          "repetitions", "repetitions_anywhere",
                                          "repeats_per_1k_tokens", "worst_pair_similarity"])
        w.writeheader()
        for row in previous:
            w.writerow(row)
        for row in rows:
            w.writerow({"commit": commit, **row})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="greedy,sampled,k8")
    ap.add_argument("--prompts", default=None, help="a JSON list of {system?, user} or {messages, tools?}")
    ap.add_argument("--pi-session", default=None,
                    help="a pi session JSONL: one prompt per --cuts timestamp, rebuilt as pi sent it (#287)")
    ap.add_argument("--cuts", default=None,
                    help="comma-separated ISO timestamps (as in the session file): the turn that starts there")
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=None, help="a fixed sampling seed (the sampler configs)")
    ap.add_argument("--para-min-len", type=int, default=40)
    ap.add_argument("--fingerprint-len", type=int, default=60)
    ap.add_argument("--threshold", type=int, default=3)
    ap.add_argument("--out", default=None, help="CSV path (bench/results/...), tagged with the commit")
    ap.add_argument("--dump", default=None, help="directory for each run's reasoning (a .txt per run)")
    ap.add_argument("--sweep", default=None,
                    help="do not run anything: score the .txt files of this directory with a grid of "
                         "loop-police settings and print where each would truncate")
    args = ap.parse_args()

    if args.sweep:
        sweep(sorted(Path(args.sweep).glob("*.txt")), Path(args.out) if args.out else None)
        return

    if args.pi_session:
        recs = pi_session.records(args.pi_session)
        prompts = []
        for cut in (args.cuts or "").split(","):
            req = pi_session.request_for_cut(recs, cut)
            req["label"] = cut
            prompts.append(req)
    else:
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
                label = p.get("label", f"p{k}")
                if "messages" in p:
                    body = {"max_tokens": args.max_tokens, "stream": False,
                            "chat_template_kwargs": {"enable_thinking": True},
                            **{key: p[key] for key in ("messages", "tools") if key in p},
                            **cfg.get("request", {})}
                else:
                    msgs = ([{"role": "system", "content": p["system"]}] if p.get("system") else []) + \
                        [{"role": "user", "content": p["user"]}]
                    body = {"messages": msgs, "max_tokens": args.max_tokens, "stream": False,
                            "chat_template_kwargs": {"enable_thinking": True}, **cfg.get("request", {})}
                if args.seed is not None:
                    body["seed"] = args.seed
                try:
                    r = post(url + "/v1/chat/completions", body)
                except urllib.error.HTTPError as e:
                    body_text = e.read().decode(errors="replace")
                    sys.exit(f"{name} {label}: the server answered {e.code}: {body_text[:400]}")
                m = r["choices"][0]["message"]
                reasoning = m.get("reasoning_content") or ""
                content = m.get("content") or ""
                text = reasoning or content
                repeats, worst, paras, skipped = score(text, args.para_min_len, args.fingerprint_len,
                                                       args.threshold, True)
                repeats_any = score(text, args.para_min_len, args.fingerprint_len, args.threshold, False)[0]
                tokens = (r["usage"].get("completion_tokens") or 0)
                # repetitions per 1000 generated tokens (was written per million, #348)
                per_k = repeats / (tokens / 1000.0) if tokens else 0.0
                rows.append({"config": name, "prompt": label, "prompt_tokens": (r["usage"].get("prompt_tokens") or 0),
                             "reasoning_chars": len(reasoning),
                             "completion_tokens": tokens, "paragraphs": paras, "paragraphs_skipped": skipped,
                             "repetitions": repeats, "repetitions_anywhere": repeats_any,
                             "repeats_per_1k_tokens": round(per_k, 3), "worst_pair_similarity": round(worst, 3)})
                if args.out:
                    write_csv(Path(args.out), rows, commit)
                if args.dump:
                    d = Path(args.dump)
                    d.mkdir(parents=True, exist_ok=True)
                    (d / f"{name}-{label}.txt").write_text(text)
                print(f"{name:12s} {label}: {tokens:5d} tokens, {paras:3d} paragraphs ({skipped} fence-skipped), "
                      f"{repeats:2d}/{repeats_any:2d} repetitions, worst pair {worst:.3f}", flush=True)
        finally:
            proc.terminate()
            proc.wait()
    if args.out:
        write_csv(Path(args.out), rows, commit)
    fail = [r for r in rows if r["repetitions"] > 0]
    print(f"{len(rows)} runs, {len(fail)} with a repetition" +
          (f" ({', '.join(sorted({r['config'] for r in fail}))})" if fail else ""))


if __name__ == "__main__":
    main()
