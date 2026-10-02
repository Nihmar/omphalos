"""HumanEval pass@1, omphalos vs llama.cpp (#195).

Each of the 164 problems is one user message ("complete this function", the
prompt in a ```python block), rendered once with the GGUF's chat template by
omph-tokenize (thinking on, reasoning effort xhigh: the template's default) and
sent as the same raw prompt to either engine's completion endpoint, greedy:

1. up to --think-budget tokens (default 8192) of generation from the open
   <think>;
2. if the thinking has not closed by then, "</think>" is appended and the
   answer is generated from there (up to --answer-tokens); if it closed but
   the answer was cut by the budget, the answer is continued the same way.

The answer's last ```python block (or the whole answer) is the completion;
the program run is the problem's prompt, the completion, the tests and
check(entry_point). Every program runs in a bubblewrap sandbox: no network,
the filesystem read-only except a private /tmp, a --timeout per problem.

    uv run python humaneval.py --engine omphalos --out he-omph.jsonl
    uv run python humaneval.py --engine llama --out he-llama.jsonl \\
        --llama-server <llama.cpp>/build-hip/bin/llama-server

Results: one JSON line per problem appended to --out (a rerun skips the
problems already there), then pass@1 and token / speed totals.
"""

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from niah import free_port, post, wait_health
from omph_model import omph_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INSTRUCTION = ("Complete the following Python function. Reply with the complete function (with any imports "
               "it needs) in a single ```python code block.\n\n```python\n{}```")
THINK_END = "</think>"


def render(tokenize: str, model: str, prompt: str) -> str:
    req = {"messages": [{"role": "user", "content": INSTRUCTION.format(prompt)}], "add_generation_prompt": True}
    return subprocess.run([tokenize, model, "--chat"], input=json.dumps(req), capture_output=True, text=True,
                          check=True).stdout


def complete(engine: str, url: str, prompt: str, max_tokens: int) -> tuple[str, bool, dict]:
    """The generated text, whether it stopped on the token limit, and the timings."""
    if engine == "omphalos":
        r = post(url + "/v1/completions", {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0})
        text, tm = r["choices"][0]["text"], r["timings"]
        limit = r["choices"][0].get("finish_reason") == "length"
    else:
        r = post(url + "/completion", {"prompt": prompt, "n_predict": max_tokens, "temperature": 0, "top_k": 1,
                                       "cache_prompt": True})
        text, tm = r["content"], r["timings"]
        limit = bool(r.get("stopped_limit")) or r.get("stop_type") == "limit"
    return text, limit, {"prompt_n": tm["prompt_n"], "prompt_ms": tm["prompt_ms"], "predicted_n": tm["predicted_n"],
                         "predicted_ms": tm["predicted_ms"]}


def solve(engine: str, url: str, prompt: str, budget: int, answer_tokens: int) -> dict:
    t0 = time.time()
    out, limit, tm = complete(engine, url, prompt, budget)
    timings = [tm]
    capped = False
    if THINK_END not in out and limit:
        capped = True  # the thinking budget ran out: close it and ask for the answer
        out += "\n" + THINK_END + "\n\n"
        more, limit, tm = complete(engine, url, prompt + out, answer_tokens)
        out += more
        timings.append(tm)
    elif limit:
        more, limit, tm = complete(engine, url, prompt + out, answer_tokens)  # the answer was cut: continue it
        out += more
        timings.append(tm)
    thinking, _, answer = out.partition(THINK_END)
    return {"thinking": thinking, "answer": answer, "capped": capped, "answer_cut": limit,
            "predicted_n": sum(t["predicted_n"] for t in timings),
            "predicted_ms": sum(t["predicted_ms"] for t in timings),
            "prompt_n": sum(t["prompt_n"] for t in timings), "prompt_ms": sum(t["prompt_ms"] for t in timings),
            "wall_s": round(time.time() - t0, 1)}


def extract(answer: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", answer, re.DOTALL)
    return blocks[-1] if blocks else answer


def run_sandboxed(program: str, timeout: float) -> tuple[bool, str]:
    """Runs the program in bubblewrap: no network, read-only filesystem, private /tmp."""
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "prog.py"
        src.write_text(program)
        cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
               "--ro-bind", str(src), "/tmp/prog.py", "--unshare-all", "--die-with-parent", "--new-session",
               "--chdir", "/tmp", sys.executable, "-I", "/tmp/prog.py"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return p.returncode == 0, (p.stderr or "")[-400:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["omphalos", "llama"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--think-budget", type=int, default=8192)
    ap.add_argument("--answer-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=20.0, help="seconds per test program")
    ap.add_argument("--limit", type=int, default=0, help="only the first N problems")
    ap.add_argument("--tasks", default="", help="only these problem numbers, e.g. 47,145,158")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    ap.add_argument("--omph-env", default="", help="extra environment for omph-server, e.g. OMPH_KV_K4_LAYERS=none")
    ap.add_argument("--llama-server", default="")
    ap.add_argument("--data", default=str(ROOT / "models/datasets/human-eval/data/HumanEval.jsonl.gz"))
    args = ap.parse_args()

    with gzip.open(args.data, "rt") as f:
        problems = [json.loads(line) for line in f]
    if args.limit:
        problems = problems[:args.limit]
    if args.tasks:
        wanted = {f"HumanEval/{t.strip()}" for t in args.tasks.split(",")}
        problems = [p for p in problems if p["task_id"] in wanted]
    done = set()
    if Path(args.out).exists():
        with open(args.out) as f:
            done = {json.loads(line)["task_id"] for line in f}
    tokenize = f"{args.omph}/omph-tokenize"
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    if args.engine == "omphalos":
        if args.omph_env:
            name, _, value = args.omph_env.partition("=")
            env[name] = value
        cmd = [f"{args.omph}/omph-server", omph_file(args.model), "--port", str(port), "--ctx", str(args.ctx)]
    else:
        if not args.llama_server:
            sys.exit("--llama-server is required for --engine llama")
        cmd = [args.llama_server, "-m", args.model, "--port", str(port), "-c", str(args.ctx), "-ngl", "999",
               "-fa", "on", "-ctk", "q8_0", "-ctv", "q4_0", "-np", "1", "--no-webui"]
    log = open(f"{args.out}.server.log", "w")  # noqa: SIM115 (open while the server runs)
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_health(url, proc)
        for i, pb in enumerate(problems):
            if pb["task_id"] in done:
                continue
            prompt = render(tokenize, args.model, pb["prompt"])
            res = solve(args.engine, url, prompt, args.think_budget, args.answer_tokens)
            code = extract(res["answer"])
            program = f"{pb['prompt']}\n\n{code}\n\n{pb['test']}\n\ncheck({pb['entry_point']})\n"
            ok, err = run_sandboxed(program, args.timeout)
            row = {"task_id": pb["task_id"], "engine": args.engine, "omph_env": args.omph_env, "passed": ok,
                   "error": "" if ok else err, "capped": res["capped"], "answer_cut": res["answer_cut"],
                   "thinking_chars": len(res["thinking"]), "completion": code, "answer": res["answer"],
                   "thinking": res["thinking"],
                   "predicted_n": res["predicted_n"], "predicted_ms": round(res["predicted_ms"], 1),
                   "prompt_n": res["prompt_n"], "prompt_ms": round(res["prompt_ms"], 1), "wall_s": res["wall_s"]}
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            tps = res["predicted_n"] / res["predicted_ms"] * 1000 if res["predicted_ms"] > 0 else 0
            print(f"[{i + 1:3d}/{len(problems)}] {pb['task_id']:14s} {'pass' if ok else 'FAIL'} "
                  f"{res['predicted_n']:5d} tok {tps:5.1f} t/s{' capped' if res['capped'] else ''}"
                  f"{' cut' if res['answer_cut'] else ''}", flush=True)
    finally:
        proc.terminate()
        proc.wait(60)
        log.close()
    with open(args.out) as f:
        rows = [json.loads(line) for line in f]
    n = len(rows)
    passed = sum(r["passed"] for r in rows)
    tok = sum(r["predicted_n"] for r in rows)
    ms = sum(r["predicted_ms"] for r in rows)
    print(f"{args.engine}: pass@1 {passed} / {n} = {100 * passed / max(n, 1):.1f} %, "
          f"{sum(r['capped'] for r in rows)} thinking capped, {sum(r['answer_cut'] for r in rows)} answers cut, "
          f"{tok} tokens generated at {tok / ms * 1000 if ms else 0:.1f} t/s")


if __name__ == "__main__":
    main()
