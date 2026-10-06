"""Time to first token through the servers: omphalos vs llama.cpp (#233).

What a user waits for, not what the engine's prefill takes: a rendered prompt
(wikitext text in a chat template, rendered once with omph-tokenize, so both
engines see the same text) is sent to either server's completion endpoint with
one token to generate and prompt caching off, streamed. The time from the POST
to the first non-empty SSE delta is the TTFT -- tokenization, the chat
template, the HTTP round trip and the prefill -- and the final chunk's
`timings.prompt_ms` splits the engine's own prefill out of it (the difference
is the server's overhead). Median of --runs runs after --warmup, per size
(PLAN.md §17).

KV: k4q4 (omphalos OMPH_KV_K4=1, llama.cpp q4_0 / q4_0), k8q4 (omphalos'
default mix, llama.cpp q8_0 / q4_0) and k8 (omphalos OMPH_KV_K4_LAYERS=none,
pure K8; llama.cpp has no equivalent mix to compare against).

    uv run python ttft.py --engine omphalos --kv k4q4 --out ../bench/results/ttft-233.jsonl \\
        --csv ../bench/results/ttft-233.csv
    uv run python ttft.py --engine llama --kv k4q4 --llama-server <llama.cpp>/build-hip/bin/llama-server \\
        --out ../bench/results/ttft-233-llama.jsonl --csv ../bench/results/ttft-233.csv

The CSV holds one row per (engine, kv, size): the median TTFT, the median
prefill time and rate, and the median overhead. The JSONL holds every run.
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from niah import count_tokens, free_port, wait_health
from omph_model import omph_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SYSTEM = "Below is a long document. Read it.\n\n<document>\n{}\n</document>"
QUESTION = "Reply with the single word: ready."


def render(tokenize: str, model: str, system: str, question: str) -> str:
    req = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
           "add_generation_prompt": True, "enable_thinking": False}
    return subprocess.run([tokenize, model, "--chat"], input=json.dumps(req), capture_output=True, text=True,
                          check=True).stdout


def fit_prompt(paragraphs: list[str], target: int, tokenize: str, model: str) -> str:
    """A prompt of ~target tokens: grow the document, then scale its text until
    the rendered prompt's count stops changing. The slice is always taken from
    the whole document, so a first guess that is too short can grow."""
    doc = "\n\n".join(paragraphs)
    chars = int(target * 4.0)  # ~4 chars per token in wikitext
    for _ in range(8):
        got = count_tokens(tokenize, model, render(tokenize, model, SYSTEM.format(doc[:chars]), QUESTION))
        if abs(got - target) <= max(2, target // 200):
            break
        chars = max(1, int(chars * target / got))
    return render(tokenize, model, SYSTEM.format(doc[:chars]), QUESTION)


def stream_ttft(url: str, path: str, body: dict, timeout: float) -> tuple[float, dict]:
    """(seconds to the first non-empty delta, the response's timings)."""
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first: float | None = None
    timings: dict = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choice = obj["choices"][0] if obj.get("choices") else {}
                if first is None and (choice.get("text") or (choice.get("delta") or {}).get("content")):
                    first = time.perf_counter() - t0
                if obj.get("timings"):
                    timings = obj["timings"]
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        sys.exit(f"{path} failed: {e}")
    return (first if first is not None else time.perf_counter() - t0), timings


def request(engine: str, url: str, prompt: str, timeout: float) -> tuple[float, dict]:
    if engine == "omphalos":
        return stream_ttft(url, "/v1/completions", {"prompt": prompt, "max_tokens": 1, "temperature": 0,
                                                    "stream": True}, timeout)
    return stream_ttft(url, "/completion", {"prompt": prompt, "n_predict": 1, "temperature": 0, "top_k": 1,
                                            "cache_prompt": False, "stream": True}, timeout)


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["engine", "kv", "size", "prompt_tokens", "runs", "ttft_ms", "prompt_ms",
                                          "overhead_ms", "prefill_tps"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["omphalos", "llama"], required=True)
    ap.add_argument("--kv", default="k8q4", choices=["k4q4", "k8q4", "k8"])
    ap.add_argument("--sizes", default="500,2000,8000,16000,32000,64000,100000")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1, help="unrecorded runs per size")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds per request")
    ap.add_argument("--ctx", type=int, default=106496)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    ap.add_argument("--llama-server", default="")
    ap.add_argument("--text", default=str(ROOT / "models/datasets/wikitext-2-raw/wiki.train.raw"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    tokenize = f"{args.omph}/omph-tokenize"
    paragraphs = [p.strip() for p in Path(args.text).read_text().split("\n") if len(p.strip()) > 40]
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    if args.engine == "omphalos":
        env = {**os.environ, **({"OMPH_KV_K4": "1"} if args.kv == "k4q4" else {})}
        if args.kv == "k8":
            env["OMPH_KV_K4_LAYERS"] = "none"
        cmd = [f"{args.omph}/omph-server", omph_file(args.model), "--port", str(port), "--ctx", str(args.ctx),
               "--cache-ram", "0", "--kv-ram", "0"]
    else:
        if not args.llama_server:
            sys.exit("--llama-server is required for --engine llama")
        env = dict(os.environ)
        cmd = [args.llama_server, "-m", args.model, "--port", str(port), "-c", str(args.ctx), "-ngl", "999",
               "-fa", "on", "-ctk", "q4_0" if args.kv == "k4q4" else "q8_0", "-ctv", "q4_0", "-np", "1",
               "--no-webui"]
    log = open(f"{args.out}.server.log", "w")  # noqa: SIM115 (open while the server runs)
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    rows: list[dict] = []
    summary: list[dict] = []
    try:
        wait_health(url, proc)
        for size in [int(x) for x in args.sizes.split(",")]:
            prompt = fit_prompt(paragraphs, size, tokenize, args.model)
            for run in range(args.warmup + args.runs):
                ttft, tm = request(args.engine, url, prompt, args.timeout)
                if run < args.warmup:
                    continue
                prompt_ms = float(tm.get("prompt_ms", 0.0))
                row = {"engine": args.engine, "kv": args.kv, "size": size, "run": run - args.warmup,
                       "prompt_tokens": int(tm.get("prompt_n", 0)), "ttft_ms": round(ttft * 1000, 1),
                       "prompt_ms": round(prompt_ms, 1), "overhead_ms": round(ttft * 1000 - prompt_ms, 1),
                       "prefill_tps": round(1000.0 * tm["prompt_n"] / prompt_ms, 1) if prompt_ms > 0 else 0.0}
                rows.append(row)
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{args.engine} {args.kv} {size:>7}: ttft {row['ttft_ms']:8.1f} ms "
                      f"(prefill {row['prompt_ms']:8.1f} ms = {row['prefill_tps']:6.0f} t/s, "
                      f"overhead {row['overhead_ms']:6.1f} ms, {row['prompt_tokens']} tok)", flush=True)
            got = [r for r in rows if r["size"] == size]
            summary.append({"engine": args.engine, "kv": args.kv, "size": size,
                            "prompt_tokens": round(statistics.median(r["prompt_tokens"] for r in got)),
                            "runs": len(got), "ttft_ms": statistics.median(r["ttft_ms"] for r in got),
                            "prompt_ms": statistics.median(r["prompt_ms"] for r in got),
                            "overhead_ms": statistics.median(r["overhead_ms"] for r in got),
                            "prefill_tps": statistics.median(r["prefill_tps"] for r in got)})
            if args.csv:
                write_csv(args.csv, summary)
    finally:
        proc.terminate()
        proc.wait(60)
        log.close()


if __name__ == "__main__":
    main()
