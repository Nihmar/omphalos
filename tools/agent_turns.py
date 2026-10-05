"""Multi-turn agent prefill on a growing context: omphalos vs llama.cpp (#234).

An agent conversation that grows the way a real one does: a system message with
a document, then turns that each append a tool output (wikitext of ~--fill
tokens) and a short greedy answer, sent as the whole growing `messages` list to
either server's /v1/chat/completions. What matters is what each turn costs:
the new tokens (prefilled) against the cached ones, the time to first token,
the wall time, and the peak VRAM / host RAM.

Two variants interrupt the happy path:
- `--edit-turn K` rewrites the first tool output on turn K (an edited file):
  the cache cannot be reused past the edit;
- `--regen-turn K` resends turn K-1's conversation (a regenerated answer):
  the cache is still valid to the end of the user message.

KV: k4q4 (omphalos OMPH_KV_K4=1, llama.cpp q4_0 / q4_0), k8q4 (omphalos'
default mix, llama.cpp q8_0 / q4_0) and k8 (omphalos OMPH_KV_K4_LAYERS=none;
llama.cpp has no equivalent mix). omphalos runs with its default checkpoints
(`--cache-ram 2048`) and conversation cache (`--kv-ram 8192`), which is what
this measures; llama.cpp's own prompt cache is left on (cache_prompt).

    uv run python agent_turns.py --engine omphalos --kv k4q4 --out ../bench/results/agent-turns-234.jsonl \\
        --csv ../bench/results/agent-turns-234.csv
    uv run python agent_turns.py --engine llama --kv k4q4 --llama-server <llama.cpp>/build-hip/bin/llama-server \\
        --out ../bench/results/agent-turns-234-llama.jsonl --csv ../bench/results/agent-turns-234.csv

The CSV holds one row per turn; the JSONL one per run of every turn.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from niah import VramPeak, free_port, wait_health
from omph_model import omph_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SYSTEM = "You are a coding agent. The project's notes follow.\n\n<document>\n{}\n</document>"
# the tool output of a turn, then the question; the model answers in one sentence
TURN = "Here is the next tool output.\n\n```text\n{}\n```\n\n{}"
QUESTION = "In one short sentence, what file does the last paragraph describe?"
EDIT = "Here is the file again, edited.\n\n```text\n{}\n```\n\n{}"


class RssPeak(threading.Thread):
    """The server process's peak resident memory (VmRSS), every 0.2 s."""

    def __init__(self, pid: int) -> None:
        super().__init__(daemon=True)
        self.pid = pid
        self.peak = 0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        self.peak = max(self.peak, int(line.split()[1]) >> 10)  # MiB
            except OSError:
                return
            time.sleep(0.2)

    def reset(self) -> int:
        p, self.peak = self.peak, 0
        return p

    def stop(self) -> None:
        self._stop.set()


def stream_turn(url: str, body: dict, timeout: float) -> tuple[float, float, str, dict, dict]:
    """(ttft seconds, wall seconds, text, timings, usage) of a streamed chat request."""
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first: float | None = None
    text = ""
    timings: dict = {}
    usage: dict = {}
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
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content"):
                    text += delta["content"]
                    if first is None:
                        first = time.perf_counter() - t0
                if obj.get("timings"):
                    timings = obj["timings"]
                if obj.get("usage"):
                    usage = obj["usage"]
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        sys.exit(f"the chat request failed: {e}")
    wall = time.perf_counter() - t0
    return (first if first is not None else wall), wall, text, timings, usage


def sliced(paragraphs: list[str], chars: int) -> str:
    return "\n\n".join(paragraphs)[:chars]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["omphalos", "llama"], required=True)
    ap.add_argument("--kv", default="k8q4", choices=["k4q4", "k8q4", "k8"])
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--initial", type=int, default=20000, help="tokens of the system document (~4 chars each)")
    ap.add_argument("--fill", type=int, default=2000, help="tokens appended by each turn")
    ap.add_argument("--answer", type=int, default=48, help="max tokens of each turn's answer")
    ap.add_argument("--edit-turn", type=int, default=0, help="rewrite the first tool output on this turn (=0: never)")
    ap.add_argument("--regen-turn", type=int, default=0, help="resend the previous conversation on this turn")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--ctx", type=int, default=106496)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    ap.add_argument("--llama-server", default="")
    ap.add_argument("--text", default=str(ROOT / "models/datasets/wikitext-2-raw/wiki.train.raw"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    paragraphs = [p.strip() for p in Path(args.text).read_text().split("\n") if len(p.strip()) > 40]
    doc = sliced(paragraphs, args.initial * 4)
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    if args.engine == "omphalos":
        env = {**os.environ, **({"OMPH_KV_K4": "1"} if args.kv == "k4q4" else {})}
        if args.kv == "k8":
            env["OMPH_KV_K4_LAYERS"] = "none"
        cmd = [f"{args.omph}/omph-server", omph_file(args.model), "--port", str(port), "--ctx", str(args.ctx)]
    else:
        if not args.llama_server:
            sys.exit("--llama-server is required for --engine llama")
        env = dict(os.environ)
        cmd = [args.llama_server, "-m", args.model, "--port", str(port), "-c", str(args.ctx), "-ngl", "999",
               "-fa", "on", "-ctk", "q4_0" if args.kv == "k4q4" else "q8_0", "-ctv", "q4_0", "-np", "1",
               "--no-webui"]
    log = open(f"{args.out}.server.log", "w")  # noqa: SIM115 (open while the server runs)
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    vram = VramPeak()
    vram.start()
    rss = RssPeak(proc.pid)
    rss.start()
    rows: list[dict] = []
    messages = [{"role": "system", "content": SYSTEM.format(doc)}]
    previous: list[dict] | None = None
    try:
        wait_health(url, proc)
        vram.reset()
        rss.reset()
        for turn in range(1, args.turns + 1):
            tool = sliced(paragraphs, args.fill * 4)
            if turn == args.regen_turn and previous is not None:
                kind, send = "regen", previous
            elif turn == args.edit_turn and len(messages) > 1:
                kind = "edit"
                messages[1]["content"] = EDIT.format(tool, QUESTION)
                send = messages
            else:
                kind = "append"
                messages.append({"role": "user", "content": TURN.format(tool, QUESTION)})
                send = messages
            previous = list(send)
            body = {"model": "x", "messages": send, "max_tokens": args.answer, "temperature": 0,
                    "stream": True, "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False}}
            ttft, wall, answer, tm, usage = stream_turn(url, body, args.timeout)
            details = usage.get("prompt_tokens_details") or {}
            prompt_ms = float(tm.get("prompt_ms", 0.0))
            predicted = int(tm.get("predicted_n", 0))
            decode_ms = float(tm.get("predicted_ms", 0.0))
            row = {"engine": args.engine, "kv": args.kv, "turn": turn, "kind": kind,
                   "prompt_tokens": int(usage.get("prompt_tokens") or tm.get("prompt_n", 0)),
                   "cached_tokens": int(details.get("cached_tokens", 0)),
                   "prompt_n": int(tm.get("prompt_n", 0)),
                   "prompt_ms": round(prompt_ms, 1), "ttft_ms": round(ttft * 1000, 1),
                   "wall_s": round(wall, 2), "predicted_n": predicted,
                   "decode_tps": round(1000.0 * predicted / decode_ms, 1) if decode_ms > 0 else 0.0,
                   "vram_peak_mib": vram.reset(), "rss_peak_mib": rss.reset()}
            rows.append(row)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            # the answer continues the history on an append (an answer to nothing is useless for the
            # next turn); the regen/edit turns keep the history they were sent
            if kind == "append":
                messages.append({"role": "assistant", "content": answer})
            print(f"{args.engine} {args.kv} turn {turn} ({kind}): {row['prompt_tokens']} prompt "
                  f"({row['cached_tokens']} cached, {row['prompt_n']} prefilled) ttft {row['ttft_ms']} ms, "
                  f"prefill {row['prompt_ms']} ms, wall {row['wall_s']} s, VRAM {row['vram_peak_mib']} MiB, "
                  f"RSS {row['rss_peak_mib']} MiB", flush=True)
    finally:
        rss.stop()
        vram.stop.set()
        proc.terminate()
        proc.wait(60)
        log.close()
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
