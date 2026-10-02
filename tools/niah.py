"""Needle in a haystack: long-context retrieval, omphalos vs llama.cpp (#166).

Each haystack (wikitext-2 train text of ~25k / 50k / 75k / 100k tokens) holds
four needles -- "The current access code for the <north|south|east|west>
vault is <code>." -- at 25 / 50 / 75 / 100 % of the text, and near-miss
distractors: other vaults' codes, the same vault's revoked code, the same
direction's gate code. The haystack is the system message and each question
a user message, so a server prefills the haystack once and answers the four
questions from its cached prefix. Greedy, thinking off; an answer is right
when it contains the needle's code.

--needles N spreads N needles (other vault names) evenly over the text;
--hard adds two more same-vault distractors per needle (a proposed but never
approved code, an "annex" with its own current code); --omph-env passes one
more variable to omph-server (e.g. OMPH_KV_K4_LAYERS=..., #175).

The same prompt text (rendered once with the GGUF's chat template by
omph-tokenize) goes to both engines through their raw completion endpoints.

    uv run python niah.py --engine omphalos --kv k8q4
    uv run python niah.py --engine llama --kv k4q4 \
        --llama-server <llama.cpp>/build-hip/bin/llama-server

Results: one JSON line per question appended to --out, and a table.
"""

import argparse
import json
import os
import random
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from omph_model import omph_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
KEYS = ["north", "south", "east", "west", "amber", "basalt", "cedar", "delta", "ember", "fjord", "granite",
        "hollow", "iris", "juniper", "kestrel", "lagoon"]
OTHER_VAULTS = ["central", "harbor", "mountain", "river", "summit", "valley"]
WORDS = ["HERON", "MAPLE", "COBALT", "FALCON", "JUNIPER", "QUARTZ", "OTTER", "SAFFRON", "TUNDRA", "VIOLET",
         "WALNUT", "ZEPHYR", "BASALT", "CEDAR", "EMBER", "GARNET", "HAZEL", "IVORY", "LUPINE", "MARLIN",
         "NUTMEG", "ORCHID", "PEWTER", "RAVEN", "SPRUCE", "TIGER", "UMBER", "WILLOW", "AMBER", "BRAMBLE"]
SYSTEM = "Below is a long document. Answer the questions about it.\n\n<document>\n{}\n</document>"
QUESTION = ("According to the document, what is the current access code for the {} vault? "
            "Reply with the code only.")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_health(url: str, proc: subprocess.Popen) -> None:
    for _ in range(900):
        if proc.poll() is not None:
            sys.exit(f"the server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                if r.status == 200 and json.loads(r.read() or b"{}").get("status", "ok") == "ok":
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(1)
    sys.exit("the server did not come up")


def code(rng: random.Random, used: set) -> str:
    while True:
        c = f"{rng.randrange(1000, 10000)}-{rng.choice(WORDS)}"
        if c not in used:
            used.add(c)
            return c


def count_tokens(tokenize: str, model: str, text: str) -> int:
    out = subprocess.run([tokenize, model], input=text, capture_output=True, text=True, check=True).stdout
    return len(out.split())


def haystack(paragraphs: list[str], target: int, seed: int, tokenize: str, model: str, n_needles: int = 4,
             hard: bool = False) -> dict:
    """~target tokens of consecutive wikitext paragraphs, with the needles and
    the distractors in it (paragraph boundaries)."""
    rng = random.Random(seed)
    start = rng.randrange(0, len(paragraphs) // 2)
    chars_per_token = 4.0  # first guess, then measured
    for _ in range(3):
        body, n = [], 0
        for p in paragraphs[start:]:
            if n >= target * chars_per_token:
                break
            body.append(p)
            n += len(p) + 1
        tokens = count_tokens(tokenize, model, "\n".join(body))
        if abs(tokens - target) < target * 0.01:
            break
        chars_per_token *= target / tokens
    used: set[str] = set()
    keys = KEYS[:n_needles]
    rng.shuffle(keys)  # which vault sits at which position
    positions = [(i + 1) / n_needles for i in range(n_needles)]
    needles = [{"key": k, "code": code(rng, used), "pos": p} for k, p in zip(keys, positions)]
    distractors = []
    for k in KEYS[:n_needles]:
        distractors.append(f"The access code for the {k} vault used to be {code(rng, used)}, "
                           "but it was revoked last spring.")
        distractors.append(f"The current access code for the {k} gate is {code(rng, used)}.")
        if hard:  # the same vault again, with a code that is not the current one
            distractors.append(f"A new access code for the {k} vault, {code(rng, used)}, was proposed "
                               "but never approved.")
            distractors.append(f"Do not confuse the {k} vault with the {k} annex, whose current access code "
                               f"is {code(rng, used)}.")
    for v in OTHER_VAULTS:
        distractors.append(f"The current access code for the {v} vault is {code(rng, used)}.")
    # insertion points: paragraph indices (the 100 % needle goes after the last one)
    inserts: dict[int, list[str]] = {}
    for nd in needles:
        at = min(len(body), round(nd["pos"] * len(body)))
        inserts.setdefault(at, []).append(f"The current access code for the {nd['key']} vault is {nd['code']}.")
    for d in distractors:
        inserts.setdefault(rng.randrange(1, len(body)), []).insert(0, d)
    text = []
    for i, p in enumerate(body):
        text.extend(inserts.get(i, []))
        text.append(p)
    text.extend(inserts.get(len(body), []))
    return {"text": "\n".join(text), "needles": needles, "codes": sorted(used)}


def render(tokenize: str, model: str, system: str, question: str) -> str:
    req = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
           "add_generation_prompt": True, "enable_thinking": False}
    return subprocess.run([tokenize, model, "--chat"], input=json.dumps(req), capture_output=True, text=True,
                          check=True).stdout


class VramPeak(threading.Thread):
    """The device's VRAM in use (sysfs, every 0.1 s; the desktop's share
    included): the peak since the last reset() and over the whole run."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        cards = sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used"))
        self.path = cards[0] if cards else None
        self.peak = self.total = 0
        self.stop = threading.Event()

    def run(self) -> None:
        while self.path is not None and not self.stop.is_set():
            used = int(self.path.read_text()) >> 20
            self.peak, self.total = max(self.peak, used), max(self.total, used)
            time.sleep(0.1)

    def reset(self) -> int:
        p, self.peak = self.peak, 0
        return p


def ask(engine: str, url: str, prompt: str) -> tuple[str, dict]:
    """The answer and the timings: prompt tokens actually prefilled and their
    ms, generated tokens and their ms (both servers report llama.cpp's
    `timings`), the request's wall time."""
    t0 = time.time()
    if engine == "omphalos":
        r = post(url + "/v1/completions", {"prompt": prompt, "max_tokens": 24, "temperature": 0})
        answer, tm = r["choices"][0]["text"], r["timings"]
    else:
        r = post(url + "/completion", {"prompt": prompt, "n_predict": 24, "temperature": 0, "top_k": 1,
                                       "cache_prompt": True})
        answer, tm = r["content"], r["timings"]
    return answer, {"prompt_n": tm["prompt_n"], "prompt_ms": round(tm["prompt_ms"], 1),
                    "predicted_n": tm["predicted_n"], "predicted_ms": round(tm["predicted_ms"], 1),
                    "wall_s": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["omphalos", "llama"], required=True)
    ap.add_argument("--kv", choices=["k8q4", "k4q4"], required=True)
    ap.add_argument("--needles", type=int, default=4, help=f"needles per haystack (up to {len(KEYS)})")
    ap.add_argument("--hard", action="store_true", help="more same-vault distractors (proposed codes, annexes)")
    ap.add_argument("--omph-env", default="", help="extra environment for omph-server, e.g. "
                    "OMPH_KV_K4_LAYERS=0,3,5 (recorded as the kv label)")
    ap.add_argument("--lengths", default="25000,50000,75000,100000")
    ap.add_argument("--trials", type=int, default=1, help="haystacks per length (different text and codes)")
    ap.add_argument("--seed", type=int, default=166)
    ap.add_argument("--ctx", type=int, default=106496)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    ap.add_argument("--llama-server", default="")
    ap.add_argument("--text", default=str(ROOT / "models/datasets/wikitext-2-raw/wiki.train.raw"))
    ap.add_argument("--out", default="niah.jsonl")
    args = ap.parse_args()

    tokenize = f"{args.omph}/omph-tokenize"
    paragraphs = [p.strip() for p in Path(args.text).read_text().split("\n") if len(p.strip()) > 40]
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    if args.engine == "omphalos":
        env = {**os.environ, **({"OMPH_KV_K4": "1"} if args.kv == "k4q4" else {})}
        if args.omph_env:
            name, _, value = args.omph_env.partition("=")
            env[name] = value
        cmd = [f"{args.omph}/omph-server", omph_file(args.model), "--port", str(port), "--ctx", str(args.ctx)]
    else:
        if not args.llama_server:
            sys.exit("--llama-server is required for --engine llama")
        k = "q8_0" if args.kv == "k8q4" else "q4_0"
        env = dict(os.environ)
        cmd = [args.llama_server, "-m", args.model, "--port", str(port), "-c", str(args.ctx), "-ngl", "999",
               "-fa", "on", "-ctk", k, "-ctv", "q4_0", "-np", "1", "--no-webui"]
    log = open(f"{args.out}.{args.engine}-{args.kv}.server.log", "w")  # noqa: SIM115 (open while the server runs)
    vram = VramPeak()
    vram.start()
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    rows = []
    try:
        wait_health(url, proc)
        vram.reset()
        for trial in range(args.trials):
            for target in [int(x) for x in args.lengths.split(",")]:
                hs = haystack(paragraphs, target, args.seed + 1000 * trial + target, tokenize, args.model,
                              args.needles, args.hard)
                system = SYSTEM.format(hs["text"])
                for nd in hs["needles"]:
                    prompt = render(tokenize, args.model, system, QUESTION.format(nd["key"]))
                    answer, tm = ask(args.engine, url, prompt)
                    found = [c for c in hs["codes"] if c.lower() in answer.lower()]
                    kv = args.kv + (f" {args.omph_env}" if args.omph_env else "")
                    row = {"engine": args.engine, "kv": kv, "trial": trial, "length": target,
                           "position": nd["pos"], "key": nd["key"], "code": nd["code"],
                           "correct": nd["code"].lower() in answer.lower(),
                           "distractor": bool(found) and nd["code"] not in found,
                           "answer": answer.strip(), **tm, "vram_peak_mib": vram.reset()}
                    rows.append(row)
                    with open(args.out, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    pp = tm["prompt_n"] / tm["prompt_ms"] * 1000 if tm["prompt_ms"] > 0 else 0
                    tg = tm["predicted_n"] / tm["predicted_ms"] * 1000 if tm["predicted_ms"] > 0 else 0
                    print(f"{args.engine} {args.kv} {target:>6} @{int(nd['pos'] * 100):>3}%: "
                          f"{'ok  ' if row['correct'] else 'MISS'} {row['answer'][:32]!r:34} "
                          f"prefill {tm['prompt_n']:>6} tok {pp:6.0f} t/s, decode {tg:5.1f} t/s, "
                          f"VRAM peak {row['vram_peak_mib']} MiB", flush=True)
    finally:
        proc.terminate()
        proc.wait(60)
        vram.stop.set()
        log.close()
    ok = sum(r["correct"] for r in rows)
    print(f"{args.engine} {args.kv}: {ok} / {len(rows)} correct, VRAM peak over the run {vram.total} MiB")


if __name__ == "__main__":
    main()
