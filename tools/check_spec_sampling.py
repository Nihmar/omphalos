"""Speculative sampling vs plain sampling: same distribution? (#197)

Speculative sampling is exact in distribution but does not reproduce plain
sampling's random stream, so the check is statistical. The same prompt is
sampled N times (seeds 0 .. N-1, temperature T, a few tokens each) by an
omph-server without the MTP block (plain sampling) and by one with it
(speculative sampling). For each generated position the two empirical token
distributions are compared by total variation distance, and a permutation
test (labels shuffled) gives the p-value of that distance: a correct
implementation gives p-values spread over (0, 1), a biased one p ~ 0.

    uv run python check_spec_sampling.py [--n 400] [--temp 1.0] [--omph ../engine/build]
"""

import argparse
import random
import subprocess
from collections import Counter
from pathlib import Path

from niah import free_port, post, wait_health
from omph_model import omph_file

ROOT = Path(__file__).resolve().parent.parent
PROMPT = ("Here is a long list of random everyday words, one per line, no repeats:\n"
          "apple\nwindow\nriver\n")


def tvd(a: list[int], b: list[int]) -> float:
    ca, cb = Counter(a), Counter(b)
    return 0.5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in set(ca) | set(cb))


def permutation_p(a: list[int], b: list[int], rounds: int, rng: random.Random) -> float:
    obs = tvd(a, b)
    pool = a + b
    hits = 0
    for _ in range(rounds):
        rng.shuffle(pool)
        if tvd(pool[:len(a)], pool[len(a):]) >= obs:
            hits += 1
    return (hits + 1) / (rounds + 1)


def collect(omph: str, model: str, mtp: bool, n: int, temp: float, max_tokens: int, top_p: float) -> list:
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    cmd = [f"{omph}/omph-server", model, "--port", str(port), "--ctx", "4096"] + ([] if mtp else ["--no-mtp"])
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    seqs = []
    try:
        wait_health(url, proc)
        for seed in range(n):
            r = post(url + "/v1/completions", {"prompt": PROMPT, "max_tokens": max_tokens, "temperature": temp,
                                               "top_p": top_p, "seed": seed})
            seqs.append(r["choices"][0]["text"])
    finally:
        proc.terminate()
        proc.wait(60)
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    args = ap.parse_args()

    tokenize = f"{args.omph}/omph-tokenize"
    plain = collect(args.omph, omph_file(args.model), False, args.n, args.temp, args.max_tokens, args.top_p)
    spec = collect(args.omph, omph_file(args.model), True, args.n, args.temp, args.max_tokens, args.top_p)

    def ids(text: str) -> list[int]:
        out = subprocess.run([tokenize, args.model, "--no-parse-special"], input=text, capture_output=True,
                             text=True, check=True).stdout
        return [int(x) for x in out.split()]

    # the texts re-tokenized: positions are token positions of the continuation
    pt = [ids(s) for s in plain]
    st = [ids(s) for s in spec]
    rng = random.Random(197)
    print(f"{args.n} samples per mode, temperature {args.temp}, top_p {args.top_p}")
    worst = 1.0
    for pos in range(args.max_tokens):
        a = [s[pos] for s in pt if len(s) > pos]
        b = [s[pos] for s in st if len(s) > pos]
        if len(a) < args.n // 2 or len(b) < args.n // 2:
            break
        half = len(a) // 2
        base = tvd(a[:half], a[half:])
        p = permutation_p(a, b, args.rounds, rng)
        worst = min(worst, p)
        print(f"position {pos}: {len(set(a) | set(b))} distinct tokens, TVD plain vs spec {tvd(a, b):.3f} "
              f"(plain half vs half {base:.3f}), permutation p = {p:.3f}")
    joint = permutation_p([hash(s) for s in plain], [hash(s) for s in spec], args.rounds, rng)
    print(f"whole continuations: {len(set(plain) | set(spec))} distinct, permutation p = {joint:.3f}")
    print("PASS" if min(worst, joint) > 0.01 else "FAIL: the distributions differ")


if __name__ == "__main__":
    main()
