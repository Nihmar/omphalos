"""Cross-request state leakage on a warm omph-server (#200).

Every other check exercises requests in isolation, but a serving engine carries
state between them: the delta-net A/B states and the conv tails, the replay
records of speculation, the FP16 KV ring, sequence checkpoints in host RAM and
their restore, the prefix reuse when a prompt extends the cached sequence, the
MTP block's KV and h. A llama.cpp HIP bug once carried that state across
requests -- later answers quoted earlier prompts, deterministically at
temperature 0.

This one starts a **fresh** server per prompt (that prompt alone, nothing
before it) and a **warm** server per configuration, runs the whole set back to
back on it twice in shuffled orders, and requires:

  * the greedy answers identical token for token (text and completion_tokens)
    to the fresh ones, and to each other across orders and configurations that
    must not change the numbers (the mixed K4/K8 default is compared within
    itself: it does change them, by design); the text compared is
    reasoning_content + content (thinking on spends the whole budget inside
    <think>), and the 2k family runs with enable_thinking false so the answer
    itself is compared too (#348);
  * the seeded-sampling requests identical too (their random stream restarts
    from the request's seed, and a sampled request follows its configuration's
    speculative stream: the no-mtp variant gets its own fresh reference);
  * a canary sentence planted in one prompt that never appears in a later
    answer.

    uv run python check_warm_server.py --omph ../engine/build
    uv run python check_warm_server.py --omph ../engine/build --long-tokens 0
    uv run python check_warm_server.py --omph ../engine/build --mmproj ../models/mmproj-Qwen3.8-27B-BF16.gguf

The servers' logs stay in /tmp/check_warm_server-<tag>.log.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from niah import count_tokens, free_port, post, wait_health
from omph_model import omph_file

ROOT = Path(__file__).resolve().parent.parent
CANARY = "ZEBRA-7731"  # the canary prompt asks for it; nothing later may show it
FAILURES: list[str] = []
# (name, environment, extra server arguments): the configurations to run warm.
# The first three must give the fresh server's answers token for token; the K4/K8
# one quantizes the cache differently and is compared within itself (and its
# divergence from the others is reported, not judged: that is what #297 asks).
VARIANTS = (
    ("default", {}, ()),
    ("no-mtp", {}, ("--no-mtp",)),
    ("no-checkpoints", {}, ("--cache-ram", "0")),
    ("k8", {"OMPH_KV_K4_LAYERS": "none"}, ()),
)
BIT_IDENTICAL_VARIANTS = 3  # how many of VARIANTS must match the fresh answers


def check(cond: bool, what: str) -> None:
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILURES.append(what)


class Server:
    """One omph-server in a subprocess, with its log, until the context exits."""

    def __init__(self, omph: str, model: str, tag: str, ctx: int, env: dict | None = None,
                 extra: tuple = (), mmproj: str | None = None) -> None:
        self.port = free_port()
        self.tag = tag
        self.log = Path(f"/tmp/check_warm_server-{tag}.log")
        self.cmd = [f"{omph}/omph-server", omph_file(model), "--port", str(self.port), "--ctx", str(ctx)]
        if mmproj:
            self.cmd += ["--mmproj", mmproj]
        self.cmd += list(extra)
        self.env = {**os.environ, **(env or {})}
        self.proc: subprocess.Popen | None = None
        self.url = ""

    def __enter__(self) -> str:
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL,
                                     stderr=open(self.log, "w"), env=self.env)
        self.url = f"http://127.0.0.1:{self.port}"
        wait_health(self.url, self.proc)
        return self.url

    def __exit__(self, *exc) -> None:
        assert self.proc is not None
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def ask(url: str, item: dict, timeout: float = 1800) -> tuple[str, int]:
    """The answer's text and its completion_tokens (-1 when unreported).

    The text is reasoning_content followed by content: with thinking on and a
    small max_tokens the whole budget goes into <think>, so comparing content
    alone compared two empty strings on every chat prompt (#348).
    """
    body = {"max_tokens": item["max_tokens"], "temperature": item.get("temperature", 0)}
    if "enable_thinking" in item:
        body["chat_template_kwargs"] = {"enable_thinking": item["enable_thinking"]}
    for k in ("repeat_penalty", "repeat_last_n"):
        if k in item:
            body[k] = item[k]
    if not item.get("temperature"):
        body["seed"] = 0
    elif "seed" in item:
        body["seed"] = item["seed"]
    if item["kind"] == "chat":
        r = post(url + "/v1/chat/completions", {**body, "messages": item["messages"]}, timeout)
        m = r["choices"][0]["message"]
        text = (m.get("reasoning_content") or "") + (m.get("content") or "")
    else:
        r = post(url + "/v1/completions", {**body, "prompt": item["prompt"]}, timeout)
        text = r["choices"][0]["text"]
    tokens = r.get("usage", {}).get("completion_tokens")
    return text, -1 if tokens is None else int(tokens)


def data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def chars_for(tokens: int, text: str, tokenize: str, model: str) -> tuple[int, int]:
    """A slice of `text` of about `tokens` tokens, and its measured length."""
    probe = 40_000
    per_char = count_tokens(tokenize, model, text[:probe]) / probe
    chars = min(len(text), int(tokens / max(per_char, 1e-6)))
    return chars, count_tokens(tokenize, model, text[:chars])


def make_prompts(args, text: str, tokenize: str, model: str, fresh_2k: str | None) -> list[dict]:
    items: list[dict] = []
    short_q = "In one sentence: why is the sky blue?"
    items.append({"tag": "short", "kind": "chat", "max_tokens": 32,
                  "messages": [{"role": "user", "content": short_q}]})
    chars2k, _tok2k = chars_for(2048, text, tokenize, model)
    ctx2k = "Answer questions about this text.\n\n" + text[100_000:100_000 + chars2k]
    q1 = "In one sentence: what is this text about?"
    q2 = "Name one thing it mentions."
    items.append({"tag": "2k", "kind": "chat", "max_tokens": 32, "enable_thinking": False,
                  "messages": [{"role": "system", "content": ctx2k}, {"role": "user", "content": q1}]})
    # shares the whole 2k prefix with the request above: the cache has to be reused
    items.append({"tag": "2k-other", "kind": "chat", "max_tokens": 32, "enable_thinking": False,
                  "messages": [{"role": "system", "content": ctx2k}, {"role": "user", "content": q2}]})
    # the same request again: the second one resumes from a saved conversation
    items.append({"tag": "2k-retry", "kind": "chat", "max_tokens": 32, "enable_thinking": False,
                  "messages": [{"role": "system", "content": ctx2k}, {"role": "user", "content": q2}]})
    # the next turn of that conversation, with the fresh answer in it
    if fresh_2k is not None:
        items.append({"tag": "2k-turn2", "kind": "chat", "max_tokens": 32, "enable_thinking": False,
                      "messages": [{"role": "system", "content": ctx2k}, {"role": "user", "content": q1},
                                   {"role": "assistant", "content": fresh_2k},
                                   {"role": "user", "content": "In one sentence: and what is the last paragraph about?"}]})
    items.append({"tag": "raw", "kind": "completion", "max_tokens": 32,
                  "prompt": text[300_000:300_000 + chars2k]})
    items.append({"tag": "sampled", "kind": "chat", "max_tokens": 32, "temperature": 0.8, "seed": 12345,
                  "messages": [{"role": "system", "content": ctx2k},
                               {"role": "user", "content": "Invent a name for a small town."}]})
    # The penalties' window is the last penalty_last_n tokens of the *sequence*,
    # which on a warm server includes what a previous request left cached: this
    # must still be the same window a fresh server sees.
    items.append({"tag": "penalties", "kind": "chat", "max_tokens": 32, "temperature": 0.6, "seed": 99,
                  "repeat_penalty": 1.1, "repeat_last_n": 1024,
                  "messages": [{"role": "system", "content": ctx2k},
                               {"role": "user", "content": "Repeat the words of the text's first "
                                                           "sentence, one after another."}]})
    items.append({"tag": "canary", "kind": "chat", "max_tokens": 48, "enable_thinking": False,
                  "messages": [{"role": "user",
                                "content": f"Repeat this sentence exactly, then say done: the password is {CANARY}."}]})
    if args.long_tokens:
        charsl, tokl = chars_for(args.long_tokens, text, tokenize, model)
        items.append({"tag": f"long{tokl // 1024}k", "kind": "chat", "max_tokens": 32,
                      "messages": [{"role": "system", "content": "Answer questions about this text.\n\n" +
                                                                 text[500_000:500_000 + charsl]},
                                   {"role": "user", "content": q1}]})
    if args.mmproj:
        items.append({"tag": "image", "kind": "chat", "max_tokens": 32,
                      "messages": [{"role": "user", "content": [
                          {"type": "image_url", "image_url": {"url": data_url(Path(args.image))}},
                          {"type": "text", "text": "In one sentence: what is in this image?"}]}]})
    print(f"     prompts: {', '.join(i['tag'] for i in items)}", flush=True)
    return items


def canary_hits(answers: dict[str, str]) -> list[str]:
    """The prompts whose answer shows part of the canary (its own prompt does,
    and only its own answer may)."""
    hits = []
    for tag, text in answers.items():
        if tag == "canary":
            continue
        low = text.lower()
        if any(w in low for w in ("zebra", "7731")):
            hits.append(tag)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    ap.add_argument("--text", default=str(ROOT / "models/datasets/wikitext-2-raw/wiki.train.raw"))
    ap.add_argument("--long-tokens", type=int, default=16384,
                    help="the long prompt's size (0: skip it; the most expensive part of the test)")
    ap.add_argument("--ctx", type=int, default=0, help="the servers' context (0: long-tokens + 2048)")
    ap.add_argument("--orders", type=int, default=2, help="shuffled orders run back to back on one server")
    ap.add_argument("--fresh-limit", type=int, default=0,
                    help="fresh servers to start (0: one per prompt; a smaller number skips the rest, "
                         "which weakens the test)")
    ap.add_argument("--mmproj", default="", help="also check one image prompt (needs the vision build)")
    ap.add_argument("--image", default=str(Path("/home/alessandro/Projects/llama.cpp-RX9060XT-16GB/"
                                                "tools/mtmd/test-1.jpeg")))
    args = ap.parse_args()
    ctx = args.ctx or max(8192, args.long_tokens + 2048)
    text = Path(args.text).read_text()
    tokenize = f"{args.omph}/omph-tokenize"

    # The fresh references: one server per prompt, that prompt alone.
    fresh: dict[str, tuple[str, int]] = {}
    started = 0
    items = make_prompts(args, text, tokenize, args.model, None)
    pending = []
    for item in items:
        if item["tag"] == "2k-retry":  # the same request as 2k-other: one reference covers both
            continue
        if args.fresh_limit and started >= args.fresh_limit:
            pending.append(item["tag"])
            continue
        with Server(args.omph, args.model, f"fresh-{item['tag']}", ctx, mmproj=args.mmproj or None) as url:
            fresh[item["tag"]] = ask(url, item)
            started += 1
        t, n = fresh[item["tag"]]
        print(f"ok   fresh {item['tag']}: {n} tokens: {json.dumps(t[:70])}", flush=True)
    if pending:
        print(f"     (fresh skipped for: {', '.join(pending)} -- the warm runs then only "
              f"cross-check each other)", flush=True)
    # The conversation's second turn needs the fresh answer of its first one; the
    # retry is the same request as 2k-other, so it shares that reference.
    items = make_prompts(args, text, tokenize, args.model, fresh.get("2k", ("", 0))[0])
    for extra in ("2k-turn2",):
        if extra in fresh or args.fresh_limit and started >= args.fresh_limit:
            continue
        item = next(i for i in items if i["tag"] == extra)
        with Server(args.omph, args.model, f"fresh-{extra}", ctx, mmproj=args.mmproj or None) as url:
            fresh[extra] = ask(url, item)
            started += 1
        t, n = fresh[extra]
        print(f"ok   fresh {extra}: {n} tokens: {json.dumps(t[:70])}", flush=True)
    if "2k-other" in fresh:
        fresh["2k-retry"] = fresh["2k-other"]
    # A sampled request follows its configuration's speculative RNG stream (#197:
    # the distribution is the same, the stream is not), so the no-mtp variant
    # cannot be compared against the default fresh reference. One more fresh
    # server for each sampled prompt (#348's rerun found it: the text comparison
    # used to compare two empty contents and never saw it).
    fresh_nomtp: dict[str, tuple[str, int]] = {}
    if not args.fresh_limit:
        for tag in (i["tag"] for i in items if i.get("temperature")):
            item = next(i for i in items if i["tag"] == tag)
            with Server(args.omph, args.model, f"fresh-{tag}-nomtp", ctx, extra=("--no-mtp",),
                        mmproj=args.mmproj or None) as url:
                fresh_nomtp[tag] = ask(url, item)
            t, n = fresh_nomtp[tag]
            print(f"ok   fresh {tag} (no-mtp): {n} tokens: {json.dumps(t[:70])}", flush=True)

    for vi, (name, env, extra) in enumerate(VARIANTS):
        rng = random.Random(20261005)
        orders = []
        for k in range(args.orders):
            order = list(items)
            rng.shuffle(order)  # a different order each pass, back to back
            orders.append(order)
        warm: list[dict[str, str]] = []
        with Server(args.omph, args.model, f"warm-{name}", ctx, env=env, extra=extra,
                    mmproj=args.mmproj or None) as url:
            for k, order in enumerate(orders):
                answers: dict[str, str] = {}
                for item in order:
                    t, n = ask(url, item)
                    answers[item["tag"]] = t
                    ref = fresh.get(item["tag"])
                    if "--no-mtp" in extra:
                        ref = fresh_nomtp.get(item["tag"], ref)
                    if ref is None:
                        continue
                    same = t == ref[0] and (n < 0 or ref[1] < 0 or n == ref[1])
                    if vi < BIT_IDENTICAL_VARIANTS:
                        check(same, f"{name}, order {k + 1}, {item['tag']}: identical to the fresh run")
                        if not same:
                            print(f"     got {json.dumps(t[:80])}\n     ref {json.dumps(ref[0][:80])}",
                                  flush=True)
                    elif k == 0 and not same:
                        print(f"     {name}, {item['tag']}: differs from the default fresh run "
                              f"(expected for another K precision, #297)", flush=True)
                warm.append(answers)
        # The two orders against each other: any difference is state leaking
        # between requests, whatever the configuration.
        for tag in (i["tag"] for i in items):
            if tag in warm[0] and tag in warm[-1]:
                check(warm[0][tag] == warm[-1][tag], f"{name}: {tag} identical across the orders")
        hits = canary_hits({**warm[0], **warm[-1]})
        check(not hits, f"{name}: the canary did not leak (in {', '.join(hits) or 'nothing'})")

    print("     logs: /tmp/check_warm_server-*.log", flush=True)
    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) failed", flush=True)
        sys.exit(1)
    print("\nwarm-server checks passed", flush=True)


if __name__ == "__main__":
    main()
