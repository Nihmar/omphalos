#!/usr/bin/env python3
"""A long conversation continued across requests, with --kv-ram on (#375).

The 2026-10-04 replay sent omph-server a 88,760-token conversation and then a
100,396-token continuation of it; the second request never answered (host spin,
GPU idle, nothing in the log) and the run was killed after 35 minutes. This is
that shape, at a smaller size and without the pi session: one long system prompt
(~36k tokens of the local text), a first turn, then a continuation that appends
~10k more tokens and asks the next question. The resumed answer must equal the
one a cold server gives for the same prompt, and the log must show the save and
the restore; a request that does not answer within --watchdog seconds fails the
check instead of hanging the run.

    uv run python check_long_restore.py [--text FILE] [--kb 150] [--watchdog 600]

The default text is the wikitext-2 set when it is present (#375's own runs used
it); any long text file works (pass e.g. the repository's PLAN.md + docs).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from niah import free_port, wait_health
from omph_model import omph_file

ROOT = Path(__file__).resolve().parent.parent


def chat_full(url: str, msgs: list, thinking: bool, max_tokens: int, timeout: float) -> tuple[str, int, float]:
    import urllib.request

    body = {"messages": msgs, "max_tokens": max_tokens, "stream": False, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    m = out["choices"][0]["message"]
    return (m.get("content") or ""), (out.get("usage") or {}).get("prompt_tokens", 0), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", default=str(ROOT / "models/datasets/wikitext-2-raw/wiki.train.raw"))
    ap.add_argument("--kb", type=int, default=150, help="KiB of text in the conversation")
    ap.add_argument("--more-kb", type=int, default=40, help="KiB appended by the continuation")
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--rounds", type=int, default=1,
                    help="distinct conversations to cycle before returning to the first "
                         "(>1 exercises the snapshot store's eviction: #375's report had six)")
    ap.add_argument("--kv-ram", type=int, default=8192, help="the server's --kv-ram budget (MiB)")
    ap.add_argument("--watchdog", type=float, default=600.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    args = ap.parse_args()

    text = Path(args.text).read_text(errors="replace")
    head = "Answer questions about this text.\n\n"
    sys_a = head + text[100_000: 100_000 + args.kb * 1024]
    more = text[400_000: 400_000 + args.more_kb * 1024]
    q1 = "In one sentence: what is the first paragraph about?"
    q2 = "In one sentence: what does the passage just added describe?"

    def server(tag: str) -> tuple[subprocess.Popen, str]:
        port = free_port()
        log = Path(f"/tmp/check_long_restore-{tag}.log")
        cmd = [f"{args.omph}/omph-server", omph_file(args.model), "--port", str(port), "--ctx", str(args.ctx),
               "--kv-ram", str(args.kv_ram)]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(log, "w"))  # noqa: SIM115
        url = f"http://127.0.0.1:{port}"
        wait_health(url, proc)
        return proc, url

    turn1 = [{"role": "system", "content": sys_a}, {"role": "user", "content": q1}]
    # the continuation: the same conversation, ~10k more tokens, then the next
    # question (thinking off, so the prompt extends the cached sequence)
    turn2 = turn1 + [{"role": "assistant", "content": "It introduces the subject of the passage."},
                     {"role": "user", "content": more + "\n\n" + q2}]

    # reference: a server that never sees the first turn
    proc, url = server("ref")
    try:
        ref_a, n_ref, _ = chat_full(url, turn1, False, args.max_tokens, args.watchdog)
        ref_b, n2_ref, _ = chat_full(url, turn2, False, args.max_tokens, args.watchdog)
    finally:
        proc.terminate()
        proc.wait()

    proc, url = server("test")
    try:
        # the other conversations first: each one makes the server save the one
        # before it, so the store fills and starts evicting (the shape of #375's
        # six-cut replay)
        span = args.kb * 1024
        for k in range(1, args.rounds):
            # a different long slice per round (wrapping: the local texts differ in
            # length, and a short prompt would not be saved at all - kSnapshotMin)
            off = (100_000 + k * 400_000) % max(1, len(text) - span - 1)
            other = head + text[off: off + span]
            chat_full(url, [{"role": "system", "content": other}, {"role": "user", "content": q1}], False,
                      args.max_tokens, args.watchdog)
        # now the first conversation's continuation: the server must restore it
        # from the store (evicting whatever it needs to reach it)
        a, n1, t1 = chat_full(url, turn1, False, args.max_tokens, args.watchdog)
        b, n2, t2 = chat_full(url, turn2, False, args.max_tokens, args.watchdog)
    finally:
        proc.terminate()
        proc.wait()
    lines = Path("/tmp/check_long_restore-test.log").read_text()
    print(f"reference: turn 1 {n_ref} prompt tokens, continuation {n2_ref} ({ref_b[:60]!r})")
    print(f"tested:    turn 1 {n1} prompt tokens in {t1:.0f} s, continuation {n2} "
          f"({n2 - n1} new) in {t2:.0f} s ({b[:60]!r})")
    same = a == ref_a and b == ref_b
    saved = "previous conversation saved" in lines
    restored = "restored" in lines
    ok = same and saved and restored
    print(f"{'ok  ' if ok else 'FAIL'}: answers {'identical' if same else 'DIFFER'}, "
          f"saved={saved}, restored={restored}, continuation {n2 - n1} new tokens")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
