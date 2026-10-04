"""Switching conversations on omph-server (#179): saved to host RAM, restored.

Two chats with long, different system prompts (wikitext-2 text, ~3000 tokens
each), greedy. The reference server runs chat A's two turns back to back; the
tested one runs A's first turn, then chat B (which leaves A: the server saves
it to host RAM), then A's second turn, which must restore A instead of
prefilling it -- most of its prompt cached -- and answer exactly as the
reference. Both with thinking off (the next turn extends the cached sequence)
and on (the template drops the reasoning, so the next turn continues only up
to the last <|im_start|>: a checkpoint point of the saved conversation), and
with each drafter given by --spec (MTP, DFlash2: their KV and ring are saved
too).

    uv run python check_conversations.py [--spec mtp,dflash]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from niah import free_port, post, wait_health
from omph_model import omph_file

ROOT = Path(__file__).resolve().parent.parent


def chat(url: str, msgs: list, thinking: bool) -> tuple[str, int]:
    r = post(url + "/v1/chat/completions",
             {"messages": msgs, "max_tokens": 160, "temperature": 0,
              "chat_template_kwargs": {"enable_thinking": thinking}})
    m = r["choices"][0]["message"]
    return m.get("content") or "", r["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="mtp,dflash")
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--drafter", default=str(ROOT / "models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    ap.add_argument("--text", default=str(ROOT / "models/datasets/wikitext-2-raw/wiki.train.raw"))
    args = ap.parse_args()

    text = Path(args.text).read_text()
    sys_a = "Answer questions about this text.\n\n" + text[100_000:113_000]
    sys_b = "Answer questions about this text.\n\n" + text[300_000:313_000]
    q1 = "In one sentence: what is the first paragraph about?"
    q2 = "Now name one person or place the text mentions."

    def server(spec: str, log: Path) -> tuple[subprocess.Popen, str]:
        port = free_port()
        cmd = [f"{args.omph}/omph-server", omph_file(args.model), "--port", str(port), "--ctx", "16384"]
        if spec == "dflash":
            cmd += ["--dflash", omph_file(args.drafter)]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(log, "w"))  # noqa: SIM115
        url = f"http://127.0.0.1:{port}"
        wait_health(url, proc)
        return proc, url

    failures = []
    for spec in args.spec.split(","):
        for thinking in (False, True):
            tag = f"{spec}, thinking {'on' if thinking else 'off'}"
            log = Path(f"/tmp/check_conversations-{spec}.log")
            proc, url = server(spec, log)
            try:
                a1, _ = chat(url, [{"role": "system", "content": sys_a}, {"role": "user", "content": q1}], thinking)
                turn2 = [{"role": "system", "content": sys_a}, {"role": "user", "content": q1},
                         {"role": "assistant", "content": a1}, {"role": "user", "content": q2}]
                ref, _ = chat(url, turn2, thinking)
            finally:
                proc.terminate()
                proc.wait()
            proc, url = server(spec, log)
            try:
                a1b, _ = chat(url, [{"role": "system", "content": sys_a}, {"role": "user", "content": q1}], thinking)
                chat(url, [{"role": "system", "content": sys_b}, {"role": "user", "content": q1}], thinking)
                got, cached = chat(url, turn2, thinking)
            finally:
                proc.terminate()
                proc.wait()
            lines = log.read_text()
            ok = a1b == a1 and got == ref and cached >= 2048 and "previous conversation saved" in lines \
                and "restored" in lines
            print(f"{'ok  ' if ok else 'FAIL'} {tag}: turn 2 {'identical' if got == ref else 'DIFFERS'}, "
                  f"{cached} prompt tokens restored", flush=True)
            if not ok:
                failures.append(tag)
                print("   ref:", json.dumps(ref[:200]), "\n   got:", json.dumps(got[:200]))
    if failures:
        sys.exit(f"{len(failures)} failure(s)")
    print("all conversation checks passed")


if __name__ == "__main__":
    main()
