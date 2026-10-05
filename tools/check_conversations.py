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

Also the rewritten history a coding agent produces when it compacts (#286):
a prompt that shares only the system prompt with the cached conversation,
diverging in its middle. The tested server restores the checkpoint at the
system/user boundary and prefills the rewritten rest over the old KV; a
reference server sees the same prompt cold. Same answer, and the tested one
really restored (the checkpoint is the first cut, at least 512 tokens in).
The negative control diverges inside the first message, where no checkpoint
is usable: the caches must restart and still give the reference answer.

With --image, a second conversation with an image (#339): the server prefills
the text before the image while the CPU encodes it (#180), and the saved
conversation must still be restored on the next turn instead of prefilled
again from that early point.

    uv run python check_conversations.py [--spec mtp,dflash] [--image FILE]
"""

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

from niah import free_port, post, wait_health
from omph_model import omph_file

ROOT = Path(__file__).resolve().parent.parent


def chat_full(url: str, msgs: list, thinking: bool, max_tokens: int = 160) -> tuple[dict, int]:
    r = post(url + "/v1/chat/completions",
             {"messages": msgs, "max_tokens": max_tokens, "temperature": 0,
              "chat_template_kwargs": {"enable_thinking": thinking}})
    m = r["choices"][0]["message"]
    return m, r["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)


def chat(url: str, msgs: list, thinking: bool) -> tuple[str, int]:
    m, cached = chat_full(url, msgs, thinking)
    return m.get("content") or "", cached


def answer(m: dict) -> tuple[str, str]:
    """What to compare between two runs: the answer, or the reasoning when the
    160-token cap is spent inside <think> (thinking on)."""
    return m.get("content") or "", m.get("reasoning_content") or ""


def answer_text(m: dict) -> str:
    """The answer for a one-line report: content, or the reasoning when the
    content is empty (thinking on, the cap inside <think>)."""
    content, reasoning = answer(m)
    return content or reasoning


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="mtp,dflash")
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--drafter", default=str(ROOT / "models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    ap.add_argument("--text", default=str(ROOT / "models/datasets/wikitext-2-raw/wiki.train.raw"))
    ap.add_argument("--mmproj", default=str(ROOT / "models/mmproj-Qwen3.8-27B-BF16.gguf"))
    ap.add_argument("--image", default=None,
                    help="also check an image conversation (#339), e.g. llama.cpp's tools/mtmd/test-1.jpeg")
    args = ap.parse_args()

    text = Path(args.text).read_text()
    sys_a = "Answer questions about this text.\n\n" + text[100_000:113_000]
    sys_b = "Answer questions about this text.\n\n" + text[300_000:313_000]
    q1 = "In one sentence: what is the first paragraph about?"
    q2 = "Now name one person or place the text mentions."
    # The rewritten history of #286: the same system prompt, the middle turns
    # replaced by a summary, then the last question. Its common prefix with the
    # conversation above is the system prompt, so the tested server must resume
    # from the first checkpoint (the system/user boundary, >= 512 tokens in) and
    # prefill the rest; the reference server only ever sees this prompt.
    sys_c = "Answer questions about this text.\n\n" + text[500_000:513_000]
    summary = ("Summary of the earlier turns: the first paragraph of the text and a person or "
               "place it mentions were identified. The reader now asks:")
    q3 = "In one sentence: what does the last paragraph describe?"
    hist = [{"role": "system", "content": sys_c}, {"role": "user", "content": q1},
            {"role": "assistant", "content": "It introduces the subject of the passage."},
            {"role": "user", "content": q2}]
    rewritten = [{"role": "system", "content": sys_c}, {"role": "user", "content": summary + " " + q3}]
    # the negative control: the divergence is inside the first message, no
    # checkpoint is usable, so the caches have to restart
    rewritten_neg = [{"role": "system", "content": sys_c.replace("Answer questions about",
                                                                  "Explain the passage about", 1)}] \
        + rewritten[1:]

    def server(spec: str, log: Path, mmproj: str | None = None) -> tuple[subprocess.Popen, str]:
        port = free_port()
        cmd = [f"{args.omph}/omph-server", omph_file(args.model), "--port", str(port), "--ctx", "16384"]
        if spec == "dflash":
            cmd += ["--dflash", omph_file(args.drafter)]
        if mmproj:
            cmd += ["--mmproj", mmproj]
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
                a1m, _ = chat_full(url, [{"role": "system", "content": sys_a},
                                         {"role": "user", "content": q1}], thinking)
                a1 = a1m.get("content") or ""
                turn2 = [{"role": "system", "content": sys_a}, {"role": "user", "content": q1},
                         {"role": "assistant", "content": a1}, {"role": "user", "content": q2}]
                refm, _ = chat_full(url, turn2, thinking)
                rw_ref, _ = chat_full(url, rewritten, thinking, 384)
                rw_neg_ref, _ = chat_full(url, rewritten_neg, thinking, 384)
            finally:
                proc.terminate()
                proc.wait()
            proc, url = server(spec, log)
            try:
                a1bm, _ = chat_full(url, [{"role": "system", "content": sys_a},
                                          {"role": "user", "content": q1}], thinking)
                chat(url, [{"role": "system", "content": sys_b}, {"role": "user", "content": q1}], thinking)
                gotm, cached = chat_full(url, turn2, thinking)
                chat(url, hist, thinking)  # the conversation that gets compacted
                rw, rw_cached = chat_full(url, rewritten, thinking, 384)
                rw_neg, _ = chat_full(url, rewritten_neg, thinking, 384)
            finally:
                proc.terminate()
                proc.wait()
            lines = log.read_text()
            ok = answer(a1bm) == answer(a1m) and answer(gotm) == answer(refm) and cached >= 2048 \
                and "previous conversation saved" in lines and "restored" in lines
            print(f"{'ok  ' if ok else 'FAIL'} {tag}: turn 2 "
                  f"{'identical' if answer(gotm) == answer(refm) else 'DIFFERS'}, "
                  f"{cached} prompt tokens restored", flush=True)
            if not ok:
                failures.append(tag)
                print("   ref:", json.dumps(answer_text(refm)[:200]),
                      "\n   got:", json.dumps(answer_text(gotm)[:200]))
            # #286: the rewritten history, restored from the first checkpoint
            ok = answer(rw) == answer(rw_ref) and rw_cached >= 512
            print(f"{'ok  ' if ok else 'FAIL'} {tag}: rewritten history "
                  f"{'identical' if answer(rw) == answer(rw_ref) else 'DIFFERS'} to the cold prefill, "
                  f"{rw_cached} prompt tokens resumed", flush=True)
            if not ok:
                failures.append(f"{tag} (rewrite)")
                print("   ref:", json.dumps(answer_text(rw_ref)[:200]),
                      "\n   got:", json.dumps(answer_text(rw)[:200]))
            # and the negative control: no checkpoint usable, the caches restart
            ok = answer(rw_neg) == answer(rw_neg_ref)
            print(f"{'ok  ' if ok else 'FAIL'} {tag}: rewrite inside the first message "
                  f"{'identical' if ok else 'DIFFERS'} to the cold prefill", flush=True)
            if not ok:
                failures.append(f"{tag} (rewrite, first message)")
                print("   ref:", json.dumps(answer_text(rw_neg_ref)[:200]),
                      "\n   got:", json.dumps(answer_text(rw_neg)[:200]))

    # #339: a saved conversation with an image. The server prefills the text
    # before the first image while the CPU encodes it (#180); the next turn must
    # still restore the saved conversation, not prefill everything again from
    # that early point.
    if args.image is not None:
        image = Path(args.image)
        if not Path(args.mmproj).exists():
            sys.exit(f"--image needs the vision encoder: {args.mmproj} does not exist")
        url_data = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()
        # a long text before the image: the conversation must reach kSnapshotMin
        # (2048 tokens) to be saved at all, and `before` (the early prefill) must
        # be long enough that restoring a few more tokens is visibly different.
        first = [{"role": "system", "content": sys_a},
                 {"role": "user", "content": [
                     {"type": "image_url", "image_url": {"url": url_data}},
                     {"type": "text", "text": "In one sentence: what is in this image?"}]}]
        q_year = "Which year is this about?"

        def ask(url: str, msgs: list) -> tuple[str, int, int]:
            r = post(url + "/v1/chat/completions",
                     {"messages": msgs, "max_tokens": 160, "temperature": 0,
                      "chat_template_kwargs": {"enable_thinking": False}})
            usage = r["usage"]
            return (r["choices"][0]["message"].get("content") or "",
                    usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                    usage.get("prompt_tokens", 0))

        for spec in args.spec.split(","):
            tag = f"{spec}, image"
            log = Path(f"/tmp/check_conversations-{spec}-image.log")
            proc, url = server(spec, log, args.mmproj)
            try:
                ref_a, _, _ = ask(url, first)
                turn = first + [{"role": "assistant", "content": ref_a},
                                {"role": "user", "content": q_year}]
                ref, _, _ = ask(url, turn)
            finally:
                proc.terminate()
                proc.wait()
            proc, url = server(spec, log, args.mmproj)
            try:
                a, _, _ = ask(url, first)
                chat(url, [{"role": "system", "content": sys_b}, {"role": "user", "content": q1}], False)
                turn = first + [{"role": "assistant", "content": a}, {"role": "user", "content": q_year}]
                got, cached, prompt = ask(url, turn)
            finally:
                proc.terminate()
                proc.wait()
            lines = log.read_text()
            # the whole conversation is restored: only the new turn is prefilled,
            # not the image's rows and the text before them again (#339)
            fresh = prompt - cached
            ok = got == ref and fresh < 128 and "restored" in lines
            print(f"{'ok  ' if ok else 'FAIL'} {tag}: next turn "
                  f"{'identical' if got == ref else 'DIFFERS'} to the cold prefill, "
                  f"{cached} prompt tokens resumed ({fresh} prefilled)", flush=True)
            if not ok:
                failures.append(f"{tag} (image)")
                print("   ref:", json.dumps(ref[:200]), "\n   got:", json.dumps(got[:200]))
    if failures:
        sys.exit(f"{len(failures)} failure(s)")
    print("all conversation checks passed")


if __name__ == "__main__":
    main()
