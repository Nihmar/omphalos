"""End-to-end check of omph-server (#156) with the official OpenAI client.

Starts the server (or uses one already running with --url), then checks:
the model listing; chat completions whole and streamed (the same text, the
same reasoning split, usage); a chat's next turn reusing the cached prompt;
a tool call round trip; stop strings and length limits; raw completions;
seeded sampling; the errors of invalid requests.

    uv run python check_server.py --server ../engine/build/omph-server \
        --model ../models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
    uv run python check_server.py --url http://127.0.0.1:8080
"""

import argparse
import base64
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import openai

from omph_model import omph_file

FAILURES: list[str] = []


def check(cond: bool, what: str) -> None:
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILURES.append(what)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_health(url: str, proc: subprocess.Popen | None, timeout: float) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            sys.exit(f"omph-server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    sys.exit("omph-server did not come up")


def raw_post(url: str, path: str, body: bytes) -> tuple[int, dict]:
    req = urllib.request.Request(url + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def stream_chat(client: openai.OpenAI, **kw) -> tuple[str, str, list, str | None, object]:
    reasoning, content, calls, finish, usage = "", "", [], None, None
    for chunk in client.chat.completions.create(stream=True, stream_options={"include_usage": True}, **kw):
        if chunk.usage is not None:
            usage = chunk.usage
        for ch in chunk.choices:
            d = ch.delta
            reasoning += getattr(d, "reasoning_content", None) or ""
            content += d.content or ""
            calls += d.tool_calls or []
            finish = ch.finish_reason or finish
    return reasoning, content, calls, finish, usage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="a running server (else one is started)")
    ap.add_argument("--server", default="../engine/build/omph-server")
    ap.add_argument("--model", default="../models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--image", help="also check images (the server gets --mmproj): an image of the moon landing "
                    "front page, e.g. llama.cpp's tools/mtmd/test-1.jpeg")
    ap.add_argument("--mmproj", default="../models/mmproj-Qwen3.8-27B-BF16.gguf")
    args = ap.parse_args()

    proc = None
    url = args.url
    if url is None:
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        vision = ["--mmproj", args.mmproj] if args.image else []
        proc = subprocess.Popen([args.server, omph_file(args.model), "--port", str(port), "--ctx", str(args.ctx), *vision])
    try:
        wait_health(url, proc, 600)
        run_checks(url)
        if args.image:
            run_image_checks(url, Path(args.image))
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(30)
    if FAILURES:
        sys.exit(f"{len(FAILURES)} check(s) failed")
    print("all server checks passed")


def run_checks(url: str) -> None:
    client = openai.OpenAI(base_url=url + "/v1", api_key="none", timeout=600)
    models = client.models.list().data
    check(len(models) == 1 and models[0].id != "", f"one model listed ({models[0].id if models else None})")
    model = models[0].id

    # thinking off: whole and streamed give the same answer
    q = [{"role": "user", "content": "What is 17 * 3? Answer with just the number."}]
    no_think = {"chat_template_kwargs": {"enable_thinking": False}}
    r = client.chat.completions.create(model=model, messages=q, max_tokens=32, extra_body=no_think)
    m = r.choices[0].message
    check("51" in (m.content or ""), f"answer: {m.content!r}")
    check(r.choices[0].finish_reason == "stop", f"finish_reason {r.choices[0].finish_reason}")
    check(getattr(m, "reasoning_content", None) is None, "no reasoning with thinking off")
    check(r.usage.prompt_tokens > 0 and 0 < r.usage.completion_tokens <= 32, f"usage {r.usage}")
    s_reasoning, s_content, _, s_finish, s_usage = stream_chat(
        client, model=model, messages=q, max_tokens=32, extra_body=no_think)
    check(s_content == m.content and s_finish == "stop", f"streamed answer: {s_content!r}")
    check(s_usage is not None and s_usage.completion_tokens == r.usage.completion_tokens, "streamed usage")

    # thinking on: reasoning split from the answer, the same when streamed
    q = [{"role": "user", "content": "Is 91 a prime number? Answer yes or no, then one sentence."}]
    effort = {"reasoning_effort": "low"}
    r = client.chat.completions.create(model=model, messages=q, max_tokens=2048, extra_body=effort)
    m = r.choices[0].message
    reasoning = getattr(m, "reasoning_content", None) or ""
    check(reasoning != "" and "</think>" not in reasoning and "<think>" not in (m.content or ""),
          f"reasoning split ({len(reasoning)} + {len(m.content or '')} chars, {r.choices[0].finish_reason})")
    check("no" in (m.content or "").lower(), f"answer: {m.content!r}")
    s_reasoning, s_content, _, _, _ = stream_chat(client, model=model, messages=q, max_tokens=2048,
                                                  extra_body=effort)
    check(s_reasoning == reasoning and s_content == m.content, "streamed reasoning and answer")

    # the next turn continues the cached sequence (the reasoning sent back)
    turn2 = q + [{"role": "assistant", "content": m.content, "reasoning_content": reasoning},
                 {"role": "user", "content": "And 97?"}]
    r2 = client.chat.completions.create(model=model, messages=turn2, max_tokens=2048, extra_body=effort)
    cached = r2.usage.prompt_tokens_details.cached_tokens
    check(cached >= r.usage.prompt_tokens, f"next turn: {cached} of {r2.usage.prompt_tokens} prompt tokens cached")
    check("yes" in (r2.choices[0].message.content or "").lower(), f"answer: {r2.choices[0].message.content!r}")

    # checkpoints (#158): a retried answer, and a next turn without the
    # reasoning, resume from the checkpoint before the generation prompt
    notes = (Path(__file__).resolve().parent.parent / "PLAN.md").read_text()[20000:24000]
    q = [{"role": "user", "content": "Summarize these notes in one sentence.\n\n" + notes}]
    r = client.chat.completions.create(model=model, messages=q, max_tokens=2048, extra_body=effort)
    again = client.chat.completions.create(model=model, messages=q, max_tokens=2048, extra_body=effort)
    n = r.usage.prompt_tokens
    cached = again.usage.prompt_tokens_details.cached_tokens
    check(n - 8 <= cached < n and again.choices[0].message.content == r.choices[0].message.content,
          f"retry: {cached} of {n} prompt tokens from a checkpoint, the same answer")
    turn2 = q + [{"role": "assistant", "content": r.choices[0].message.content},
                 {"role": "user", "content": "Shorter."}]
    r2 = client.chat.completions.create(model=model, messages=turn2, max_tokens=2048, extra_body=effort)
    cached = r2.usage.prompt_tokens_details.cached_tokens
    check(n - 8 <= cached < n, f"next turn without the reasoning: {cached} of {r2.usage.prompt_tokens} cached")

    # a tool call round trip
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "The weather forecast for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer", "description": "days ahead"}},
            "required": ["city", "days"]}}}]
    q = [{"role": "user", "content": "What will the weather be in Rome over the next 3 days?"}]
    r = client.chat.completions.create(model=model, messages=q, tools=tools, max_tokens=256, extra_body=no_think)
    m = r.choices[0].message
    calls = m.tool_calls or []
    args = json.loads(calls[0].function.arguments) if calls else {}
    check(r.choices[0].finish_reason == "tool_calls" and len(calls) == 1 and calls[0].function.name == "get_weather",
          f"tool call: {calls}")
    check(args.get("city", "").startswith("Rome") and args.get("days") == 3, f"arguments {args}")
    _, s_content, s_calls, s_finish, _ = stream_chat(client, model=model, messages=q, tools=tools, max_tokens=256,
                                                     extra_body=no_think)
    check(s_finish == "tool_calls" and len(s_calls) == 1 and s_calls[0].function.arguments == calls[0].function.arguments
          and s_content == (m.content or ""), "streamed tool call")
    if calls:
        q2 = q + [m.model_dump(exclude_none=True),
                  {"role": "tool", "tool_call_id": calls[0].id,
                   "content": json.dumps({"forecast": ["sunny 24C", "cloudy 21C", "rain 18C"]})}]
        r = client.chat.completions.create(model=model, messages=q2, tools=tools, max_tokens=256, extra_body=no_think)
        text = (r.choices[0].message.content or "").lower()
        check(r.choices[0].finish_reason == "stop" and "rain" in text, f"answer from the tool: {text!r}")

    # stop strings and limits
    r = client.completions.create(model=model, prompt="1, 2, 3, 4, 5, 6,", max_tokens=32, stop=[" 9"])
    t = r.choices[0].text
    check(r.choices[0].finish_reason == "stop" and t.startswith(" 7, 8,") and "9" not in t, f"stop string: {t!r}")
    r = client.completions.create(model=model, prompt="Once upon a time", max_tokens=8, echo=True)
    check(r.choices[0].finish_reason == "length" and r.usage.completion_tokens == 8
          and r.choices[0].text.startswith("Once upon a time"), f"length limit, echo: {r.choices[0].text!r}")
    chunks = [c.choices[0].text for c in client.completions.create(
        model=model, prompt="Once upon a time", max_tokens=8, stream=True) if c.choices]
    check("".join(chunks) == r.choices[0].text[len("Once upon a time"):], "streamed completion")

    # seeded sampling repeats
    kw = {"model": model, "messages": [{"role": "user", "content": "Name a color."}], "max_tokens": 16,
          "temperature": 0.9, "seed": 1234, "extra_body": no_think}
    a = client.chat.completions.create(**kw).choices[0].message.content
    b = client.chat.completions.create(**kw).choices[0].message.content
    check(a == b and a, f"seeded sampling repeats: {a!r}")

    # errors
    code, body = raw_post(url, "/v1/chat/completions", b"{not json")
    check(code == 400 and "error" in body, "invalid JSON: 400")
    for what, extra in [("n = 2", {"n": 2}), ("a non-base64 image URL", None)]:
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,"}}]}] \
            if extra is None else [{"role": "user", "content": "Hi"}]
        try:
            client.chat.completions.create(model=model, messages=msgs, **(extra or {}))
            check(False, f"{what}: rejected")
        except openai.BadRequestError:
            check(True, f"{what}: 400")
    code, body = raw_post(url, "/v1/embeddings", b"{}")
    check(code == 404 and "error" in body, "unknown endpoint: 404")


def run_image_checks(url: str, image: Path) -> None:
    client = openai.OpenAI(base_url=url + "/v1", api_key="none", timeout=600)
    model = client.models.list().data[0].id
    data = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()
    q = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data}},
        {"type": "text", "text": "What event is this newspaper front page about? One sentence."}]}]
    no_think = {"chat_template_kwargs": {"enable_thinking": False}}
    r = client.chat.completions.create(model=model, messages=q, max_tokens=96, extra_body=no_think)
    text = r.choices[0].message.content or ""
    check("moon" in text.lower(), f"image: {text!r} ({r.usage.prompt_tokens} prompt tokens)")
    _, s_content, _, _, _ = stream_chat(client, model=model, messages=q, max_tokens=96, extra_body=no_think)
    check(s_content == text, "image: streamed answer")
    turn2 = q + [{"role": "assistant", "content": text}, {"role": "user", "content": "Which year?"}]
    r2 = client.chat.completions.create(model=model, messages=turn2, max_tokens=64, extra_body=no_think)
    cached = r2.usage.prompt_tokens_details.cached_tokens
    check(cached >= r.usage.prompt_tokens and "1969" in (r2.choices[0].message.content or ""),
          f"image: next turn {cached} of {r2.usage.prompt_tokens} cached: {r2.choices[0].message.content!r}")
    try:
        client.chat.completions.create(model=model, max_tokens=8, messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,bm90IGFuIGltYWdl"}}]}])
        check(False, "image: undecodable data rejected")
    except openai.BadRequestError:
        check(True, "image: undecodable data: 400")


if __name__ == "__main__":
    main()
