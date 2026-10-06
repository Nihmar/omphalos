"""End-to-end check of omph-server (#156) with the official OpenAI client.

Starts the server (or uses one already running with --url), then checks:
the model listing; chat completions whole and streamed (the same text, the
same reasoning split, usage); a chat's next turn reusing the cached prompt;
a tool call round trip; stop strings and length limits; raw completions;
seeded sampling; the errors of invalid requests; and the --log-json events the
TUI reads (#304).

    uv run python check_server.py --server ../engine/build/omph-server \
        --model ../models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
    uv run python check_server.py --url http://127.0.0.1:8080
"""

import argparse
import base64
import gzip
import json
import re
import socket
import subprocess
import sys
import tempfile
import threading
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


def raw_post(url: str, path: str, body: bytes, headers: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(url + path, data=body,
                                 headers={"Content-Type": "application/json", **(headers or {})})
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


def drain_stderr(proc: subprocess.Popen, lines: list[str]) -> None:
    """Keep the server's log live on this terminal and its lines for the check."""
    assert proc.stderr is not None
    for line in proc.stderr:
        lines.append(line)
        sys.stderr.write(line)


def check_json_log(lines: list[str]) -> None:
    """--log-json (#304): one JSON object per line, the shape tools/tui/log.py
    reads, instead of the ready line, the progress lines and the summary."""
    events: list[dict] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue  # the ablation banner, the image line, a startup error: still human
        try:
            events.append(json.loads(stripped))
        except json.JSONDecodeError:
            check(False, f"a --log-json line is not JSON: {stripped[:100]!r}")
            return
    ready = [e for e in events if e.get("event") == "ready"]
    check(len(ready) == 1 and ready[0].get("load_ms", 0) > 0 and ready[0].get("context", 0) > 0,
          f"ready event with the load time: {ready[0] if ready else None}")
    starts = [e for e in events if e.get("event") == "request_start"]
    reqs = [e for e in events if e.get("event") == "request"]
    check(bool(starts) and len({e["id"] for e in starts}) == len(starts),
          f"{len(starts)} request_start events, unique ids")
    start_ids = {e["id"] for e in starts}
    check(bool(reqs) and all(e.get("id") in start_ids for e in reqs),
          f"{len(reqs)} request summaries, each with its start")
    progress = [e for e in events if e.get("event") == "progress"]
    check(bool(progress) and all(e.get("phase") in ("prefill", "decode") for e in progress),
          f"{len(progress)} progress events, each with a phase")
    check(any(e.get("phase") == "decode" for e in progress), "at least one decode progress event")
    need = ("method", "path", "sampling_line", "prompt_tokens", "cached_tokens", "restored", "prefill_ms",
            "completion_tokens", "decode_ms", "drafts", "accepted", "stop", "client_gone")
    missing = sorted({k for e in reqs for k in need if k not in e})
    check(not missing, f"every summary carries the fields the TUI reads (missing {missing})")
    check(not [line for line in lines if line.startswith(("POST ", "GET ")) and ": sampling " in line],
          "--log-json replaced the human summary lines")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="a running server (else one is started)")
    ap.add_argument("--server", default="../engine/build/omph-server")
    ap.add_argument("--model", default="../models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--tools", default="", help="start the server with --tools (e.g. all) and check the agent "
                    "tools (#380); empty: check that /tools answers 403")
    ap.add_argument("--image", help="also check images (the server gets --mmproj): an image of the moon landing "
                    "front page, e.g. llama.cpp's tools/mtmd/test-1.jpeg")
    ap.add_argument("--mmproj", default="../models/mmproj-Qwen3.8-27B-BF16.gguf")
    args = ap.parse_args()

    proc = None
    log_lines: list[str] = []
    url = args.url
    if url is None:
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        vision = ["--mmproj", args.mmproj] if args.image else []
        proc = subprocess.Popen([args.server, omph_file(args.model), "--port", str(port), "--ctx", str(args.ctx),
                                 "--log-json", *(["--tools", args.tools] if args.tools else []), *vision],
                                stderr=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=drain_stderr, args=(proc, log_lines), daemon=True).start()
    try:
        wait_health(url, proc, 600)
        run_checks(url)
        run_webui_checks(url)
        run_tools_checks(url, args.tools)
        run_timings_checks(url)
        if args.image:
            run_image_checks(url, Path(args.image))
        if proc is not None:
            check_json_log(log_lines)
        else:
            print("skip --log-json checks: --url points at a server this script did not start")
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
    # #338: the access checks
    code, body = raw_post(url, "/v1/completions", b'{"prompt":"hi","max_tokens":1}',
                          {"Content-Type": "text/plain"})
    check(code == 415 and "error" in body, f"a text/plain POST: {code}")
    code, body = raw_post(url, "/v1/completions", b'{"prompt":"hi","max_tokens":1}',
                          {"Host": "evil.example"})
    check(code == 403 and "error" in body, f"a rebinding Host: {code}")
    # #338: a lone surrogate the JSON parser combined into CESU-8 cannot come
    # back as invalid UTF-8 (json.loads would raise on the raw bytes)
    code, body = raw_post(url, "/v1/completions", b'{"prompt":"\\ud800","max_tokens":1,"echo":true}')
    echoed = body.get("choices", [{}])[0].get("text", "")
    check(code == 200 and "\ufffd" in echoed, f"a lone surrogate in the echo: {echoed!r}")
    run_early_error_checks(url)
    run_abandon_checks(url)


def run_early_error_checks(url: str) -> None:
    """#364: an early 413/431 must reach the client even with its request still
    in flight; closing with unread data would reset the connection instead."""
    import socket
    from urllib.parse import urlsplit
    parts = urlsplit(url)
    host, port = parts.hostname or "127.0.0.1", parts.port or 80

    def status_of(payload: bytes) -> bytes:
        sock = socket.create_connection((host, port), timeout=10)
        try:
            sock.sendall(payload)
            head = b""
            while b"\r\n" not in head:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                head += chunk
            return head[:24]
        finally:
            sock.close()

    # > 64 KiB of headers with no terminator: the server answers 431 while the
    # client is still sending (the terminator in the same read would end the
    # loop and 411 instead)
    check(status_of(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nX-Big: " + b"a" * (70 << 10))
          .startswith(b"HTTP/1.1 431"), "a 70 KiB header block gets its 431")
    check(status_of(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                    b"Content-Type: application/json\r\nContent-Length: 100000000\r\n\r\n"
                    + b"x" * (1 << 20)).startswith(b"HTTP/1.1 413"),
          "a 100 MB Content-Length gets its 413")


def run_abandon_checks(url: str) -> None:
    """#338: a non-streamed request whose client is gone must stop; the server
    is free for the next connection (it serves one at a time)."""
    import socket
    from urllib.parse import urlsplit
    parts = urlsplit(url)
    host, port = parts.hostname or "127.0.0.1", parts.port or 80
    body = json.dumps({"prompt": "Repeat the word spam forever, one per line:", "max_tokens": 4096})
    sock = socket.create_connection((host, port), timeout=10)
    sock.sendall(f"POST /v1/completions HTTP/1.1\r\nHost: {host}:{port}\r\n"
                 f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n{body}".encode())
    time.sleep(2.0)  # it is generating by now
    sock.close()
    t0 = time.monotonic()
    ok = False
    while time.monotonic() - t0 < 15:
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                ok = r.status == 200
                break
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.5)
    check(ok, f"an abandoned request stopped ({time.monotonic() - t0:.1f} s until the next one)")


def raw_get(url: str, path: str, headers: dict | None = None) -> tuple[int, dict, bytes]:
    req = urllib.request.Request(url + path, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def stream_chunks(url: str, body: dict) -> list[dict]:
    """The SSE data objects of a streamed chat request, as they arrive."""
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out: list[dict] = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: ") and line[6:] != "[DONE]":
                out.append(json.loads(line[6:]))
    return out


def run_timings_checks(url: str) -> None:
    """#382: the web UI's live statistics. Every streamed chunk carries the
    running `timings` (with llama.cpp's fields) and the prefill sends
    `prompt_progress`."""
    notes = (Path(__file__).resolve().parent.parent / "PLAN.md").read_text()[20000:36000]
    chunks = stream_chunks(url, {"messages": [{"role": "user", "content": "Summarize in one sentence.\n\n" + notes}],
                                 "max_tokens": 48, "stream": True, "timings_per_token": True,
                                 "return_progress": True})
    timed = [c["timings"] for c in chunks if "timings" in c]
    timeline = [int(t["predicted_n"]) for t in timed]
    check(len(timeline) >= 2 and timeline == sorted(timeline),
          f"per-chunk timings, predicted_n rising: {timeline[:8]}...")
    check(all("prompt_n" in t and "prompt_ms" in t and "predicted_ms" in t for t in timed),
          "every chunk's timings carries the running counters")
    progress = [c["prompt_progress"] for c in chunks if "prompt_progress" in c]
    check(bool(progress) and int(progress[-1]["total"]) > 0 and 0 <= int(progress[-1]["cache"]) <= int(progress[-1]["processed"]),
          f"prompt_progress during the prefill: {progress[-1] if progress else None}")
    final = timed[-1] if timed else {}
    check("cache_n" in final and "predicted_per_second" in final and "prompt_per_second" in final,
          f"the final timings carries llama.cpp's fields: {sorted(final)}")
    check(int(final.get("predicted_n", -1)) == max(timeline or [0]), "the last chunk's predicted_n is the total")


def run_webui_checks(url: str) -> None:
    """The llama.cpp web UI omph-server serves (#378): the assets, /props, /slots."""
    status, headers, body = raw_get(url, "/", {"Accept-Encoding": "gzip"})
    if status == 404:
        print("skip the web UI checks: this build has no embedded assets (OMPH_WEBUI_DIR)")
        return
    check(status == 200 and headers.get("Content-Encoding") == "gzip",
          f"GET /: {status} {headers.get('Content-Encoding')} {headers.get('Content-Type')}")
    html = gzip.decompress(body).decode("utf-8", "replace")
    check("<html" in html.lower(), f"GET / is HTML ({len(html)} chars)")
    etag = headers.get("ETag", "")
    check(etag.startswith('"') and len(etag) == 66 and
          headers.get("Cross-Origin-Opener-Policy") == "same-origin" and
          headers.get("Cache-Control") == "no-cache", f"the index's headers (ETag {etag[:14]}...)")
    status2, _, _ = raw_get(url, "/", {"Accept-Encoding": "gzip", "If-None-Match": etag})
    check(status2 == 304, f"a revalidated index: {status2}")
    status3, _, _ = raw_get(url, "/")
    check(status3 == 415, f"a client that cannot gunzip the assets: {status3}")
    status4, _, _ = raw_get(url, "/no-such-asset.js", {"Accept-Encoding": "gzip"})
    check(status4 == 404, f"an unknown asset is a 404, not the index: {status4}")
    match = re.search(r'href="\./(_app/immutable/[^"]+\.js)"', html)
    check(match is not None, "the index links its hashed bundle")
    if match:
        status5, h5, _ = raw_get(url, "/" + match.group(1), {"Accept-Encoding": "gzip"})
        check(status5 == 200 and "immutable" in h5.get("Cache-Control", "") and
              "javascript" in h5.get("Content-Type", ""),
              f"the bundle: {status5} {h5.get('Cache-Control')}")
    status6, _, pbody = raw_get(url, "/props")
    try:
        props = json.loads(pbody)
    except json.JSONDecodeError:
        props = {}
    check(status6 == 200 and props.get("total_slots") == 1 and
          props.get("default_generation_settings", {}).get("n_ctx", 0) > 0,
          f"GET /props: {str(props)[:120]}")
    check(props.get("endpoint_slots") is True and props.get("endpoint_metrics") is False and
          props.get("model_alias") and len(props.get("chat_template") or "") > 0,
          "props flags: slots on, metrics off, alias and chat template present")
    status7, _, sbody = raw_get(url, "/slots")
    try:
        slots = json.loads(sbody)
    except json.JSONDecodeError:
        slots = []
    check(status7 == 200 and len(slots) == 1 and slots[0].get("n_ctx", 0) > 0, f"GET /slots: {sbody[:120]!r}")
    status8, _, _ = raw_get(url, "/metrics")
    check(status8 == 501, f"llama.cpp's /metrics is a 501 here: {status8}")


def run_tools_checks(url: str, enabled: str) -> None:
    """The server-side agent tools (#380): the listing, an invocation of each
    kind, and the 403 when the server was started without --tools."""
    req = urllib.request.Request(url + "/tools")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            listed = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            check(not enabled, f"GET /tools without --tools: 403 ({e.code})")
            return
        check(False, f"GET /tools: {e.code}")
        return
    if not enabled:
        check(False, "GET /tools answered without --tools")
        return
    names = [t["tool"] for t in listed]
    check(names == ["read_file", "file_glob_search", "grep_search", "exec_shell_command", "write_file",
                    "edit_file", "get_info"], f"GET /tools lists the seven: {names}")
    check(all(t["type"] == "server" and t["uses_cwd"] and "definition" in t for t in listed),
          "every entry has the UI's shape")
    with tempfile.TemporaryDirectory() as d:
        headers = {"x-tool-cwd": d}
        status, body = raw_post(url, "/tools", json.dumps(
            {"tool": "write_file", "params": {"path": "a.txt", "content": "one\ntwo\n"}}).encode(), headers)
        check(status == 200 and body.get("result") == "file written successfully", f"write_file: {body}")
        status, body = raw_post(url, "/tools", json.dumps(
            {"tool": "read_file", "params": {"path": "a.txt", "append_loc": True}}).encode(), headers)
        check(status == 200 and body.get("plain_text_response", "").startswith("1\u2192one"),
              f"read_file: {body}")
        status, body = raw_post(url, "/tools", json.dumps(
            {"tool": "edit_file", "params": {"path": "a.txt", "edits": [{"old_text": "two", "new_text": "TWO"}]}})
            .encode(), headers)
        check(status == 200 and body.get("edits_applied") == 1, f"edit_file: {body}")
        status, body = raw_post(url, "/tools", json.dumps(
            {"tool": "grep_search", "params": {"path": ".", "pattern": "TWO", "return_line_numbers": True}})
            .encode(), headers)
        check(status == 200 and "a.txt:2:TWO" in body.get("plain_text_response", ""), f"grep_search: {body}")
        status, body = raw_post(url, "/tools", json.dumps(
            {"tool": "file_glob_search", "params": {"path": ".", "include": "*.txt"}}).encode(), headers)
        check(status == 200 and body.get("entries") and body["entries"][0]["path"] == "a.txt",
              f"file_glob_search: {body}")
        status, body = raw_post(url, "/tools", json.dumps(
            {"tool": "exec_shell_command", "params": {"command": "echo hi"}}).encode(), headers)
        check(status == 200 and "hi" in body.get("plain_text_response", "") and
              "[exit code: 0]" in body.get("plain_text_response", ""), f"exec_shell_command: {body}")
        status, body = raw_post(url, "/tools", json.dumps({"tool": "get_info", "params": {}}).encode(), headers)
        check(status == 200 and body.get("cwd") == d and "Linux" in body.get("os", ""), f"get_info: {body}")
        status, body = raw_post(url, "/tools", json.dumps({"tool": "nope", "params": {}}).encode(), headers)
        check(status == 500, f"an unknown tool: {status}")


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
    # The n-gram drafts copy tokens of the sequence (#199): the text before an
    # image is repeated by the model, and its continuation (the image's
    # negative placeholder ids) must be cut, not verified (#334). Before the fix
    # this read the embedding table out of bounds and aborted the server.
    phrase = ("The silver wombat carries a lantern through the violet cave at midnight, "
              "and the pangolin follows the river to the sea.")
    q_echo = [{"role": "user", "content": [
        {"type": "text", "text": f"Repeat this exact sentence, and nothing else: {phrase} {phrase}"},
        {"type": "image_url", "image_url": {"url": data}},
        {"type": "text", "text": "Now repeat the sentence."}]}]
    r3 = client.chat.completions.create(model=model, messages=q_echo, max_tokens=48, extra_body=no_think)
    text3 = r3.choices[0].message.content or ""
    check(len(text3) > 0, f"image: the n-gram drafts did not crash the server ({text3!r})")
    assert r3.usage is not None
    check(r3.usage.completion_tokens > 0, "image: the n-gram case generated something")


if __name__ == "__main__":
    main()
