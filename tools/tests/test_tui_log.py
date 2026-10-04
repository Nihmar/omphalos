"""The log parser, on the lines real runs produced (#304)."""

from __future__ import annotations

import json

from tui import log

READY = "omph-server: Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.omph on http://127.0.0.1:7070 (context 131072)"

REQUEST = ("POST /v1/chat/completions: sampling temp 0.60 top-k 20 top-p 0.95 min-p 0.00 repeat 1.10 "
           "freq 0.00 pres 0.00 last-n 1024 seed 20261004; prompt 98255 tokens (0 cached) in 147138 ms "
           "(667.8 t/s); 4096 tokens in 104232 ms (39.3 t/s, drafts accepted 2762 / 10939); stop: length")

RESUMED = ("POST /v1/chat/completions: sampling temp 0.60 top-k 20 top-p 0.95 min-p 0.00 seed 1; "
           "prompt 45206 tokens (45193 cached, restored) in 62 ms (209.7 t/s); 889 tokens in 21268 ms "
           "(41.8 t/s, drafts accepted 612 / 1611); stop: end of generation")

GONE = ("POST /v1/chat/completions: sampling greedy; prompt 19 tokens (0 cached) in 12 ms (1583.3 t/s); "
        "7 tokens in 210 ms (33.3 t/s, drafts accepted 6 / 12); stop: stopped (client gone)")

PROGRESS = "  2048 tokens, 39.1 t/s (last 3 s: 40.7 t/s), drafts accepted 71 %"
PREFILL = "  prefill 12288 / 98255 tokens, 820.4 t/s"
PROGRESS_PLAIN = "  512 tokens, 41.8 t/s (last 3 s: 41.8 t/s)"
LISTEN = "error: cannot listen on 127.0.0.1:7070: Address already in use"


def test_ready() -> None:
    ev = log.parse_line(READY)
    assert ev.kind == "ready"
    assert ev.fields["model"] == "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.omph"
    assert ev.fields["port"] == 7070 and ev.fields["context"] == 131072


def test_request() -> None:
    ev = log.parse_line(REQUEST)
    assert ev.kind == "request"
    f = ev.fields
    assert (f["method"], f["path"]) == ("POST", "/v1/chat/completions")
    assert f["prompt_tokens"] == 98255 and f["cached_tokens"] == 0
    assert f["completion_tokens"] == 4096
    assert f["accepted"] == 2762 and f["drafted"] == 10939
    assert abs(f["decode_tps"] - 39.3) < 0.05 and abs(f["prefill_tps"] - 667.8) < 0.1
    assert f["stop"] == "length" and not f["client_gone"] and not f["restored"]


def test_request_resumed_and_gone() -> None:
    f = log.parse_line(RESUMED).fields
    assert f["restored"] and f["cached_tokens"] == 45193 and f["stop"] == "end of generation"
    gone = log.parse_line(GONE).fields
    assert gone["client_gone"] and gone["stop"] == "stopped"


def test_progress() -> None:
    f = log.parse_line(PROGRESS).fields
    assert (f["tokens"], f["tps"], f["accepted_pct"]) == (2048, 39.1, 71.0)
    assert "accepted_pct" not in log.parse_line(PROGRESS_PLAIN).fields


def test_prefill_progress() -> None:
    """#310: the prefill has its own progress line, so a long prompt is not
    minutes of silence."""
    f = log.parse_line(PREFILL).fields
    assert f["phase"] == "prefill" and f["tokens"] == 12288 and f["total"] == 98255
    assert abs(f["tps"] - 820.4) < 0.05
    assert log.parse_line(PROGRESS).fields["phase"] == "decode"


def test_error_and_raw() -> None:
    ev = log.parse_line(LISTEN)
    assert ev.kind == "error" and "Address already in use" in ev.fields["message"]
    assert log.parse_line("").kind == "raw"


def test_json_events() -> None:
    """--log-json: the same fields, plus the human line the pane shows."""
    ready = log.parse_line(json.dumps({"event": "ready", "model": "m.omph", "host": "127.0.0.1",
                                       "port": 7070, "context": 131072, "load_ms": 1610}))
    assert ready.kind == "ready" and ready.fields["port"] == 7070
    prog = log.parse_line(json.dumps({"event": "progress", "id": 1, "tokens": 1536, "t_s": 40.9,
                                      "accepted": 980, "drafted": 2480}))
    assert prog.kind == "progress" and prog.fields["tokens"] == 1536
    assert abs(prog.fields["accepted_pct"] - 39.5) < 0.1
    req = log.parse_line(json.dumps({
        "event": "request", "id": 4, "path": "/v1/chat/completions", "method": "POST",
        "prompt_tokens": 45206, "cached_tokens": 45193, "restored": True, "prefill_ms": 62,
        "completion_tokens": 889, "decode_ms": 21268, "drafts": 1611, "accepted": 612, "stop": "length"}))
    assert req.kind == "request"
    assert req.fields["drafted"] == 1611 and req.fields["restored"]
    assert abs(req.fields["decode_tps"] - 41.8) < 0.1
    http = log.parse_line(json.dumps({"event": "http", "status": 400, "path": "/v1/completions",
                                      "error": "frequency_penalty is not implemented"}))
    assert http.kind == "error" and http.fields["status"] == 400


def test_sampling_summary() -> None:
    assert log.sampling_summary("temp 0.60 top-k 20 top-p 0.95 min-p 0.00") == "0.6/20/0.95"
    assert log.sampling_summary("temp 0.60 top-k 20 top-p 0.95 min-p 0.00 repeat 1.10") == "0.6/20/0.95 r1.1"
    assert log.sampling_summary("greedy") == "greedy"
