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
# The summary since c435ca5 (#318) names the checkpoints' time: the regex must
# take it and the pre-#318 shape (#347).
REQUEST_CHECKPOINTS = ("POST /v1/chat/completions: sampling greedy; prompt 378 tokens (63 cached) "
                       "in 370 ms (checkpoints 0 ms, 850.4 t/s); 48 tokens in 249 ms "
                       "(193.1 t/s, drafts accepted 45 / 53); stop: length")
REQUEST_CHECKPOINTS_USED = ("POST /v1/chat/completions: sampling greedy; prompt 45206 tokens (45193 cached, "
                            "restored) in 4270 ms (checkpoints 3900 ms, 3560.6 t/s); 889 tokens in 21268 ms "
                            "(41.8 t/s, drafts accepted 612 / 1611); stop: end of generation")
BAD_VALUE = "--ctx wants an integer, got 'abc'"
UNKNOWN = "unknown option --nope"
REQUEST_ERROR = "POST /v1/chat/completions: a prompt is required"
ABLATION = "omphalos: ablation active: OMPH_SKIP_FFN -- results are not the engine's"


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


def test_request_with_checkpoints() -> None:
    """#318 added "(checkpoints N ms, ...)" to the summary; before #347 every
    such line fell through as raw, so the live row never finished."""
    f = log.parse_line(REQUEST_CHECKPOINTS).fields
    assert f["prompt_tokens"] == 378 and f["cached_tokens"] == 63
    assert abs(f["prefill_tps"] - 850.4) < 0.1 and f["accepted"] == 45 and f["drafted"] == 53
    f = log.parse_line(REQUEST_CHECKPOINTS_USED).fields
    assert f["restored"] and f["stop"] == "end of generation" and abs(f["prefill_tps"] - 3560.6) < 0.1
    # and the pre-#318 lines still parse (the fixtures of the other tests)
    assert log.parse_line(REQUEST).kind == "request"


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


def test_errors_without_the_error_prefix() -> None:
    """"errors only" used to count only "^error: ", so a refused option, an
    exception while serving and the ablation banner were hidden (#347)."""
    assert log.parse_line(BAD_VALUE).kind == "error"
    assert log.parse_line(UNKNOWN).kind == "error"
    ev = log.parse_line(REQUEST_ERROR)
    assert ev.kind == "error" and ev.fields["message"] == "a prompt is required"
    assert log.parse_line(ABLATION).kind == "warn"
    assert log.parse_line(ABLATION).fields["ablation"].startswith("OMPH_SKIP_FFN")
    # the request summary starts the same way but is not an error
    assert log.parse_line(REQUEST).kind == "request"


def test_json_events() -> None:
    """--log-json: the same fields, plus the human line the pane shows."""
    ready = log.parse_line(json.dumps({"event": "ready", "model": "m.omph", "host": "127.0.0.1",
                                       "port": 7070, "context": 131072, "load_ms": 1610}))
    assert ready.kind == "ready" and ready.fields["port"] == 7070
    prog = log.parse_line(json.dumps({"event": "progress", "id": 1, "phase": "decode", "tokens": 1536,
                                      "t_s": 40.9, "last_t_s": 42.1, "accepted": 980, "drafted": 2480}))
    assert prog.kind == "progress" and prog.fields["tokens"] == 1536
    assert abs(prog.fields["accepted_pct"] - 39.5) < 0.1
    assert prog.fields["phase"] == "decode" and abs(prog.fields["last_tps"] - 42.1) < 0.05
    req = log.parse_line(json.dumps({
        "event": "request", "id": 4, "path": "/v1/chat/completions", "method": "POST",
        "sampling_line": "temp 0.60 top-k 20 top-p 0.95 min-p 0.00",
        "prompt_tokens": 45206, "cached_tokens": 45193, "restored": True, "prefill_ms": 62,
        "completion_tokens": 889, "decode_ms": 21268, "drafts": 1611, "accepted": 612, "stop": "length"}))
    assert req.kind == "request"
    assert req.fields["drafted"] == 1611 and req.fields["restored"]
    assert abs(req.fields["decode_tps"] - 41.8) < 0.1
    assert log.sampling_summary(req.fields["sampling"]) == "0.6/20/0.95"
    start = log.parse_line(json.dumps({"event": "request_start", "id": 4, "method": "POST",
                                       "path": "/v1/chat/completions"}))
    assert start.kind == "raw" and "chat/completions" in start.text
    http = log.parse_line(json.dumps({"event": "http", "status": 400, "path": "/v1/completions",
                                      "error": "frequency_penalty is not implemented"}))
    assert http.kind == "error" and http.fields["status"] == 400


def test_json_prefill_progress() -> None:
    """The JSON prefill event fills the same live row as the human line (#310):
    the phase and the total have to survive _from_json."""
    ev = log.parse_line(json.dumps({"event": "progress", "id": 1, "phase": "prefill", "tokens": 12288,
                                    "total": 98255, "t_s": 820.4}))
    assert ev.kind == "progress"
    assert ev.fields["phase"] == "prefill" and ev.fields["total"] == 98255
    assert ev.fields["tokens"] == 12288 and abs(ev.fields["tps"] - 820.4) < 0.05
    assert "prefill 12288 / 98255 tokens" in ev.text


def test_sampling_summary() -> None:
    assert log.sampling_summary("temp 0.60 top-k 20 top-p 0.95 min-p 0.00") == "0.6/20/0.95"
    assert log.sampling_summary("temp 0.60 top-k 20 top-p 0.95 min-p 0.00 repeat 1.10") == "0.6/20/0.95 r1.1"
    assert log.sampling_summary("greedy") == "greedy"
