"""Parse what omph-server writes, in either format (#304).

Two sources, one event model: the human lines the server has always printed
(the per-request summary and the progress line every 3 seconds), and the one
JSON object per line that ``omph-server --log-json`` prints (the TUI prefers it
when the binary has the flag). Everything else is passed through as a raw line,
so nothing the server says is hidden.

    from tui import log
    for ev, text in log.parse_lines(lines):
        ...
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# "omph-server: <model> on http://127.0.0.1:7070 (context 131072)"
READY = re.compile(r"^omph-server: (?P<model>.+?) on http://(?P<host>[^:]+):(?P<port>\d+) "
                   r"\(context (?P<context>\d+)\)$")
# "POST /v1/chat/completions: sampling temp 0.60 ...; prompt 98255 tokens (0 cached, restored) in
#  146815 ms (669.2 t/s); 4096 tokens in 104232 ms (39.3 t/s, drafts accepted 2762 / 10939); stop: length"
REQUEST = re.compile(
    r"^(?P<method>GET|POST) (?P<path>[^:]+): sampling (?P<sampling>[^;]*); "
    r"prompt (?P<prompt_tokens>\d+) tokens \((?P<cached_tokens>\d+) cached(?P<flags>[^)]*)\) "
    r"in (?P<prefill_ms>[\d.]+) ms \((?P<prefill_tps>[\d.]+) t/s\); "
    r"(?P<completion_tokens>\d+) tokens in (?P<decode_ms>[\d.]+) ms "
    r"\((?P<decode_tps>[\d.]+) t/s, drafts accepted (?P<accepted>\d+) / (?P<drafted>\d+)\); "
    r"stop: (?P<stop>.*)$")
# "  prefill 12288 / 98255 tokens, 820.4 t/s" (#310)
PREFILL = re.compile(r"^  prefill (?P<done>\d+) / (?P<total>\d+) tokens, (?P<tps>[\d.]+) t/s$")
# "  2048 tokens, 39.1 t/s (last 3 s: 40.7 t/s), drafts accepted 71 %"
PROGRESS = re.compile(r"^  (?P<tokens>\d+) tokens, (?P<tps>[\d.]+) t/s "
                      r"\(last 3 s: (?P<last_tps>[\d.]+) t/s\)(, drafts accepted (?P<accepted_pct>[\d.]+) %)?$")
# "error: cannot listen on 127.0.0.1:7070: Address already in use"
FAILED = re.compile(r"^error: (?P<message>.*)$")


@dataclass
class Event:
    kind: str                    # ready | request | progress | error | raw
    text: str                    # what the log pane shows
    id: int | None = None        # the request id, assigned by the app
    fields: dict = field(default_factory=dict)


def _ts(ms: float | None, tokens: int | None) -> float | None:
    if not ms or tokens is None:
        return None
    return 1000.0 * tokens / ms


def parse_line(line: str) -> Event:
    line = line.rstrip("\n")
    stripped = line.strip()

    if stripped.startswith("{") and '"event"' in stripped:
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return Event("raw", line)
        return _from_json(obj)

    m = READY.match(stripped)
    if m:
        return Event("ready",
                     f"ready: {m['model']} on http://{m['host']}:{m['port']} (context {m['context']})",
                     fields={"model": m["model"], "host": m["host"], "port": int(m["port"]),
                             "context": int(m["context"])})

    m = REQUEST.match(stripped)
    if m:
        f = {k: m[k] for k in ("method", "path", "sampling")}
        for k in ("prompt_tokens", "cached_tokens", "completion_tokens", "drafted", "accepted"):
            f[k] = int(m[k])
        prefill_ms, decode_ms = float(m["prefill_ms"]), float(m["decode_ms"])
        f["prefill_ms"], f["decode_ms"] = prefill_ms, decode_ms
        f["prefill_tps"] = float(m["prefill_tps"])
        f["decode_tps"] = float(m["decode_tps"])
        stop = m["stop"]
        f["client_gone"] = stop.endswith("(client gone)")
        f["stop"] = stop.replace(" (client gone)", "")
        f["restored"] = "restored" in m["flags"]
        f["saved"] = "previous conversation saved" in m["flags"]
        return Event("request", stripped, fields=f)

    m = PREFILL.match(line)
    if m:
        f = {"phase": "prefill", "tokens": int(m["done"]), "total": int(m["total"]), "tps": float(m["tps"]),
             "last_tps": float(m["tps"])}
        return Event("progress", stripped, fields=f)

    m = PROGRESS.match(line)
    if m:
        f = {"phase": "decode", "tokens": int(m["tokens"]), "tps": float(m["tps"]),
             "last_tps": float(m["last_tps"])}
        if m["accepted_pct"] is not None:
            f["accepted_pct"] = float(m["accepted_pct"])
        return Event("progress", stripped, fields=f)

    m = FAILED.match(stripped)
    if m:
        return Event("error", stripped, fields={"message": m["message"]})

    return Event("raw", line)


def _from_json(obj: dict) -> Event:
    """A --log-json line, turned into the same event and a human line."""
    kind = obj.get("event", "")
    if kind == "ready":
        f = {k: obj.get(k) for k in ("model", "host", "port", "context", "load_ms")}
        return Event("ready", f"ready: {f['model']} on http://{f['host']}:{f['port']} "
                              f"(context {f['context']})", fields=f)
    if kind == "request_start":
        return Event("raw", f"start {obj.get('path', '')}", fields=dict(obj))
    if kind == "progress":
        f = {"tokens": obj.get("tokens"), "tps": obj.get("t_s"), "last_tps": obj.get("t_s")}
        if obj.get("drafted"):
            f["accepted_pct"] = 100.0 * (obj.get("accepted") or 0) / obj["drafted"]
        text = f"  {f['tokens']} tokens, {f['tps']:.1f} t/s"
        if f.get("accepted_pct") is not None:
            text += f", drafts accepted {f['accepted_pct']:.0f} %"
        return Event("progress", text, fields=f)
    if kind == "request":
        f = dict(obj)
        full = f.get("completion_tokens") or 0
        f["prompt_tokens"] = f.get("prompt_tokens") or 0
        f["cached_tokens"] = f.get("cached_tokens") or 0
        f["drafted"] = f.get("drafts") or 0
        f["accepted"] = f.get("accepted") or 0
        f.setdefault("prefill_tps", _ts(f.get("prefill_ms"), f["prompt_tokens"] - f["cached_tokens"]))
        f.setdefault("decode_tps", _ts(f.get("decode_ms"), full))
        f.setdefault("stop", f.get("stop") or "")
        f.setdefault("client_gone", bool(f.get("client_gone")))
        text = (f"{f.get('method', 'POST')} {f.get('path', '')}: sampling {f.get('sampling_line', '')}; "
                f"prompt {f['prompt_tokens']} tokens ({f['cached_tokens']} cached"
                f"{', restored' if f.get('restored') else ''}) in {f.get('prefill_ms', 0):.0f} ms "
                f"({f.get('prefill_tps') or 0:.1f} t/s); {full} tokens in {f.get('decode_ms', 0):.0f} ms "
                f"({f.get('decode_tps') or 0:.1f} t/s, drafts accepted {f['accepted']} / {f['drafted']}); "
                f"stop: {f['stop']}" + (" (client gone)" if f.get("client_gone") else ""))
        return Event("request", text, fields=f)
    if kind == "http":
        status = obj.get("status", 0)
        msg = obj.get("error", "")
        text = f"{status} {obj.get('path', '')}" + (f"  {msg}" if msg else "")
        return Event("error" if status >= 400 else "raw", text, fields=dict(obj))
    return Event("raw", json.dumps(obj))


HUMAN_SAMPLING = re.compile(r"temp ([\d.]+) top-k (\d+) top-p ([\d.]+) min-p ([\d.]+)")


def sampling_summary(sampling: str) -> str:
    """'temp 0.60 top-k 20 top-p 0.95 min-p 0.00 ...' as '0.6/20/0.95' plus the
    penalties when they are on, for the table's narrow column."""
    m = HUMAN_SAMPLING.match(sampling)
    if not m:
        return sampling[:18]
    out = f"{float(m[1]):g}/{m[2]}/{float(m[3]):g}"
    rep = re.search(r"repeat ([\d.]+)", sampling)
    if rep and float(rep[1]) != 1.0:
        out += f" r{float(rep[1]):g}"
    return out
