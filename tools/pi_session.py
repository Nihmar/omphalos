"""Rebuild the request a pi coding agent sent, from its session JSONL (#287).

The loop-police post-mortem of 2026-10-04 needed the model's own context back:
the repetition happened in a turn whose prompt was 25k-110k tokens of a real
pi session, and a fresh single-turn prompt does not reproduce it
(bench/results/loop-rate-287.txt). pi's session file records the messages, the
system-prompt sections, the tool declarations and the compactions, and the
request the engine saw is a faithful conversion of them; this rebuilds it so
the replay can ask the model the same question.

The mapping is pi 1.0.2's own (dist/bundle, `convertMessages`):

- the system prompt is the merge of every system message up to the cut: the
  `content` texts in order, then the `sections` by name (later wins, null
  deletes), joined by blank lines;
- the tools are the declared ones (`toolsAdded`) minus the removed ones
  (`toolsRemoved`), in the OpenAI shape pi sends (name, description,
  parameters -- no `strict` for a provider that does not support it);
- `custom` messages (loop-police's warnings) are user turns;
- a compaction replaces the transcript with a prefix + summary + suffix
  message and the tail kept by `firstKeptEntryId`;
- an assistant's thinking goes to `reasoning_content`, its tool calls to
  `tool_calls` with a JSON-string argument object;
- a tool result is `role: "tool"` with its call id and the text joined by
  newlines (or "(no tool output)").

    uv run python pi_session.py <session.jsonl> --cut 2026-10-04T10:29:38Z [--json]
"""

import argparse
import json
import sys
from pathlib import Path

# pi's own markers around a compaction's summary (convertToLlm).
COMPACTION_PREFIX = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
COMPACTION_SUFFIX = "\n</summary>"

# the thinking signatures pi maps back to a request field
REASONING_FIELDS = ("reasoning", "reasoning_content", "reasoning_text")


def records(path: str | Path) -> list[dict]:
    """The session file's records, in order (malformed lines skipped)."""
    out = []
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def system_prompt(recs: list[dict]) -> tuple[str, list[dict]]:
    """pi's `getCurrentSystemMessage`: the merge of every system message so
    far, and the tools still declared (`getCurrentTools`)."""
    parts: list[str] = []
    sections: dict[str, object] = {}
    tools: dict[str, dict] = {}
    for r in recs:
        if r.get("type") != "message":
            continue
        m = r["message"]
        if m.get("role") != "system":
            continue
        text = m.get("content")
        if isinstance(text, str) and text:
            parts.append(text)
        for name, value in (m.get("sections") or {}).items():
            if value is None:
                sections.pop(name, None)
            else:
                sections[name] = value
        for t in m.get("toolsAdded") or []:
            tools[t["name"]] = t
        for t in m.get("toolsRemoved") or []:
            tools.pop(t["name"], None)
    text = "\n\n".join([p for p in parts if p] + [str(v) for v in sections.values() if v])
    return text, list(tools.values())


def transcript(recs: list[dict], cut: str) -> list[tuple[str, dict]]:
    """The context messages before `cut` (exclusive), as (record id, message).

    Compactions drop everything before their `firstKeptEntryId` and put their
    summary first, as pi does when it rebuilds a transcript."""
    live: list[tuple[str, dict]] = []
    key = cut[:19]  # second granularity: the records carry fractional seconds
    for r in recs:
        ts = r.get("timestamp", "")
        if key and ts and ts[:19] >= key:
            break
        kind = r.get("type")
        if kind == "message":
            live.append((r.get("id", ""), r["message"]))
        elif kind == "custom_message":
            content = r.get("content")
            live.append((r.get("id", ""), {
                "role": "custom",
                "content": [{"type": "text", "text": content}] if isinstance(content, str) else content,
            }))
        elif kind == "compaction" and r.get("summary"):
            kept = r.get("firstKeptEntryId")
            at = next((i for i, (rid, _) in enumerate(live) if rid == kept), 0 if kept is None else len(live))
            summary = (r.get("id", ""), {"role": "compactionSummary", "summary": r["summary"]})
            live = [summary, *live[at:]]
    return live


def _assistant(m: dict) -> dict:
    blocks = [b for b in (m.get("content") or []) if isinstance(b, dict)]
    texts = [b.get("text") or "" for b in blocks if b.get("type") == "text" and (b.get("text") or "").strip()]
    thinking = [b.get("thinking") or "" for b in blocks
                if b.get("type") == "thinking" and (b.get("thinking") or "").strip()]
    calls = [b for b in blocks if b.get("type") == "toolCall"]
    out: dict = {"role": "assistant", "content": [{"type": "text", "text": t} for t in texts] or None}
    signature = next((b.get("thinkingSignature") for b in blocks if b.get("type") == "thinking"), None)
    if thinking and (signature is None or signature in REASONING_FIELDS):
        out["reasoning_content"] = "\n".join(thinking)
    if calls:
        out["tool_calls"] = [{"id": c.get("id"), "type": "function",
                              "function": {"name": c.get("name"),
                                           "arguments": json.dumps(c.get("arguments") or {})}}
                             for c in calls]
    return out


def _tool_result(m: dict) -> dict:
    text = "\n".join(b.get("text") or "" for b in (m.get("content") or []) if isinstance(b, dict)
                     and b.get("type") == "text")
    return {"role": "tool", "content": text or "(no tool output)", "tool_call_id": m.get("toolCallId")}


def to_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t["parameters"]}}
            for t in tools]


def request_for_cut(recs: list[dict], cut: str) -> dict:
    """The OpenAI request pi sent for the turn starting at `cut`.

    pi's `transformMessages` drops an assistant message whose stop reason is
    "error" or "aborted" (an interrupt, or a loop-police truncation), and gives
    a tool call with no result a synthetic "No result provided" result: both
    are part of what the model saw, so both are reproduced here."""
    key = cut[:19]
    system, tools = system_prompt([r for r in recs if not key or r.get("timestamp", "")[:19] < key])
    messages = [{"role": "system", "content": system}] if system else []
    pending: list[tuple[str, str]] = []
    answered: set[str] = set()

    def close_pending() -> None:
        for call_id, name in pending:
            if call_id not in answered:
                messages.append({"role": "toolResult", "toolCallId": call_id, "toolName": name,
                                 "content": [{"type": "text", "text": "No result provided"}],
                                 "isError": True})
        pending.clear()
        answered.clear()

    for _, m in transcript(recs, cut):
        role = m.get("role")
        if role == "assistant":
            close_pending()
            if m.get("stopReason") in ("error", "aborted"):
                continue
            calls = [b for b in (m.get("content") or [])
                     if isinstance(b, dict) and b.get("type") == "toolCall"]
            if calls:
                pending.extend((c.get("id"), c.get("name")) for c in calls)
                answered.clear()
            messages.append(_assistant(m))
        elif role == "toolResult":
            answered.add(m.get("toolCallId"))
            messages.append(_tool_result(m))
        else:
            close_pending()
            if role == "user" or role == "custom":
                messages.append({"role": "user", "content": m.get("content")})
            elif role == "compactionSummary":
                messages.append({"role": "user",
                                 "content": [{"type": "text",
                                              "text": COMPACTION_PREFIX + m["summary"] + COMPACTION_SUFFIX}]})
    out = {"messages": messages}
    if tools:
        out["tools"] = to_tools(tools)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--cut", required=True, help="ISO timestamp (UTC, as in the file): the turn's start")
    ap.add_argument("--json", action="store_true", help="print the request body instead of a summary")
    args = ap.parse_args()
    recs = records(args.session)
    req = request_for_cut(recs, args.cut)
    if args.json:
        json.dump(req, sys.stdout)
        print()
        return
    msgs = req["messages"]
    print(f"{len(msgs)} messages, {len(req.get('tools', []))} tools")
    from collections import Counter
    print("  roles:", dict(Counter(m["role"] for m in msgs)))
    print("  system prompt:", len(msgs[0]["content"]), "chars")
    print("  last message:", json.dumps(msgs[-1])[:200])
    print("  tools:", [t["function"]["name"] for t in req.get("tools", [])])


if __name__ == "__main__":
    main()
