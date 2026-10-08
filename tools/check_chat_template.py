"""Compare the engine's chat template (omph-tokenize --chat) with the GGUF's own Jinja template
rendered by jinja2 the way HF transformers renders it (#150): byte for byte, on a suite of
requests, including the ones the template rejects.

With a second argument (llama.cpp's llama-tokenize), the chat *ids* of the same rendering are
compared too, for the requests whose own text holds no added-token string (#344: the engine keeps
`<tool_response>` text, llama.cpp turns it into a token). This is what catches a pre-tokenizer cut
at a segment boundary: "...does." + "\n" must tokenize like llama.cpp's one-pass rendering.

usage: uv run python check_chat_template.py <model.gguf> [<llama-tokenize>]
"""

import json
import pathlib
import subprocess
import sys
import tempfile

import jinja2
import jinja2.ext
from gguf import GGUFReader
from jinja2.sandbox import ImmutableSandboxedEnvironment

import llama_pin

REPO = pathlib.Path(__file__).resolve().parent.parent
ENGINE = REPO / "engine" / "build" / "omph-tokenize"

SYS = {"role": "system", "content": "You are a helpful assistant."}
U = {"role": "user", "content": "What is 2+2?"}
A = {"role": "assistant", "content": "4", "reasoning_content": "Simple arithmetic."}
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city, \"quoted\" & <tagged>",
                                      "parameters": {"type": "object", "properties": {
                                          "city": {"type": "string"}, "days": {"type": "integer", "minimum": 1}},
                                          "required": ["city"]}}},
    {"type": "function", "function": {"name": "nop", "parameters": {}}},
]
CALL = {"role": "assistant", "content": "Let me check.",
        "tool_calls": [{"type": "function", "function": {"name": "get_weather",
                                                         "arguments": {"city": "Roma", "days": 3, "flag": True,
                                                                       "extra": {"a": [1, 2.5, None]}}}},
                       {"function": {"name": "nop", "arguments": {}}}]}
TOOL1 = {"role": "tool", "content": "{\"temp\": 21}"}
TOOL2 = {"role": "tool", "content": "  sunny  "}

CASES = {
    "plain": {"messages": [U]},
    "generation prompt": {"messages": [U], "add_generation_prompt": True},
    "system": {"messages": [SYS, U], "add_generation_prompt": True},
    "empty system": {"messages": [{"role": "system", "content": "   "}, U], "add_generation_prompt": True},
    "thinking off": {"messages": [SYS, U], "add_generation_prompt": True, "enable_thinking": False},
    "thinking on": {"messages": [U], "add_generation_prompt": True, "enable_thinking": True},
    "effort low": {"messages": [SYS, U], "add_generation_prompt": True, "reasoning_effort": "low"},
    "effort medium": {"messages": [U], "add_generation_prompt": True, "reasoning_effort": "medium"},
    "effort medium no system": {"messages": [SYS, U], "reasoning_effort": "medium"},
    "multi-turn": {"messages": [SYS, U, A, {"role": "user", "content": "And 3+3?"}], "add_generation_prompt": True},
    "multi-turn, no preserve": {"messages": [SYS, U, A, {"role": "user", "content": "And 3+3?"}, A],
                                "preserve_thinking": False},
    "no reasoning content": {"messages": [U, {"role": "assistant", "content": "  hi  "}, U]},
    "tools": {"messages": [SYS, U], "tools": TOOLS, "add_generation_prompt": True},
    "tools, no system": {"messages": [U], "tools": TOOLS, "enable_thinking": False},
    "tool calls": {"messages": [SYS, U, CALL, TOOL1, TOOL2, {"role": "assistant", "content": "Sunny, 21."}],
                   "tools": TOOLS},
    "tool call, empty content": {"messages": [U, {"role": "assistant", "content": "",
                                                  "tool_calls": CALL["tool_calls"][:1]}, TOOL1],
                                 "add_generation_prompt": True},
    "tool response user": {"messages": [U, CALL, {"role": "user", "content": "<tool_response>x</tool_response>"}],
                           "preserve_thinking": False},
    "parts": {"messages": [{"role": "user", "content": [{"type": "text", "text": "Look: "},
                                                        {"type": "image", "image": "a.png"},
                                                        {"type": "text", "text": " and "},
                                                        {"type": "video", "video": "b.mp4"},
                                                        {"image_url": {"url": "x"}}]}],
              "add_generation_prompt": True},
    "parts, vision ids": {"messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "1"},
                                                                    {"type": "image"}]},
                                       {"role": "assistant", "content": "ok"},
                                       {"role": "user", "content": [{"type": "image"}, {"type": "video"}]}],
                          "add_vision_id": True},
    "null content": {"messages": [U, {"role": "assistant", "content": None}]},
    "unicode": {"messages": [{"role": "user", "content": "  café 日本 \U0001f600 　"}],
                "add_generation_prompt": True},
    # the template rejects these
    "error: no messages": {"messages": []},
    "error: system not first": {"messages": [U, SYS]},
    "error: bad role": {"messages": [U, {"role": "robot", "content": "x"}]},
    "error: only tool responses": {"messages": [{"role": "user", "content": "<tool_response>a</tool_response>"}]},
    "error: bad effort": {"messages": [U], "reasoning_effort": "max"},
    "error: image in system": {"messages": [{"role": "system", "content": [{"type": "image"}]}, U]},
}


def template(model: str) -> str:
    r = GGUFReader(model)
    f = r.fields["tokenizer.chat_template"]
    return bytes(f.parts[f.data[0]]).decode()


def jinja_render(tpl: str, req: dict) -> str:
    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(tpl).render(**req)


def has_request_specials(req: dict) -> bool:
    """Whether the request's own strings hold an added-token marker: those cases
    are not id-compared (the engine keeps them text since #292, llama.cpp does
    not, #344)."""
    markers = ("<|", "|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
               "<tool_response>", "</tool_response>")

    def strings(x):
        if isinstance(x, str):
            yield x
        elif isinstance(x, dict):
            for v in x.values():
                yield from strings(v)
        elif isinstance(x, list):
            for v in x:
                yield from strings(v)

    return any(m in s for s in strings(req) for m in markers)


def llama_ids(llama: str, model: str, text: str) -> list[int]:
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as f:
        f.write(text.encode())
        path = f.name
    try:
        p = subprocess.run([llama, "-m", model, "-f", path, "--ids", "--no-escape", "--log-disable"],
                           capture_output=True, check=True)
        return [int(t) for t in p.stdout.decode().strip().strip("[]").split(",") if t.strip()]
    finally:
        pathlib.Path(path).unlink()


def main() -> int:
    model = sys.argv[1]
    llama = sys.argv[2] if len(sys.argv) > 2 else None
    if llama:
        print(llama_pin.check(llama), flush=True)  # the reference this check is against (#393)
    tpl = template(model)
    bad = 0
    ids_checked = 0
    for name, req in CASES.items():
        try:
            ref = jinja_render(tpl, req)
        except Exception as e:  # noqa: BLE001 - any template error is a reference "error"
            ref = None
            ref_err = str(e)
        p = subprocess.run([str(ENGINE), model, "--chat"], input=json.dumps(req).encode(),
                           capture_output=True, check=False)
        mine = p.stdout.decode() if p.returncode == 0 else None
        if ref is None and mine is None:
            continue
        if ref is None or mine is None or ref != mine:
            bad += 1
            print(f"MISMATCH {name}:")
            print("  jinja :", repr(ref) if ref is not None else "error: " + ref_err)
            print("  engine:", repr(mine) if mine is not None else "error: " + p.stderr.decode().strip())
            continue
        if llama is None or has_request_specials(req) or "<|image_pad|>" in ref or \
                "<|video_pad|>" in ref or "<__media__>" in ref:
            continue  # no ids comparison for these
        ref_ids = llama_ids(llama, model, ref)
        q = subprocess.run([str(ENGINE), model, "--chat-ids"], input=json.dumps(req).encode(),
                           capture_output=True, check=False)
        mine_ids = [int(x) for x in q.stdout.split()] if q.returncode == 0 else None
        ids_checked += 1
        if mine_ids != ref_ids:
            bad += 1
            print(f"IDS MISMATCH {name}: engine {len(mine_ids or [])} ids, llama.cpp {len(ref_ids)}")
            if mine_ids is not None and len(mine_ids) == len(ref_ids):
                first = next((i for i, (a, b) in enumerate(zip(mine_ids, ref_ids, strict=True)) if a != b), 0)
                print(f"  first at {first}: engine {mine_ids[first:first + 6]}, llama {ref_ids[first:first + 6]}")
    print(f"{len(CASES)} requests: {bad} mismatches ({ids_checked} id comparisons)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
