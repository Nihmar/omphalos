"""Compare the engine's chat template (omph-tokenize --chat) with the GGUF's own Jinja template
rendered by jinja2 the way HF transformers renders it (#150): byte for byte, on a suite of
requests, including the ones the template rejects.

usage: uv run python check_chat_template.py <model.gguf>
"""

import json
import pathlib
import subprocess
import sys

import jinja2
import jinja2.ext
from gguf import GGUFReader
from jinja2.sandbox import ImmutableSandboxedEnvironment

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


def main() -> int:
    model = sys.argv[1]
    tpl = template(model)
    bad = 0
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
    print(f"{len(CASES)} requests: {bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
