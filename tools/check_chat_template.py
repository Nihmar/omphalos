"""Compare the engine's chat template (omph-tokenize --chat) with the GGUF's own Jinja template
rendered by jinja2 the way HF transformers renders it (#150): byte for byte, on a suite of
requests, including the ones the template rejects.

With a second argument (llama.cpp's llama-tokenize), the chat *ids* of the same rendering are
compared too, for the requests whose own text holds no added-token string (#344: the engine keeps
`<tool_response>` text, llama.cpp turns it into a token). This is what catches a pre-tokenizer cut
at a segment boundary: "...does." + "\n" must tokenize like llama.cpp's one-pass rendering.

usage: uv run python check_chat_template.py <model.gguf> [<llama-tokenize>] [--template original|sharp]

--template sharp (#392) renders the vendored engine/templates/qwen-sharp-v22.5.0.jinja with the engine's
second renderer and compares it the same way (the SHARP_CASES below cover the extra kwargs).
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
SHARP_TPL = REPO / "engine" / "templates" / "qwen-sharp-v22.5.0.jinja"

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

# Qwen Sharp's own features (#392); the same CASES also run against it.
SHARP_CASES = {
    "sharp: terse off": {"messages": [SYS, U], "terse": False, "add_generation_prompt": True},
    "sharp: marker think_off": {"messages": [{"role": "system", "content": "Be brief. <|think_off|>"}, U],
                                "add_generation_prompt": True},
    "sharp: marker in parts": {"messages": [{"role": "user",
                                              "content": [{"type": "text", "text": "hi <|think_low|>"}]}],
                               "add_generation_prompt": True},
    "sharp: effort max": {"messages": [SYS, U], "reasoning_effort": "max", "add_generation_prompt": True},
    "sharp: effort off": {"messages": [SYS, U], "reasoning_effort": "off", "add_generation_prompt": True},
    "sharp: explicit thinking": {"messages": [U, {"role": "assistant", "content": "answer",
                                                  "thinking": "step by step"}, U],
                                 "add_generation_prompt": True},
    "sharp: reasoning field": {"messages": [U, {"role": "assistant", "content": "answer",
                                                "reasoning": "why"}, U],
                               "add_generation_prompt": True},
    "sharp: reasoning in content": {"messages": [U, {"role": "assistant",
                                                     "content": "<think>\nplan\n</think>\n\nthe answer"}, U],
                                    "add_generation_prompt": True},
    "sharp: thinking tags in content": {"messages": [U, {"role": "assistant",
                                                         "content": "<thinking>\nplan\n</thinking>\nthe answer"}, U],
                                        "add_generation_prompt": True},
    "sharp: explicit reasoning wins": {"messages": [U, {"role": "assistant",
                                                          "content": "<think>\nold\n</think>\nanswer",
                                                          "reasoning_content": "explicit"}, U]},
    "sharp: tools json": {"messages": [SYS, U], "tools": TOOLS, "tool_call_format": "json",
                          "add_generation_prompt": True},
    "sharp: tools json no thinking": {"messages": [U], "tools": TOOLS, "tool_call_format": "json",
                                      "enable_thinking": False},
    "sharp: tool calls json": {"messages": [SYS, U, CALL, TOOL1], "tools": TOOLS,
                               "tool_call_format": "json"},
    "sharp: tool errors": {"messages": [U, CALL,
                                        {"role": "tool", "content": "Traceback (most recent call last):\nboom"},
                                        {"role": "tool", "content": "failed to open file"}],
                           "tools": TOOLS, "add_generation_prompt": True},
    "sharp: tool warn once": {"messages": [U, CALL, {"role": "tool", "content": "exit code: 1"}],
                              "tools": TOOLS, "add_generation_prompt": True},
    "sharp: tool success": {"messages": [U, CALL, {"role": "tool", "content": "ok\nexit code: 0"}],
                            "tools": TOOLS, "add_generation_prompt": True},
    "sharp: tool json payload": {"messages": [U, CALL, {"role": "tool", "content": '{"temp": 21}'}],
                                 "tools": TOOLS, "tool_call_format": "json"},
    "sharp: truncate args": {"messages": [U, CALL], "tools": TOOLS, "max_tool_arg_chars": 4},
    "sharp: truncate responses": {"messages": [U, CALL, {"role": "tool", "content": "x" * 700}],
                                  "max_tool_response_chars": 100, "add_generation_prompt": True},
    "sharp: suppress tools": {"messages": [SYS, U], "tools": TOOLS, "suppress_tool_instructions": True},
    "sharp: runtime protocol": {"messages": [{"role": "system", "content": "[TOOL_REQUEST] use tools"}, U],
                                "tools": TOOLS, "add_generation_prompt": True},
    "sharp: video_url": {"messages": [{"role": "user",
                                        "content": [{"type": "video_url", "video_url": {"url": "x"}}]}],
                         "add_generation_prompt": True},
    "sharp: developer": {"messages": [{"role": "developer", "content": "dev"}, U],
                         "add_generation_prompt": True},
    "sharp: system after user": {"messages": [U, SYS, {"role": "user", "content": "again"}]},
    "sharp: unknown role": {"messages": [U, {"role": "robot", "content": "x"}]},
    "sharp: preserve reasoning off": {"messages": [U, A, U, A], "preserve_reasoning": False},
    "sharp: auto disable with tools": {"messages": [SYS, U], "tools": TOOLS,
                                       "auto_disable_thinking_with_tools": True,
                                       "add_generation_prompt": True},
    "sharp: tool message first": {"messages": [{"role": "tool", "content": "x"}],
                                  "add_generation_prompt": True},
    "sharp: multi tool responses": {"messages": [U, CALL, TOOL1, TOOL2,
                                                   {"role": "assistant", "content": "done"}]},
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
    args = sys.argv[1:]
    template_name = "original"
    if "--template" in args:
        i = args.index("--template")
        if i + 1 >= len(args):
            sys.exit("--template needs original or sharp")
        template_name = args[i + 1]
        del args[i:i + 2]
    if not args:
        sys.exit("usage: check_chat_template.py <model.gguf> [<llama-tokenize>] [--template original|sharp]")
    model = args[0]
    llama = args[1] if len(args) > 1 else None
    if llama:
        print(llama_pin.check(llama), flush=True)  # the reference this check is against (#393)
    cases = dict(CASES)
    if template_name == "sharp":
        tpl = SHARP_TPL.read_text()
        cases.update(SHARP_CASES)
    else:
        tpl = template(model)
    extra = ["--chat-template", template_name]
    bad = 0
    ids_checked = 0
    for name, req in cases.items():
        try:
            ref = jinja_render(tpl, req)
        except Exception as e:  # noqa: BLE001 - any template error is a reference "error"
            ref = None
            ref_err = str(e)
        p = subprocess.run([str(ENGINE), model, "--chat", *extra], input=json.dumps(req).encode(),
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
        q = subprocess.run([str(ENGINE), model, "--chat-ids", *extra], input=json.dumps(req).encode(),
                           capture_output=True, check=False)
        mine_ids = [int(x) for x in q.stdout.split()] if q.returncode == 0 else None
        ids_checked += 1
        if mine_ids != ref_ids:
            bad += 1
            print(f"IDS MISMATCH {name}: engine {len(mine_ids or [])} ids, llama.cpp {len(ref_ids)}")
            if mine_ids is not None and len(mine_ids) == len(ref_ids):
                first = next((i for i, (a, b) in enumerate(zip(mine_ids, ref_ids, strict=True)) if a != b), 0)
                print(f"  first at {first}: engine {mine_ids[first:first + 6]}, llama {ref_ids[first:first + 6]}")
    print(f"{len(cases)} requests ({template_name}): {bad} mismatches ({ids_checked} id comparisons)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
