"""Compare the engine's tokenizer (omph-tokenize) with llama.cpp's (llama-tokenize), token for
token, and check that decoding gives the text back (#148).

usage: uv run python check_tokenizer.py <model.gguf> <llama-tokenize> [--fuzz N] [--seed S]

The corpus: whole files (wikitext-2 test, the repo's own C++ / Python / Markdown), handwritten
edge cases (contractions, whitespace and newline mixes, combining marks, scripts, emoji, special
tokens inside text) and N random strings drawn from a palette that mixes every class of the
pre-tokenizer regex. Each case runs with special-token parsing on and off.
"""

import argparse
import pathlib
import random
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
ENGINE = REPO / "engine" / "build" / "omph-tokenize"

EDGE = [
    "",
    " ",
    "\n",
    "\r\n",
    "a",
    "Hello world",
    " Hello  world   again    ",
    "It's I'M we'Re they'VE she'll HE'D you'd 'tis o'clock ''s",
    "line one\nline two\r\nline three\rfour\n\n\n  \n\t\n",
    "   leading and trailing   ",
    "tabs\tand\t\ttabs \t mixed\t \n",
    "x\n\n\n\ny",
    "a  \n  b",
    "numbers 1 12 123 1234 12345 3.14159 -42 1e10 0x1F ½ ² Ⅻ ٣ ३",
    "punct!!! ??? ... --- === *** ### @@@ $$$ %%% ^^^ &&& ((( ))) [[[ ]]] {{{ }}}",
    "emoji 😀😃😄 👍\U0001f3fd 👨\u200d👩\u200d👧\u200d👦 🇮🇹 ☕\ufe0f ❤\ufe0f\u200d🔥",
    "café naïve résumé coöperate Ångström e\u0301 a\u0308\u0304",
    "日本語のテキスト、カタカナ、ひらがな。中文文本测试。한국어 텍스트",
    "العربية النص مع أرقام ١٢٣ ٤٥٦",
    "ภาษาไทย ไม\u0e48ม\u0e35ช\u0e48องว\u0e48าง",
    "हिन\u094dदी पाठ, स\u0902स\u094dक\u0943तम\u094d",
    "Ελληνικά κείμενα, Русский текст, Українська",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nhello<|im_end|>",
    "text<|endoftext|>more<|im_start|><|im_start|>",
    "def f(x):\n    return x ** 2  # square\n\nprint(f(3))\n",
    "for (int i = 0; i < n; ++i) {\n\tsum += a[i] * b[i];\n}\n",
    "https://example.com/path?q=1&r=two#frag user@mail.example.org",
    "\u00a0nbsp\u2003em\u3000ideo\u2028ls\u2029ps\u0085nel",
    "zero\u200bwidth\u200djoiner\ufeffbom",
    "\x00\x01\x02 control \x1b[31m ansi \x7f",
]

PALETTE = (list("abcXYZ") + list("0123456789") + list(" \t\n\r") + list("'\"!?.,;:-_()[]{}<>/\\@#$%^&*+=~`|")
           + ["é", "ß", "\u0301", "\u0308", "日", "本", "ア", "한", "ع", "ไ", "\u0e31", "न", "\u094d", "Ω",
              "😀", "👍", "\U0001f3fd", "\u200d", "\u00a0", "\u3000", "\u2028", "½", "²", "٣", "s", "S",
              "re", "ll", "<|im_start|>", "<|im_end|>", "<think>"])


def run(cmd, data: bytes) -> bytes:
    return subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout


def engine_ids(model: str, text: bytes, special: bool) -> list[int]:
    cmd = [str(ENGINE), model] + ([] if special else ["--no-parse-special"])
    return [int(t) for t in run(cmd, text).split()]


def engine_decode(model: str, ids: list[int]) -> bytes:
    return run([str(ENGINE), model, "--decode"], " ".join(map(str, ids)).encode())


def llama_ids(llama: str, model: str, text: bytes, special: bool) -> list[int]:
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        f.write(text)
        f.flush()
        cmd = [llama, "-m", model, "-f", f.name, "--ids", "--no-escape", "--log-disable"]
        if not special:
            cmd.append("--no-parse-special")
        out = run(cmd, b"").decode().strip()
    return [int(t) for t in out.strip("[]").split(",") if t.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("llama_tokenize")
    ap.add_argument("--fuzz", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    cases: list[tuple[str, bytes]] = [(f"edge {i}", s.encode()) for i, s in enumerate(EDGE)]
    wiki = REPO / "models" / "datasets" / "wikitext-2-raw" / "wiki.test.raw"
    if wiki.exists():
        cases.append(("wikitext-2 test", wiki.read_bytes()))
    for p in [REPO / "PLAN.md", REPO / "AGENTS.md", REPO / "engine" / "src" / "kernels" / "attn.hip",
              REPO / "engine" / "src" / "model" / "runner.cc", REPO / "tools" / "check_tokenizer.py"]:
        cases.append((p.name, p.read_bytes()))
    rng = random.Random(args.seed)
    for i in range(args.fuzz):
        n = rng.randint(1, 60)
        cases.append((f"fuzz {i}", "".join(rng.choice(PALETTE) for _ in range(n)).encode()))

    bad = 0
    tokens = 0
    for name, text in cases:
        for special in (True, False):
            mine = engine_ids(args.model, text, special)
            ref = llama_ids(args.llama_tokenize, args.model, text, special)
            tokens += len(ref)
            if mine != ref:
                bad += 1
                k = next((j for j, (a, b) in enumerate(zip(mine, ref)) if a != b), min(len(mine), len(ref)))
                print(f"MISMATCH {name} special={special}: {len(mine)} vs {len(ref)} tokens, first at {k}: "
                      f"{mine[k:k + 6]} vs {ref[k:k + 6]}  text {text[:80]!r}")
            elif special and engine_decode(args.model, mine) != text:
                bad += 1
                print(f"ROUND TRIP {name}: decode differs from the text")
    print(f"{len(cases)} cases x 2 modes, {tokens} reference tokens: {bad} failures")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
