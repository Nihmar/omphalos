#!/usr/bin/env python3
"""Lint the Markdown math for what GitHub's renderer mangles (#89, #105).

GitHub applies CommonMark backslash escapes inside $...$ before MathJax sees
it, so a backslash followed by ASCII punctuation (\\, \\! \\{ \\} \\\\ \\& ...) is
eaten or changed; and $$...$$ blocks are not recognised in every position.
The rules that render (#106): display math in ```math fences, inline math
without backslash-punctuation. This checks both, offline.

usage: uv run python check_doc_math.py [FILE.md ...]   (default: ../docs/*.md)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# A backslash before ASCII punctuation: a CommonMark escape.
ESCAPE = re.compile(r"\\[!-/:-@\[-`{-~]")
INLINE = re.compile(r"(?<![\\$])\$(?!\$)(.+?)(?<![\\$])\$(?!\$)")
CODE_SPAN = re.compile(r"`[^`]*`")


def check(path: Path) -> list[str]:
    problems: list[str] = []
    fence: str | None = None
    for n, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            fence = None if fence is not None else stripped[3:].strip() or "code"
            continue
        if fence is not None:
            continue  # code, or ```math (passed to MathJax verbatim)
        text = CODE_SPAN.sub("", line)
        if "$$" in text:
            problems.append(f"{path}:{n}: $$ block math; use a ```math fence")
        for m in INLINE.finditer(text):
            for e in ESCAPE.finditer(m.group(1)):
                problems.append(f"{path}:{n}: '{e.group(0)}' in inline math ${m.group(1)}$")
    if fence is not None:
        problems.append(f"{path}: unclosed ``` fence")
    return problems


def main() -> int:
    args = sys.argv[1:]
    paths = [Path(a) for a in args] or sorted((Path(__file__).parent / "../docs").glob("*.md"))
    problems = [p for path in paths for p in check(path)]
    for p in problems:
        print(p)
    print(f"{len(paths)} file(s), {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
