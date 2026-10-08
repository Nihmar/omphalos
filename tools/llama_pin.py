"""The llama.cpp reference pin (#393): read bench/llama.cpp.pin, find the
checkout a path belongs to, and say whether it is the reference.

Every tool or script that runs llama.cpp calls `check()` (or the CLI) so that
a result says which llama.cpp it measured, and a checkout that drifted from the
pin is at least warned about. `OMPH_LLAMA_STRICT=1` makes the warning fatal.

    uv run python tools/llama_pin.py                  # the pin
    uv run python tools/llama_pin.py --check <path>   # e.g. <llama.cpp>/build-hip/bin
    uv run python tools/llama_pin.py --field tag
    uv run python tools/llama_pin.py --json

`<path>` may be anything inside the checkout: a source tree, a build
directory, or a built binary that does not exist yet (its parent is searched).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PIN_FILE = ROOT / "bench" / "llama.cpp.pin"


def load(path: Path = PIN_FILE) -> dict[str, str]:
    """The pin's `key = value` lines (comments and blanks ignored)."""
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def checkout_of(path: str | Path) -> Path | None:
    """The git checkout `path` lives in (a binary's parent when it is missing)."""
    p = Path(path)
    if not p.exists():
        p = p.parent
    p = p.resolve()
    if p.is_file():
        p = p.parent
    for candidate in (p, *p.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def checkout_of_binary(path: str | Path) -> Path | None:
    """The llama.cpp checkout a built binary links: a tool built in the
    omphalos tree (tools/native/dump_mtmd_logits) resolves its libllama from
    the checkout it was built against, which ldd names."""
    p = Path(path)
    if not p.is_file():
        return None
    r = subprocess.run(["ldd", str(p)], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        if not any(name in line for name in ("libllama", "libmtmd", "libggml")):
            continue
        for token in line.split():
            if token.startswith("/"):
                found = checkout_of(Path(token).parent)
                if found is not None:
                    return found
    return None


def checkout_for(path: str | Path) -> Path | None:
    """The checkout to judge `path` by: a binary's linked llama.cpp when it
    has one (a tool built in this tree), else the checkout it lives in."""
    return checkout_of_binary(path) or checkout_of(path)


def head_commit(checkout: Path) -> str | None:
    r = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True,
                       check=False)
    return r.stdout.strip() if r.returncode == 0 else None


def short(commit: str) -> str:
    return commit[:8]


def ref_commit(line: str) -> str:
    """The short commit inside a describe()/check() label (the results' key)."""
    if "@ " in line:
        return line.split("@ ", 1)[1].split(" ", 1)[0]
    return ""


def describe(commit: str, pin: dict[str, str] | None = None) -> str:
    """A one-line label for a checkout at `commit`, as the results carry it."""
    pin = pin or load()
    if commit == pin.get("commit"):
        return f"llama.cpp: {pin.get('tag', '?')} {pin.get('release', '?')} @ {short(commit)} (pin)"
    if commit == pin.get("overlay_commit"):
        return (f"llama.cpp: {pin.get('overlay_branch', 'overlay')} @ {short(commit)} "
                f"(RX 9060 XT overlay, base {short(pin.get('overlay_base', ''))}, "
                f"pin {pin.get('tag', '?')})")
    return (f"llama.cpp: @ {short(commit)} (NOT the pin: {pin.get('tag', '?')} "
            f"{pin.get('release', '?')} @ {short(pin.get('commit', ''))})")


def check(path: str | Path, *, strict: bool | None = None, quiet: bool = False) -> str:
    """The reference line for the checkout containing `path`. A checkout that
    is not the pin (or the fork overlay) is warned about on stderr, or raised
    with OMPH_LLAMA_STRICT=1 / strict=True."""
    pin = load()
    if strict is None:
        strict = os.environ.get("OMPH_LLAMA_STRICT", "") not in ("", "0")
    checkout = checkout_for(path)
    if checkout is None:
        line = f"llama.cpp: {path} (no checkout: cannot say which commit; the pin is {pin.get('tag', '?')})"
        if strict:
            raise SystemExit(line)
        if not quiet:
            print(f"omphalos: {line}", file=sys.stderr)
        return line
    commit = head_commit(checkout)
    if commit is None:
        line = f"llama.cpp: {checkout} (git rev-parse failed; the pin is {pin.get('tag', '?')})"
        if strict:
            raise SystemExit(line)
        if not quiet:
            print(f"omphalos: {line}", file=sys.stderr)
        return line
    line = describe(commit, pin)
    if "NOT the pin" in line:
        if strict:
            raise SystemExit(f"omphalos: {commit} at {checkout} is not the llama.cpp pin: "
                             f"{pin.get('tag')} {pin.get('release')} @ {pin.get('commit')}")
        if not quiet:
            print(f"omphalos: warning: {checkout} is at {short(commit)}, not the pin "
                  f"({pin.get('tag')} {pin.get('release')} @ {short(pin.get('commit', ''))}); results name "
                  f"{short(commit)}", file=sys.stderr)
    return line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", metavar="PATH", help="a checkout (or any path inside one): print its label")
    ap.add_argument("--field", help="print one field of the pin (url, tag, release, commit, ...)")
    ap.add_argument("--json", action="store_true", help="the pin as JSON")
    ap.add_argument("--strict", action="store_true", help="fail when the checkout is not the pin")
    ap.add_argument("--quiet", action="store_true", help="no warning on stderr")
    args = ap.parse_args()
    pin = load()
    if args.json:
        print(json.dumps(pin, indent=2, sort_keys=True))
        return
    if args.field:
        print(pin.get(args.field, ""))
        return
    if args.check:
        print(check(args.check, strict=args.strict or None, quiet=args.quiet))
        return
    for key in ("url", "tag", "release", "commit", "date", "overlay_url", "overlay_branch", "overlay_commit",
                "overlay_base", "overlay_date", "builds"):
        print(f"{key} = {pin.get(key, '')}")


if __name__ == "__main__":
    main()
