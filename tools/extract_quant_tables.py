#!/usr/bin/env python3
"""Extract ggml quantization tables from ggml-common.h into a Python module.

The IQ grids, sign/mask tables and kvalues_iq4nl are needed by the NumPy
dequantization port; they are embedded as base64 blobs so the tools project is
self-contained (no llama.cpp checkout needed at runtime).

usage: uv run python extract_quant_tables.py <ggml-common.h> [-o <out.py>]
"""

from __future__ import annotations

import argparse
import base64
import re
import textwrap
from pathlib import Path

import numpy as np

BEGIN_RE = re.compile(
    r"GGML_TABLE_BEGIN\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)"
)
END_RE = re.compile(r"GGML_TABLE_END\(\)")

WANTED = (
    "iq2xxs_grid",
    "iq2xs_grid",
    "iq2s_grid",
    "iq3xxs_grid",
    "iq3s_grid",
    "iq1s_grid",
    "ksigns_iq2xs",
    "kmask_iq2xs",
    "kvalues_iq4nl",
)

DTYPES = {
    "uint8_t": np.uint8,
    "int8_t": np.int8,
    "uint16_t": np.uint16,
    "int32_t": np.int32,
    "uint32_t": np.uint32,
    "uint64_t": np.uint64,
}


def parse_tables(path: Path) -> list[tuple[str, np.ndarray]]:
    text = path.read_text()
    found = []
    for match in BEGIN_RE.finditer(text):
        ctype, name, size = match.group(1), match.group(2), match.group(3)
        if name not in WANTED:
            continue
        end = END_RE.search(text, match.end())
        if end is None:
            raise SystemExit(f"{name}: GGML_TABLE_END not found")
        body = re.sub(r"//[^\n]*", "", text[match.end():end.start()])
        values = [int(v, 0) for v in re.findall(r"-?0[xX][0-9a-fA-F]+|-?\d+", body)]
        arr = np.array(values, dtype=DTYPES[ctype])
        expected = int(size) if size.isdigit() else len(arr)
        if len(values) != expected:
            raise SystemExit(f"{name}: expected {expected} values, found {len(values)}")
        found.append((name, arr))
    missing = [n for n in WANTED if n not in {n for n, _ in found}]
    if missing:
        raise SystemExit(f"tables not found in {path}: {missing}")
    return found


def write_module(tables: list[tuple[str, np.ndarray]], out: Path, source: str) -> None:
    lines = [
        '"""ggml quantization tables — generated, do not edit by hand.',
        "",
        f"Source: {source}",
        "Generator: tools/extract_quant_tables.py",
        '"""',
        "",
        "import base64",
        "",
        "import numpy as np",
        "",
        "",
        "def _table(dtype, blob: str) -> np.ndarray:",
        '    return np.frombuffer(base64.b64decode(blob), dtype=dtype)',
        "",
    ]
    for name, arr in tables:
        blob = base64.b64encode(arr.tobytes()).decode("ascii")
        chunks = textwrap.wrap(blob, 88)
        lines.append(f"{name.upper()} = _table(np.{arr.dtype.name},")
        for i, chunk in enumerate(chunks):
            suffix = ")" if i == len(chunks) - 1 else ""
            lines.append(f'    "{chunk}"{suffix}')
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract ggml quantization tables")
    ap.add_argument("header", type=Path, help="path to ggml-common.h")
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    tables = parse_tables(args.header)
    out = args.out or Path(__file__).resolve().parent / "src" / "omphalos_tools" / "quant_tables.py"
    write_module(tables, out, str(args.header))
    total = sum(a.nbytes for _, a in tables)
    print(f"wrote {out} ({len(tables)} tables, {total / 1024:.1f} KiB of data)")


if __name__ == "__main__":
    main()
