"""The .omph file the engine loads, next to a GGUF (#178).

The engine binaries that run the model (omph-run, omph-generate, omph-server)
read only the .omph file omph-convert writes from the GGUF; the validation
tools that compare against the GGUF (gguf-py, llama.cpp, omph-dequant) keep
the GGUF path. Scripts take the GGUF and derive the .omph from it, and the
.omph's `omph.source_sha256` must still be the GGUF's (#346): a GGUF replaced
in place (an updated Hugging Face file with the same name) would otherwise be
compared against a stale conversion, silently.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

# The GGUF's SHA-256, keyed by path:size:mtime, so the 11 GiB are hashed once
# per file version (a full hash takes seconds; the .omph check runs from every
# tool).
_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "omphalos" / "gguf-sha256.json"
_MEM: dict[str, str] = {}


def _cache_load() -> dict:
    try:
        return json.loads(_CACHE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _cache_store(cache: dict) -> None:
    try:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        # keep the recent versions only
        while len(cache) > 8:
            cache.pop(next(iter(cache)))
        _CACHE.write_text(json.dumps(cache))
    except OSError:
        pass  # a cache that cannot be written is not an error


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, cached by its size and mtime."""
    st = path.stat()
    key = f"{path.resolve()}:{st.st_size}:{st.st_mtime_ns}"
    if key in _MEM:
        return _MEM[key]
    cache = _cache_load()
    if key in cache:
        _MEM[key] = cache[key]
        return cache[key]
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _MEM[key] = digest
    cache[key] = digest
    _cache_store(cache)
    return digest


# GGUF value type sizes (the .omph header is the GGUF header plus four keys)
_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _skip(f, vtype: int) -> None:
    if vtype == 8:  # string
        f.seek(int.from_bytes(f.read(8), "little"), 1)
    elif vtype == 9:  # array
        elem = int.from_bytes(f.read(4), "little")
        for _ in range(int.from_bytes(f.read(8), "little")):
            _skip(f, elem)
    else:
        f.seek(_FIXED.get(vtype, 0), 1)


def source_sha256(omph: Path) -> str | None:
    """The `omph.source_sha256` the converter recorded, or None when the file
    is not an .omph or predates the key."""
    try:
        with omph.open("rb") as f:
            if f.read(4) != b"OMPH":
                return None
            f.read(12)  # container version + tensor count
            n_kv = int.from_bytes(f.read(8), "little")
            for _ in range(n_kv):
                key = f.read(int.from_bytes(f.read(8), "little")).decode("utf-8", "replace")
                vtype = int.from_bytes(f.read(4), "little")
                if key == "omph.source_sha256" and vtype == 8:
                    return f.read(int.from_bytes(f.read(8), "little")).decode()
                _skip(f, vtype)
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return None


def omph_file(model: str) -> str:
    """The .omph next to `model` (a .gguf), or `model` itself if it is one.

    The .omph's recorded source digest is compared with the GGUF's: a mismatch
    means the conversion is stale and the tools would compare two different
    models, so it is a hard error (reconvert)."""
    p = Path(model)
    if p.suffix == ".omph":
        return str(p)
    o = p.with_suffix(".omph")
    if not o.exists():
        sys.exit(f"{o} not found: convert the model first (engine/build/omph-convert {p})")
    recorded = source_sha256(o)
    if recorded is not None:
        got = file_sha256(p)
        if got != recorded:
            sys.exit(f"{o} was converted from another {p.name} (recorded {recorded[:12]}..., the file is "
                     f"{got[:12]}...): reconvert it (engine/build/omph-convert {p})")
    return str(o)
