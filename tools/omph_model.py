"""The .omph file the engine loads, next to a GGUF (#178).

The engine binaries that run the model (omph-run, omph-generate, omph-server)
read only the .omph file omph-convert writes from the GGUF; the validation
tools that compare against the GGUF (gguf-py, llama.cpp, omph-dequant) keep
the GGUF path. Scripts take the GGUF and derive the .omph from it.
"""

import sys
from pathlib import Path


def omph_file(model: str) -> str:
    """The .omph next to `model` (a .gguf), or `model` itself if it is one."""
    p = Path(model)
    if p.suffix == ".omph":
        return str(p)
    o = p.with_suffix(".omph")
    if not o.exists():
        sys.exit(f"{o} not found: convert the model first (engine/build/omph-convert {p})")
    return str(o)
