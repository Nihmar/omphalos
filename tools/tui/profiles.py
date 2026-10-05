"""The server profiles: built-in presets and the user's file (#304).

A profile is the form's values as JSON, so it can be reviewed in a diff, pasted
into a script, or written by hand. The presets are the recipes this repository
measured: ``coding-agent`` is the README's (#285/#298), ``long-context`` is the
safe-KV one, ``fast`` is greedy with the drafter, ``default`` is the server's
own defaults.

The file is written atomically and never over a file this module cannot parse:
a corrupt ``profiles.json`` used to be replaced by whatever one save carried,
losing every profile in it (#347).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import schema

CONFIG_DIR = Path.home() / ".config" / "omphalos"
PROFILES_PATH = CONFIG_DIR / "profiles.json"

PRESETS: dict[str, dict] = {
    "default": {**schema.defaults(), "ctx": 8192},
    "coding-agent": {**schema.defaults(), "ctx": 131072, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                     "repeat_penalty": 1.1, "repeat_last_n": 1024, "max_tokens": 16384},
    "long-context": {**schema.defaults(), "ctx": 131072, "kv_k4_layers": "none", "cache_ram": 4096,
                     "temperature": 0.6, "top_p": 0.95, "top_k": 20, "repeat_penalty": 1.1,
                     "repeat_last_n": 1024, "max_tokens": 16384},
    "fast": {**schema.defaults(), "ctx": 32768, "cache_ram": 0, "kv_ram": 0},
    "vision": {**schema.defaults(), "ctx": 131072, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
               "repeat_penalty": 1.1, "repeat_last_n": 1024, "max_tokens": 16384},
}


def _read_user() -> dict | None:
    """The user's profiles, or None when the file exists but is not readable
    JSON (a save must not wipe it, #347)."""
    if not PROFILES_PATH.exists():
        return {}
    try:
        raw = json.loads(PROFILES_PATH.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


def _write_user(profiles: dict) -> None:
    """Atomic: a crash mid-write leaves the previous file, not half a file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PROFILES_PATH.with_name(PROFILES_PATH.name + ".tmp")
    tmp.write_text(json.dumps(profiles, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, PROFILES_PATH)


def load() -> dict[str, dict]:
    """The presets plus whatever the user saved, user keys winning."""
    out = {k: dict(v) for k, v in PRESETS.items()}
    user = _read_user()
    for name, values in (user or {}).items():
        out[name] = {**schema.defaults(), **values}
    return out


def save(name: str, values: dict) -> Path | None:
    """Write one profile into the user's file, keeping the others. Returns the
    path, or None when the file exists and cannot be parsed."""
    existing = _read_user()
    if existing is None:
        return None
    existing[name] = {k: v for k, v in values.items() if k in schema.BY_KEY}
    _write_user(existing)
    return PROFILES_PATH


def delete(name: str) -> bool:
    existing = _read_user()
    if not existing or name not in existing:
        return False
    del existing[name]
    _write_user(existing)
    return True
