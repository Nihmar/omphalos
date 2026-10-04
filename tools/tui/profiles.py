"""The server profiles: built-in presets and the user's file (#304).

A profile is the form's values as JSON, so it can be reviewed in a diff, pasted
into a script, or written by hand. The presets are the recipes this repository
measured: ``coding-agent`` is the README's (#285/#298), ``long-context`` is the
safe-KV one, ``fast`` is greedy with the drafter, ``default`` is the server's
own defaults.
"""

from __future__ import annotations

import json
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


def load() -> dict[str, dict]:
    """The presets plus whatever the user saved, user keys winning."""
    out = {k: dict(v) for k, v in PRESETS.items()}
    if PROFILES_PATH.exists():
        try:
            raw = json.loads(PROFILES_PATH.read_text())
        except json.JSONDecodeError:
            return out
        if isinstance(raw, dict):
            for name, values in raw.items():
                if isinstance(values, dict):
                    out[name] = {**schema.defaults(), **values}
    return out


def save(name: str, values: dict) -> Path:
    """Write one profile into the user's file, keeping the others."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if PROFILES_PATH.exists():
        try:
            raw = json.loads(PROFILES_PATH.read_text())
            if isinstance(raw, dict):
                existing = {k: v for k, v in raw.items() if isinstance(v, dict)}
        except json.JSONDecodeError:
            existing = {}
    existing[name] = {k: v for k, v in values.items() if k in schema.BY_KEY}
    PROFILES_PATH.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    return PROFILES_PATH


def delete(name: str) -> bool:
    if not PROFILES_PATH.exists():
        return False
    try:
        raw = json.loads(PROFILES_PATH.read_text())
    except json.JSONDecodeError:
        return False
    if not isinstance(raw, dict) or name not in raw:
        return False
    del raw[name]
    PROFILES_PATH.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    return True
