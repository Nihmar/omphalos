"""The server profiles: built-in presets and the user's file (#304).

A profile is the form's values as JSON, so it can be reviewed in a diff, pasted
into a script, or written by hand. The presets are the recipes this repository
measured: ``coding-agent`` is the README's (#285/#298), ``long-context`` is the
safe-KV one, ``fast`` is greedy with the drafter, ``default`` is the server's
own defaults; ``moe-chat`` and ``moe-coding-agent`` are the second model's
(#527). A profile's empty file field means its model family's file.

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

# The two models the engine runs (#527), each with its files in the repo's
# models/ (the .omph omph-convert writes; the DFlash2 drafter and the vision
# encoder are Qwen3.8's - Ornith has neither) and the family a file name says.
REPO_MODELS = Path(__file__).resolve().parents[2] / "models"
MODELS: dict[str, dict[str, Path | None]] = {
    "qwen3.8": {"model": REPO_MODELS / "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.omph",
                "dflash": REPO_MODELS / "Qwen3.8-27B-DFlash2-Q4_K_M.omph",
                "mmproj": REPO_MODELS / "mmproj-Qwen3.8-27B-BF16.gguf"},
    "ornith": {"model": REPO_MODELS / "Ornith-1.5-35B-A3B-Q4_K_M.omph", "dflash": None, "mmproj": None},
}
FILE_KEYS = ("model", "dflash", "mmproj")


def family(model: object) -> str:
    """The family of a model file, by its name (the dense model by default)."""
    return "ornith" if "ornith" in Path(str(model or "")).name.lower() else "qwen3.8"


def default_files(name: str) -> dict[str, str]:
    """A family's files that exist here ("" for a missing or absent one: the
    form leaves an optional file out rather than failing the preflight)."""
    return {k: str(p) if p is not None and p.is_file() else "" for k, p in MODELS[name].items()}


def with_files(values: dict) -> dict:
    """The profile's empty file fields filled with its model family's files
    (an empty field means "the default", #304; the model's family decides
    which, #527)."""
    out = dict(values)
    files = default_files(family(out.get("model")) if out.get("model") else "qwen3.8")
    for k in FILE_KEYS:
        if not out.get(k):
            out[k] = files[k]
    return out


def switch_model(values: dict, name: str) -> dict:
    """The same form on another model: its files, every other choice kept,
    except what the target cannot run (Qwen Sharp is Qwen3.8's template,
    refused for the Qwen3.5 one, #497)."""
    out = {**values, **default_files(name)}
    if name == "ornith" and out.get("chat_template") == "sharp":
        out["chat_template"] = "original"
    return out

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
    # the MoE model (#527): its own file; K4/V4 (no measurable NLL cost, 10 %
    # faster at depth, #516), a 1024 chunk for long prompts (the 128k prefill
    # in 97 vs 127 s, #516); MTP drafts 1 token per step on it by itself (#496)
    "moe-chat": {**schema.defaults(), "model": str(MODELS["ornith"]["model"]), "ctx": 32768, "kv_k4": True},
    "moe-coding-agent": {**schema.defaults(), "model": str(MODELS["ornith"]["model"]), "ctx": 131072,
                         "chunk": 1024, "kv_k4": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
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
