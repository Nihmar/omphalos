"""The omph-server options as data (#304).

The form, the command preview, the profiles and the consistency test all read
this table instead of repeating the server's flags: a flag that is not here
cannot be set from the TUI, and ``tools/tests/test_tui_schema.py`` fails when a
``--flag`` in the table is not in ``engine/src/tools/server.cc``'s parser (or
the other way round, for the flags the TUI means to expose).

An entry is either a command-line flag (``flag``) or one of the engine's
environment switches (``env``); the second kind is only written into the
environment when its value differs from the engine's default, so the command
preview stays readable. A switch the engine reads with ``flag()`` (set = on,
value ignored) is a ``bool`` and is **only ever written as ``NAME=1``**: writing
``NAME=0`` would turn it on (#328).
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

GROUPS = ("model", "sampling", "drafting", "cache", "vision", "advanced", "tools")


@dataclass(frozen=True)
class Option:
    key: str                     # profile key, widget id
    group: str
    kind: str                    # "flag" | "env"
    name: str                    # "--ctx" or "OMPH_MTP_WINDOW"
    type: str                    # "bool" | "int" | "float" | "str" | "path" | "enum"
    default: object
    help: str
    choices: tuple[str, ...] = ()
    note: str = ""               # shown in the form, after the help

    def is_default(self, value: object) -> bool:
        if self.type == "bool":
            return bool(value) == bool(self.default) or not value
        if self.type in ("str", "path"):
            return str(value or "") == str(self.default or "")
        return value == self.default


def _opt(*args, **kw) -> Option:
    return Option(*args, **kw)


# The order here is the order of the form, the command and the profile file.
SCHEMA: tuple[Option, ...] = (
    # --- model
    _opt("model", "model", "flag", "", "path", "", "the .omph omph-convert writes", note="required"),
    _opt("dflash", "drafting", "flag", "--dflash", "path", "",
         "a DFlash2 drafter .omph instead of the MTP block, 7 drafts per step"),
    _opt("mmproj", "vision", "flag", "--mmproj", "path", "",
         "the vision encoder (needs a build with OMPH_LLAMA_DIR)"),
    _opt("host", "model", "flag", "--host", "str", "127.0.0.1", "address to bind"),
    _opt("port", "model", "flag", "--port", "int", 8080, "port", note="the api"),
    _opt("alias", "model", "flag", "--alias", "str", "", "the model id in the api (default: the file name)"),
    _opt("ctx", "model", "flag", "--ctx", "int", 8192, "KV capacity in tokens",
         note="a longer prompt is a 400; ~26 KiB/token"),
    _opt("chunk", "model", "flag", "--chunk", "int", 512, "tokens per prefill chunk (activation buffers)"),
    # --- sampling
    _opt("temperature", "sampling", "flag", "--temp", "float", 0.0, "temperature; 0 is greedy",
         note="the fastest path"),
    _opt("top_k", "sampling", "flag", "--top-k", "int", 0, "top-k; 0 is off"),
    _opt("top_p", "sampling", "flag", "--top-p", "float", 1.0, "top-p; 1 is off"),
    _opt("min_p", "sampling", "flag", "--min-p", "float", 0.0, "min-p; 0 is off"),
    _opt("repeat_penalty", "sampling", "flag", "--repeat-penalty", "float", 1.0,
         "llama.cpp's penalty; 1 is off", note="#298: 1.1 with a long window"),
    _opt("repeat_last_n", "sampling", "flag", "--repeat-last-n", "int", 64,
         "the window the three penalties look at; 0 is off", note="1024 covers a paragraph"),
    _opt("frequency_penalty", "sampling", "flag", "--frequency-penalty", "float", 0.0, "OpenAI's frequency penalty"),
    _opt("presence_penalty", "sampling", "flag", "--presence-penalty", "float", 0.0, "OpenAI's presence penalty"),
    _opt("max_tokens", "sampling", "flag", "--max-tokens", "int", -1,
         "default cap per request; -1 runs until the context is full"),
    # --- drafting
    _opt("no_mtp", "drafting", "flag", "--no-mtp", "bool", False, "do not load the MTP block",
         note="-352 MiB of VRAM, no drafts"),
    _opt("mtp_window", "drafting", "env", "OMPH_MTP_WINDOW", "int", 16384,
         "the MTP block's KV window (16 sinks + the last N..2N)", note="-150 MiB at 128k"),
    _opt("draft_vocab", "drafting", "env", "OMPH_DRAFT_VOCAB", "int", 98304,
         "drafts over the first N token ids; 0 is the whole head"),
    _opt("ngram", "drafting", "env", "OMPH_NGRAM", "int", 1,
         "n-gram (prompt lookup) drafts; 0 is off"),
    _opt("ngram_min", "drafting", "env", "OMPH_NGRAM_MIN", "int", 4,
         "the fewest n-gram drafts that replace the model's"),
    _opt("dflash_keep", "drafting", "env", "OMPH_DFLASH_KEEP", "float", 0.12,
         "DFlash2 drafts position n while 1..n survive with probability >= P"),
    _opt("dflash_pmin", "drafting", "env", "OMPH_DFLASH_PMIN", "float", 0.0,
         "DFlash2 stops where the selector is below P; 0 is off"),
    # --- cache
    _opt("cache_ram", "cache", "flag", "--cache-ram", "int", 2048,
         "pinned host RAM for sequence checkpoints (MiB); 0 is none"),
    _opt("kv_ram", "cache", "flag", "--kv-ram", "int", 8192,
         "pinned host RAM for whole conversations (MiB); 0 is none"),
    _opt("kv_k4_layers", "cache", "env", "OMPH_KV_K4_LAYERS", "enum", "2,4,5,6,7,9,11,15",
         "K in Q4 on those attention layers (0-15), Q8 on the rest",
         choices=("2,4,5,6,7,9,11,15", "none", "2,5,7,15"), note="the default is the measured mix"),
    _opt("kv_k4", "cache", "env", "OMPH_KV_K4", "bool", False,
         "K4 on every layer: an experiment, over the q8_0/q4_0 KL budget"),
    _opt("kv_window", "cache", "env", "OMPH_KV_WINDOW", "int", 512,
         "the exact FP16 window of the quantized cache; 0 is off", note="512 since #318"),
    _opt("kv_f32", "cache", "env", "OMPH_KV_F32", "bool", False, "exact f32 KV in VRAM (reference)"),
    _opt("kv_host", "cache", "env", "OMPH_KV_HOST", "bool", False, "exact f32 KV in pinned host RAM (reference)"),
    # --- advanced
    _opt("api_key", "advanced", "flag", "--api-key", "str", "", "require Authorization: Bearer KEY"),
    _opt("cors", "advanced", "flag", "--cors", "str", "", "allow browser requests from ORIGIN (* for any)"),
    # --- tools
    _opt("tools", "tools", "flag", "--tools", "str", "",
         "llama.cpp's server tools the web UI lists (read_file, file_glob_search, "
         "grep_search, exec_shell_command, write_file, edit_file, get_info) or all",
         note="experimental: files and shell with the server's own permissions (#380)"),
    _opt("agent", "tools", "flag", "--agent", "bool", False, "enable every tool (--tools all)",
         note="llama.cpp's shortcut"),
    _opt("log_json", "advanced", "flag", "--log-json", "bool", False,
         "one JSON object per line on stderr for the ready line, the requests and their progress",
         note="#304: the TUI reads both this and the human lines"),
    _opt("gemm_min", "advanced", "env", "OMPH_GEMM_MIN", "int", 16, "tokens from which a run takes the GEMM path"),
    _opt("overlap", "advanced", "env", "OMPH_OVERLAP", "bool", False, "a side stream for sibling GEMVs"),
    _opt("host_argmax", "advanced", "env", "OMPH_HOST_ARGMAX", "bool", False, "greedy argmax on the host"),
    _opt("timing", "advanced", "env", "OMPH_TIMING", "bool", False, "VRAM and a per-step timing line on stderr"),
    _opt("no_fused_gemm", "advanced", "env", "OMPH_NO_FUSED_GEMM", "bool", False, "dequant + GEMM instead of the fused one"),
    _opt("no_swiglu_gemm", "advanced", "env", "OMPH_NO_SWIGLU_GEMM", "bool", False, "SwiGLU after the up GEMM"),
    _opt("no_group", "advanced", "env", "OMPH_NO_GROUP", "bool", False, "sibling GEMVs as separate launches"),
    _opt("no_b4", "advanced", "env", "OMPH_NO_B4", "bool", False, "no NT = 2..4 GEMVs"),
    _opt("no_bf16_gemv", "advanced", "env", "OMPH_NO_BF16_GEMV", "bool", False, "BF16 weights through the f16 path"),
    _opt("no_f16_cache", "advanced", "env", "OMPH_NO_F16_CACHE", "bool", False, "re-convert f16 weights on every call"),
    _opt("attn_scalar", "advanced", "env", "OMPH_ATTN_SCALAR", "bool", False, "prefill attention on the scalar kernel"),
    _opt("attn_dec_scalar", "advanced", "env", "OMPH_ATTN_DEC_SCALAR", "bool", False, "decode attention on the scalar kernel"),
    _opt("gdn_exact", "advanced", "env", "OMPH_GDN_EXACT", "bool", False, "the delta rule per token"),
    _opt("gdn_serial", "advanced", "env", "OMPH_GDN_SERIAL", "bool", False, "a multi-token delta rule in one launch"),
    _opt("spec_check", "advanced", "env", "OMPH_SPEC_CHECK", "bool", False, "check every speculative rollback (slow)"),
    _opt("phases", "advanced", "env", "OMPH_PHASES", "bool", False, "per-phase GPU totals (adds ~2.6 ms per step)"),
)

BY_KEY = {o.key: o for o in SCHEMA}
FLAGS = tuple(o for o in SCHEMA if o.kind == "flag" and o.name)
ENVS = tuple(o for o in SCHEMA if o.kind == "env")

# The model is the positional argument; everything else is a flag.
POSITIONAL = "model"


def defaults() -> dict[str, object]:
    return {o.key: o.default for o in SCHEMA}


def group_options(group: str) -> list[Option]:
    return [o for o in SCHEMA if o.group == group]


def build_command(values: dict[str, object], binary: str = "engine/build/omph-server",
                  include_defaults: bool = False) -> tuple[dict[str, str], list[str]]:
    """(environment, argv) for the values of a form.

    A flag is written when it is set and (``include_defaults`` or it differs
    from the server's default); a bool flag only when true. An environment
    switch is written only when it differs from the engine's default, since the
    engine's defaults are the engine's."""
    env: dict[str, str] = {}
    argv: list[str] = [binary]
    model = str(values.get(POSITIONAL) or "")
    if model:
        argv.append(model)
    for o in SCHEMA:
        value = values.get(o.key, o.default)
        if o.key == POSITIONAL:
            continue
        if o.kind == "env":
            if not include_defaults and o.is_default(value):
                continue
            if o.type == "bool":
                if value:
                    env[o.name] = "1"
            elif str(value) != "":
                env[o.name] = str(value)
            continue
        if o.type == "bool":
            if value:
                argv.append(o.name)
            continue
        if o.is_default(value) and not include_defaults:
            continue
        if str(value) == "":
            continue
        argv += [o.name, str(value)]
    return env, argv


def format_command(env: dict[str, str], argv: list[str], width: int = 96) -> str:
    """The shell line, wrapped for a terminal. Every word is quoted for the
    shell: a pasted `--cors *` must not glob, and a path or an alias with a
    space must survive (#347)."""
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    parts = ([prefix] if prefix else []) + [shlex.quote(a) for a in argv]
    lines, line = [], ""
    for part in parts:
        if line and len(line) + 1 + len(part) > width:
            lines.append(line)
            line = "    " + part
        else:
            line = f"{line} {part}".strip()
    if line:
        lines.append(line)
    return " \\\n".join(lines)
