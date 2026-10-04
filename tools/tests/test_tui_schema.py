"""The option table, the command it builds, and its drifts (#304)."""

from __future__ import annotations

import re
from pathlib import Path

from tui import profiles, schema

SERVER_CC = Path(__file__).resolve().parents[2] / "engine/src/tools/server.cc"
OPTIONS_HH = Path(__file__).resolve().parents[2] / "engine/src/runtime/options.hh"


def test_every_flag_exists_in_the_server() -> None:
    """The schema cannot invent a flag the server does not parse."""
    source = SERVER_CC.read_text()
    parsed = set(re.findall(r'strcmp\(argv\[i\], "(--[a-z0-9-]+)"\)', source))
    for o in schema.FLAGS:
        assert o.name in parsed, f"{o.name} is in the TUI schema but not in server.cc"


def test_every_server_flag_is_in_the_schema() -> None:
    """A new server flag has to be added to the TUI (or consciously ignored)."""
    source = SERVER_CC.read_text()
    parsed = set(re.findall(r'strcmp\(argv\[i\], "(--[a-z0-9-]+)"\)', source))
    known = {o.name for o in schema.FLAGS}
    assert parsed - known == set(), f"server flags missing from the TUI: {sorted(parsed - known)}"


def test_every_env_exists_in_the_engine() -> None:
    source = OPTIONS_HH.read_text()
    for o in schema.ENVS:
        assert o.name in source, f"{o.name} is in the TUI schema but not in runtime/options.hh"


def test_command_of_the_defaults_is_minimal() -> None:
    env, argv = schema.build_command({**schema.defaults(), "model": "m.omph"})
    assert argv == ["engine/build/omph-server", "m.omph"]
    assert not env


def test_command_of_the_coding_agent_recipe() -> None:
    values = {**profiles.PRESETS["coding-agent"], "model": "m.omph"}
    _, argv = schema.build_command(values)
    assert argv[argv.index("--temp") + 1] == "0.6"
    assert argv[argv.index("--repeat-penalty") + 1] == "1.1"
    assert argv[argv.index("--repeat-last-n") + 1] == "1024"
    assert argv[argv.index("--ctx") + 1] == "131072"
    assert "OMPH_KV_K4_LAYERS" in schema.build_command(profiles.PRESETS["long-context"])[0]
    assert schema.build_command(profiles.PRESETS["long-context"])[0]["OMPH_KV_K4_LAYERS"] == "none"


def test_include_defaults_shows_everything() -> None:
    env, argv = schema.build_command({**schema.defaults(), "model": "m.omph"}, include_defaults=True)
    assert "--ctx" in argv and "--top-k" in argv
    assert "OMPH_MTP_WINDOW" in env


def test_bools_and_empty_strings() -> None:
    values = {**schema.defaults(), "model": "m.omph", "no_mtp": True, "dflash": "", "alias": ""}
    env, argv = schema.build_command(values)
    assert "--no-mtp" in argv
    assert "--alias" not in argv and "--dflash" not in argv
    env, argv = schema.build_command({**values, "ngram": 0})
    assert "OMPH_NGRAM" in env and env["OMPH_NGRAM"] == "0"


def test_profiles_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(profiles, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(profiles, "PROFILES_PATH", tmp_path / "profiles.json")
    profiles.save("mine", {**schema.defaults(), "ctx": 4096, "temperature": 0.7, "nope": 1})
    loaded = profiles.load()
    assert loaded["mine"]["ctx"] == 4096 and loaded["mine"]["temperature"] == 0.7
    assert "nope" not in loaded["mine"]
    assert profiles.delete("mine") and "mine" not in profiles.load()
