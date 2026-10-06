"""The option table, the command it builds, and its drifts (#304)."""

from __future__ import annotations

import re
from pathlib import Path

from tui import profiles, schema

SERVER_CC = Path(__file__).resolve().parents[2] / "engine/src/tools/server.cc"
OPTIONS_CC = Path(__file__).resolve().parents[2] / "engine/src/runtime/options.cc"
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


def test_every_env_is_classified_like_the_engine() -> None:
    """A switch the engine reads with flag() (set = on, the value is ignored)
    must be a bool in the TUI, and one read with getenv() must not be: typing
    OMPH_KV_F32=0 and calling it "off" would turn it on (#328)."""
    source = OPTIONS_CC.read_text()
    flag_envs = set(re.findall(r'flag\("(OMPH_[A-Z0-9_]+)"\)', source))
    value_envs = set(re.findall(r'getenv\("(OMPH_[A-Z0-9_]+)"\)', source))
    assert flag_envs and value_envs, "the options parser changed shape; fix this test"
    for o in schema.ENVS:
        assert o.name in flag_envs or o.name in value_envs, f"{o.name} is not read in options.cc"
        assert not (o.name in flag_envs and o.name in value_envs), f"{o.name} is both a flag and a value"
        assert (o.type == "bool") == (o.name in flag_envs), \
            f"{o.name}: engine flag()={o.name in flag_envs}, schema type={o.type}"


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


def test_env_defaults_match_the_engine() -> None:
    """#347: the form showed OMPH_KV_WINDOW 128 while the engine had run 512
    since d9ad4e0, and choosing 128 was impossible. Every env in the schema is
    checked against its field's initializer in options.hh."""
    cc, hh = OPTIONS_CC.read_text(), OPTIONS_HH.read_text()
    initializers = {m.group(1): m.group(2).strip() for m in re.finditer(
        r"(?:(?:int64_t|uint64_t|int|float|bool)\s+)(\w+)\s*=\s*([^;]+);", hh)}
    assert "kv_window" in initializers, "options.hh changed shape; fix this test"
    for o in schema.ENVS:
        m = re.search(r'o\.(\w+)\s*=\s*flag\("' + o.name + r'"\)', cc)
        if m is None:
            # a value switch: the assignment sits in the `if (getenv(...))`
            # block, possibly nested one level deeper (OMPH_GEMM_MIN)
            found = re.search(r'getenv\("' + o.name + r'"\)', cc)
            assert found, f"{o.name} is not read in options.cc"
            m = re.search(r"\bo\.(\w+)\s*=", cc[found.end():found.end() + 300])
        assert m, f"{o.name} is not read in options.cc"
        field = m.group(1)
        assert field in initializers, f"{field} has no initializer in options.hh"
        if o.name == "OMPH_KV_K4_LAYERS":
            continue  # the C++ default is a bitmask constant, the schema a list
        raw = initializers[field]
        if o.type == "bool":
            assert raw in ("false", "true"), f"{o.name}: options.hh says {raw!r}"
            want: object = raw == "true"
        elif o.type == "int":
            want = int(raw)
        else:
            want = float(raw.removesuffix("f"))
        assert o.default == want, f"{o.name}: the TUI shows {o.default!r}, the engine defaults to {want!r}"


def test_the_command_quotes_its_words() -> None:
    """#347: a pasted `--cors *` glob-expanded, a path with a space broke."""
    env, argv = schema.build_command({**schema.defaults(), "model": "/tmp/my models/m.omph",
                                      "cors": "*", "alias": "my alias"})
    line = schema.format_command(env, argv)
    assert "'/tmp/my models/m.omph'" in line
    assert "'*'" in line and "'my alias'" in line
    import shlex
    assert shlex.split(line.replace("\\\n", " ")) == argv


def test_include_defaults_shows_everything() -> None:
    env, argv = schema.build_command({**schema.defaults(), "model": "m.omph"}, include_defaults=True)
    assert "--ctx" in argv and "--top-k" in argv
    assert "OMPH_MTP_WINDOW" in env


def test_include_defaults_never_writes_a_false_flag_env() -> None:
    """#328: OMPH_OVERLAP=0 enables OMPH_OVERLAP (flag()), so a flag env must
    only ever appear as "1"."""
    source = OPTIONS_CC.read_text()
    flag_envs = set(re.findall(r'flag\("(OMPH_[A-Z0-9_]+)"\)', source))
    env, _ = schema.build_command({**schema.defaults(), "model": "m.omph"}, include_defaults=True)
    for name in flag_envs:
        assert name not in env, f"{name} is a flag env but the default command writes {name}={env[name]!r}"


def test_a_flag_env_is_written_as_one() -> None:
    env, _ = schema.build_command({**schema.defaults(), "model": "m.omph",
                                   "kv_f32": True, "overlap": True})
    assert env["OMPH_KV_F32"] == "1" and env["OMPH_OVERLAP"] == "1"


def test_bools_and_empty_strings() -> None:
    values = {**schema.defaults(), "model": "m.omph", "no_mtp": True, "dflash": "", "alias": ""}
    env, argv = schema.build_command(values)
    assert "--no-mtp" in argv
    assert "--alias" not in argv and "--dflash" not in argv
    env, argv = schema.build_command({**values, "ngram": 0})
    assert "OMPH_NGRAM" in env and env["OMPH_NGRAM"] == "0"


def test_the_tools_entries_build_the_command() -> None:
    """#380: the tools tab reaches the server as --tools/--agent."""
    _, argv = schema.build_command({**schema.defaults(), "model": "m.omph", "tools": "read_file,grep_search"})
    assert argv[argv.index("--tools") + 1] == "read_file,grep_search"
    _, argv = schema.build_command({**schema.defaults(), "model": "m.omph", "agent": True})
    assert "--agent" in argv and "--tools" not in argv
    _, argv = schema.build_command({**schema.defaults(), "model": "m.omph"})
    assert "--tools" not in argv and "--agent" not in argv


def test_a_dflash_path_is_passed() -> None:
    """The TUI pre-fills the DFlash2 drafter (#332); the field has to reach the
    server as --dflash."""
    _, argv = schema.build_command({**schema.defaults(), "model": "m.omph", "dflash": "/m/d.omph"})
    assert argv[argv.index("--dflash") + 1] == "/m/d.omph"


def test_profiles_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(profiles, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(profiles, "PROFILES_PATH", tmp_path / "profiles.json")
    profiles.save("mine", {**schema.defaults(), "ctx": 4096, "temperature": 0.7, "nope": 1})
    loaded = profiles.load()
    assert loaded["mine"]["ctx"] == 4096 and loaded["mine"]["temperature"] == 0.7
    assert "nope" not in loaded["mine"]
    assert profiles.delete("mine") and "mine" not in profiles.load()
