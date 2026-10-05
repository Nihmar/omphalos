"""omph-tui driven headlessly: the widgets, the mouse, the phone layout, and the
process rules (#304)."""

from __future__ import annotations

import asyncio
import os
import queue
import time
from pathlib import Path

from textual.widgets import Button, DataTable, TabbedContent

import omph_tui
from tui import log, profiles, schema
from tui import server as srv

READY = "omph-server: m.omph on http://127.0.0.1:7070 (context 131072)"
REQUEST = ("POST /v1/chat/completions: sampling temp 0.60 top-k 20 top-p 0.95 min-p 0.00 repeat 1.10 "
           "freq 0.00 pres 0.00 last-n 1024 seed 20261004; prompt 98255 tokens (0 cached) in 147138 ms "
           "(667.8 t/s); 4096 tokens in 104232 ms (39.3 t/s, drafts accepted 2762 / 10939); stop: length")
PROGRESS = "  2048 tokens, 39.1 t/s (last 3 s: 40.7 t/s), drafts accepted 71 %"
PROGRESS2 = "  3072 tokens, 40.9 t/s (last 3 s: 41.0 t/s), drafts accepted 72 %"
PREFILL = "  prefill 12288 / 98255 tokens, 820.4 t/s"
REQUEST2 = ("POST /v1/chat/completions: sampling temp 0.60 top-k 20 top-p 0.95 min-p 0.00 seed 1; "
            "prompt 45206 tokens (45193 cached, restored) in 62 ms (209.7 t/s); 889 tokens in 21268 ms "
            "(41.8 t/s, drafts accepted 612 / 1611); stop: end of generation")


def fixture(tmp_path: Path) -> Path:
    path = tmp_path / "server.log"
    path.write_text("".join(f"{line}\n" for line in (READY, PREFILL, PROGRESS, PROGRESS2, REQUEST, REQUEST2)))
    return path


def make_app(tmp_path: Path, **kw) -> omph_tui.OmphTui:
    values = {**schema.defaults(), "model": "models/x.omph", "port": 7070}
    return omph_tui.OmphTui(values, binary="engine/build/omph-server",
                            demo_file=str(tmp_path / "server.log"), demo_delay=0.0, poll=0.02, **kw)


async def click_at(pilot, app, widget) -> None:
    """Click a widget and fail loudly when the click misses: the pilot returns
    False when it lands on a child (a button's label, say), which would make a
    silently broken UI look tested."""
    x = widget.region.x + widget.region.width // 2
    y = widget.region.y + widget.region.height // 2
    under = app.get_widget_at(x, y)[0]
    assert under is widget or widget in under.ancestors, f"{(x, y)} lands on {under!r}, not {widget!r}"
    assert await pilot.click(offset=(x, y)), f"the click at {(x, y)} did not land on {widget!r}"


def test_demo_fills_the_table_and_the_keys_work(tmp_path) -> None:
    fixture(tmp_path)

    async def run() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(140, 46)) as pilot:
            await pilot.pause(0.4)
            assert app.table.row_count == 2, "the two replayed requests became two rows"
            assert [r["prompt_tokens"] for r in app.requests] == [98255, 45206]
            assert app.requests[1]["restored"] is True
            assert "requests: 2" in app.stats_text
            assert len(app.columns) == len(omph_tui.COLS) - 0, "wide: every column"
            await pilot.press("4")                       # digits switch the tab
            assert app.query_one(TabbedContent).active == "tab-cache"
            await pilot.press("f9")
            await pilot.pause(0.1)
            assert isinstance(app.screen, omph_tui.CommandScreen)
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert not isinstance(app.screen, omph_tui.CommandScreen)

    asyncio.run(run())


def test_the_live_row_tracks_every_progress_line(tmp_path) -> None:
    """The row for the request in flight is created by the first progress line
    (prefill or decode) and updated in place by the next ones (#304, #310)."""
    fixture(tmp_path)

    async def run() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(140, 46)) as pilot:
            await pilot.pause(0.4)
            app.requests.clear()
            app.live_row = None
            app.table.clear()
            app.update_live(log.parse_line(PREFILL))
            row = " ".join(str(c) for c in app.table.get_row_at(0))
            assert "12288/98255" in row and "prefilling" in row
            app.update_live(log.parse_line(PROGRESS))
            row = " ".join(str(c) for c in app.table.get_row_at(0))
            assert "2048" in row and "39.1" in row and "71 %" in row
            app.update_live(log.parse_line(PROGRESS2))
            row = " ".join(str(c) for c in app.table.get_row_at(0))
            assert "3072" in row and "40.9" in row, "the row shows the latest line"
            assert app.table.row_count == 1, "one request in flight, one row"
            # the summary replaces the live row instead of adding one
            app.finish_request(log.parse_line(REQUEST))
            assert app.table.row_count == 1 and app.live_row is None
            row = " ".join(str(c) for c in app.table.get_row_at(0))
            assert "98255" in row and "4096" in row and "length" in row

    asyncio.run(run())


def test_mouse_click_copies_the_command(tmp_path) -> None:
    fixture(tmp_path)

    async def run() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(140, 46)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("f9")
            await pilot.pause(0.2)
            await click_at(pilot, app, app.screen.query_one("#btn-copy", Button))
            assert "omph-server" in app.last_copied and "--port 7070" in app.last_copied

    asyncio.run(run())


def test_double_click_opens_the_detail(tmp_path) -> None:
    fixture(tmp_path)

    async def run() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(140, 46)) as pilot:
            await pilot.pause(0.4)
            table = app.table
            x = table.region.x + 8
            y = table.region.y + 2
            assert await pilot.click(offset=(x, y))          # one click highlights
            assert not isinstance(app.screen, omph_tui.DetailScreen)
            assert await pilot.double_click(offset=(x, y))   # the second selects
            await pilot.pause(0.2)
            assert isinstance(app.screen, omph_tui.DetailScreen)
            await click_at(pilot, app, app.screen.query_one("#btn-close", Button))
            await pilot.pause(0.2)
            assert not isinstance(app.screen, omph_tui.DetailScreen)

    asyncio.run(run())


def test_profiles_screen_loads_a_preset(tmp_path) -> None:
    fixture(tmp_path)

    async def run() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(140, 46)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("p")                     # the phone-friendly binding
            await pilot.pause(0.2)
            assert isinstance(app.screen, omph_tui.ProfileScreen)
            table = app.screen.query_one("#profiles", DataTable)
            rows = [str(table.coordinate_to_cell_key((i, 0)).row_key.value) for i in range(table.row_count)]
            assert "coding-agent" in rows
            table.focus()
            table.move_cursor(row=rows.index("long-context"))
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert app.values["kv_k4_layers"] == "none" and app.values["ctx"] == 131072

    asyncio.run(run())


def test_narrow_layout_for_a_phone_over_ssh(tmp_path) -> None:
    """58x30 is a phone in portrait over ssh: the layout stacks, the table keeps
    the columns that matter, every action still has a key and a button."""
    fixture(tmp_path)

    async def run() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(58, 30)) as pilot:
            await pilot.pause(0.4)
            assert app.narrow and app.screen.has_class("narrow")
            keys = [key for _, key in app.columns]
            assert keys == list(omph_tui.NARROW_COLS), "the narrow columns"
            assert app.table.row_count == 2
            assert "m.omph" not in app.chips_text, "the narrow chips drop the model name"
            await pilot.press("?")
            await pilot.pause(0.2)
            assert isinstance(app.screen, omph_tui.HelpScreen)
            await click_at(pilot, app, app.screen.query_one("#btn-close", Button))
            await pilot.pause(0.2)
            # the tabs and the table still work at this size
            await pilot.press("3")
            assert app.query_one(TabbedContent).active == "tab-drafting"
            x = app.table.region.x + 4
            y = app.table.region.y + 2
            assert await pilot.double_click(offset=(x, y))
            await pilot.pause(0.2)
            assert isinstance(app.screen, omph_tui.DetailScreen)
            await click_at(pilot, app, app.screen.query_one("#btn-close", Button))
            await pilot.pause(0.2)
            # and it becomes wide again when the terminal grows
            await pilot.resize_terminal(120, 40)
            await pilot.pause(0.3)
            assert not app.narrow and len(app.columns) == len(omph_tui.COLS)

    asyncio.run(run())


def test_quit_asks_before_killing_a_running_server(tmp_path) -> None:
    fixture(tmp_path)

    async def run() -> None:
        app = make_app(tmp_path, exec_command="sleep 30")
        async with app.run_test(size=(140, 46)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("s")                      # start the fake server
            await pilot.pause(0.5)
            assert app.server.running()
            app.action_quit()
            await pilot.pause(0.2)
            assert isinstance(app.screen, omph_tui.ConfirmQuit)
            await pilot.press("n")                      # no: the server keeps running
            await pilot.pause(0.2)
            assert app.server.running() and not isinstance(app.screen, omph_tui.ConfirmQuit)

    asyncio.run(run())


def test_stop_is_off_the_ui_thread(tmp_path) -> None:
    """#328: stop() waits up to 4 s for SIGTERM; on the UI thread that freezes
    the interface, so it runs in a worker."""
    fixture(tmp_path)

    async def run() -> None:
        app = make_app(tmp_path, exec_command="sleep 30")
        async with app.run_test(size=(140, 46)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("s")
            await pilot.pause(0.5)
            assert app.server.running()
            await pilot.press("x")
            deadline = time.monotonic() + 5
            while app.server.running() and time.monotonic() < deadline:
                await pilot.pause(0.1)
            assert not app.server.running(), "the worker killed the child"

    asyncio.run(run())


def test_d_deletes_a_profile(tmp_path, monkeypatch) -> None:
    """docs/tools.md promises `d`; it was only a button (#328)."""
    monkeypatch.setattr(profiles, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(profiles, "PROFILES_PATH", tmp_path / "profiles.json")
    fixture(tmp_path)
    profiles.save("mine", {**schema.defaults(), "ctx": 4096})
    values = {**schema.defaults(), "model": "m.omph"}

    async def run() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(140, 46)) as pilot:
            await pilot.pause(0.2)
            app.push_screen(omph_tui.ProfileScreen(values))
            await pilot.pause(0.2)
            table = app.screen.query_one("#profiles", DataTable)
            rows = [str(table.coordinate_to_cell_key((i, 0)).row_key.value) for i in range(table.row_count)]
            table.focus()
            table.move_cursor(row=rows.index("mine"))
            await pilot.press("d")
            await pilot.pause(0.2)

    asyncio.run(run())
    assert "mine" not in profiles.load()


def test_process_lines_and_the_group_kill(tmp_path) -> None:
    proc = srv.ServerProcess()
    proc.start(["/bin/sh", "-c", "echo hello; sleep 30"])
    deadline = time.monotonic() + 5
    saw_line = False
    while time.monotonic() < deadline and not saw_line:
        try:
            kind, payload = proc.queue.get(timeout=0.2)
        except queue.Empty:
            continue
        saw_line = kind == "line" and payload == "hello"
    assert saw_line, "the reader thread delivers the child's stdout"
    pid = proc.proc.pid if proc.proc else 0
    proc.stop()
    assert not proc.running()
    with __import__("pytest").raises(ProcessLookupError):
        os.kill(pid, 0)


def test_preflight_catches_the_wrong_model_file(tmp_path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_text("x")
    checks = {label: (level, detail) for level, label, detail in
              srv.preflight({**schema.defaults(), "model": str(gguf), "port": 1})}
    assert checks["model"][0] == "err" and "omph-convert" in checks["model"][1]
    missing = srv.preflight({**schema.defaults(), "model": str(tmp_path / "nope.omph"), "port": 1})
    assert any(level == "err" and label == "model" for level, label, _ in missing)
    est = srv.vram_estimate({**schema.defaults(), "ctx": 131072})
    assert 14.0 < est < 15.6, f"11.3 GiB of weights + 3.3 GiB of KV at 128k, got {est:.2f}"


def test_the_dflash_default_is_the_repos_when_it_exists(tmp_path) -> None:
    """The form pre-fills the model path; it should pre-fill the DFlash2 drafter
    the same way, but only when the file is there (it is optional)."""
    missing = tmp_path / "nope.omph"
    assert omph_tui.default_dflash(missing) == ""
    present = tmp_path / "d.omph"
    present.write_text("x")
    assert omph_tui.default_dflash(present) == str(present)
    # precedence: --dflash, then the profile's drafter, then the repo's
    assert omph_tui.resolve_dflash("/cli.omph", "/profile.omph", "/repo.omph") == "/cli.omph"
    assert omph_tui.resolve_dflash(None, "/profile.omph", "/repo.omph") == "/profile.omph"
    assert omph_tui.resolve_dflash(None, "", "/repo.omph") == "/repo.omph"
    assert omph_tui.resolve_dflash("", "/profile.omph", "/repo.omph") == "", "--dflash '' turns it off"


def test_preflight_does_not_call_our_own_child_another_server(tmp_path, monkeypatch) -> None:
    """#328: the F9 preflight must not warn about the app's own server."""
    monkeypatch.setattr(srv, "other_servers", lambda: [123, 456])
    values = {**schema.defaults(), "model": str(tmp_path / "m.omph"), "port": 1}
    (tmp_path / "m.omph").write_text("x")
    checks = {label: (level, detail) for level, label, detail in srv.preflight(values, {123, 456})}
    assert checks["servers"][0] == "ok"
    checks = {label: (level, detail) for level, label, detail in srv.preflight(values, {123})}
    assert checks["servers"][0] == "warn" and "456" in checks["servers"][1]


def test_copy_json_of_a_request(tmp_path) -> None:
    """#328: the raw self-reference made json.dumps raise Circular reference."""
    import json
    fixture(tmp_path)

    async def run() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(140, 46)) as pilot:
            await pilot.pause(0.4)
            app.open_details(0)
            await pilot.pause(0.2)
            assert isinstance(app.screen, omph_tui.DetailScreen)
            app.screen.action_copy_json()
            payload = json.loads(app.last_copied)
            assert payload["prompt_tokens"] == 98255 and payload["completion_tokens"] == 4096

    asyncio.run(run())


def test_export_keeps_every_number(tmp_path) -> None:
    """#328: the fieldnames were the display names, so every numeric column
    (t/s, drafts, out) came out empty."""
    import csv
    fixture(tmp_path)
    out = tmp_path / "requests.csv"

    async def run() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(140, 46)) as pilot:
            await pilot.pause(0.4)
            app.action_export(str(out))

    asyncio.run(run())
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2
    first = rows[0]
    assert first["prompt_tokens"] == "98255" and first["completion_tokens"] == "4096"
    assert float(first["prefill_tps"]) > 600 and float(first["decode_tps"]) > 30
    assert first["drafts"] == "2762/10939" and first["stop"] == "length"
    assert rows[1]["restored"] == "True"


def test_the_child_does_not_inherit_omph_switches(tmp_path, monkeypatch) -> None:
    """#328: a stale ablation in the shell must not reach the server behind the
    form's back."""
    monkeypatch.setenv("OMPH_SKIP_ATTN", "1")
    monkeypatch.setenv("OMPH_KV_F32", "1")
    proc = srv.ServerProcess()
    proc.start(["/bin/sh", "-c", "env | grep '^OMPH_' || echo none"])
    lines = []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            kind, payload = proc.queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if kind == "exit":
            break
        if kind == "line":
            lines.append(payload)
    assert lines and lines[-1] == "none", f"the shell's OMPH_* leaked in: {lines}"


def test_save_does_not_clobber_a_preset(tmp_path, monkeypatch) -> None:
    """#328: `s` on a selected preset used to overwrite the recipe."""
    monkeypatch.setattr(profiles, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(profiles, "PROFILES_PATH", tmp_path / "profiles.json")
    values = {**schema.defaults(), "model": "m.omph", "ctx": 8192}
    fixture(tmp_path)

    async def run() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(140, 46)) as pilot:
            await pilot.pause(0.2)
            app.push_screen(omph_tui.ProfileScreen(values))
            await pilot.pause(0.2)
            table = app.screen.query_one("#profiles", DataTable)
            rows = [str(table.coordinate_to_cell_key((i, 0)).row_key.value) for i in range(table.row_count)]
            table.focus()
            table.move_cursor(row=rows.index("coding-agent"))
            await pilot.press("s")  # save while a preset is selected
            await pilot.pause(0.2)

    asyncio.run(run())
    raw = __import__("json").loads(profiles.PROFILES_PATH.read_text())
    assert "coding-agent" not in raw, "the preset was written into the user's file"
    assert raw["my-profile"]["ctx"] == 8192
    assert profiles.load()["coding-agent"] == profiles.PRESETS["coding-agent"], "the preset changed"
