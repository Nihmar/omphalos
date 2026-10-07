#!/usr/bin/env python3
"""omph-tui — a terminal UI around omph-server (#304).

Pick the options in the form, start the server, watch what it does: the log as
the server writes it, one row per request with the prompt size, the cache hits,
the prefill and decode speed and the drafts accepted, and a running total in
the status bar. Keyboard and mouse are both first class: every action has a
binding, every visible target is clickable (tabs, rows, selectors, checkboxes,
the modal buttons, the scrollbars).

    uv run python omph_tui.py [--exec CMD | --demo FILE] [--model FILE] [--dflash FILE]
                              [--mmproj FILE] [--profile NAME] [--poll SECONDS]
                              [--screenshot FILE.svg]

The model, the DFlash2 drafter and the vision encoder default to the
repository's own `models/` files (`--dflash ""` starts on the MTP head instead,
`--mmproj ""` without vision).

`--exec` runs any command instead of omph-server and `--demo` replays a saved
log: both are how the UI is developed and tested without a GPU (the pytest in
tools/tests/test_tui.py drives the same paths through Textual's Pilot).
"""

from __future__ import annotations

import argparse
import csv
import queue
import shlex
import signal
import threading
import time
import webbrowser
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from rich.markup import escape
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Input,
    RichLog,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)
from textual.widgets.data_table import CellDoesNotExist
from textual.widgets.select import InvalidSelectValueError

import tui.log as logmod
import tui.server as srv
from tui import profiles, schema

HERE = Path(__file__).resolve().parent
# The repo's own files, for the form's defaults: the .omph omph-convert writes,
# the DFlash2 drafter and the vision encoder next to it (README's "everything
# on" / coding-agent recipes; the encoder runs on the CPU, no VRAM, #385). Both
# are optional, so a checkout without one leaves its field empty rather than
# failing the preflight.
DEFAULT_MODEL = HERE.parent / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.omph"
DEFAULT_DFLASH = HERE.parent / "models/Qwen3.8-27B-DFlash2-Q4_K_M.omph"
DEFAULT_MMPROJ = HERE.parent / "models/mmproj-Qwen3.8-27B-BF16.gguf"


def default_file(path: Path) -> str:
    """An optional file (drafter, encoder) to pre-fill the form with, or ""
    when it is not there (the model field is always filled; these only when
    they can run)."""
    return str(path) if path.is_file() else ""


def resolve_file(cli: str | None, profile_value: object, repo_default: str) -> str:
    """`--dflash` / `--mmproj` wins, then the profile's own file, then the
    repo's (the profile's empty field means "the default", `--dflash ""` off)."""
    if cli is not None:
        return cli
    return str(profile_value or repo_default)


COLS = [("#", "num"), ("time", "time"), ("endpoint", "path"), ("sampling", "sampling"), ("prompt", "prompt"),
        ("cached", "cached"), ("prefill", "prefill"), ("t/s", "prefill_tps"), ("out", "out"),
        ("t/s", "decode_tps"), ("drafts", "drafts"), ("acc", "acc"), ("stop", "stop")]
# A 480-wide phone in portrait gives ~50 columns over ssh: keep what answers
# "did it use the cache, how fast, did it stop".
NARROW_COLS = ("num", "time", "prompt", "cached", "out", "decode_tps", "stop")
NARROW_WIDTH = 100


class FormPane(VerticalScroll):
    """One tab of the form: a row per option of a schema group."""

    def __init__(self, group: str) -> None:
        super().__init__(id=f"form-{group}")
        self.group = group
        self.widgets: dict[str, object] = {}

    def compose(self) -> ComposeResult:
        for o in schema.group_options(self.group):
            if o.key == schema.POSITIONAL:
                widget = Input(placeholder="/path/to/model.omph", id=f"in-{o.key}", value=str(o.default))
            elif o.type == "bool":
                widget = Checkbox(value=bool(o.default), id=f"in-{o.key}")
            elif o.type == "enum":
                widget = Select([(c, c) for c in o.choices], value=str(o.default), allow_blank=False,
                                id=f"in-{o.key}")
            elif o.type in ("int", "float"):
                widget = Input(value=str(o.default), type="integer" if o.type == "int" else "number",
                               id=f"in-{o.key}")
            else:
                widget = Input(value=str(o.default), id=f"in-{o.key}")
            self.widgets[o.key] = widget
            label = o.name or "model"
            hint = f"  [dim]{escape(o.help)}[/dim]"
            if o.note:
                hint += f" [dim italic]({escape(o.note)})[/dim italic]"
            yield Horizontal(Static(f"[b]{label}[/b]{hint}", classes="field-label"), widget, classes="field")

    def values(self) -> dict:
        out: dict = {}
        for key, widget in self.widgets.items():
            o = schema.BY_KEY[key]
            if isinstance(widget, Checkbox):
                out[key] = widget.value
            elif isinstance(widget, Select):
                out[key] = widget.value if widget.value is not Select.BLANK else o.default
            else:
                raw = widget.value.strip()
                if o.type in ("int", "float"):
                    try:
                        out[key] = int(raw) if o.type == "int" else float(raw)
                    except ValueError:
                        out[key] = o.default
                else:
                    out[key] = raw
        return out

    def set_values(self, values: dict) -> None:
        for key, widget in self.widgets.items():
            if key not in values:
                continue
            value = values[key]
            if isinstance(widget, Checkbox):
                widget.value = bool(value)
            elif isinstance(widget, Select):
                try:
                    widget.value = value
                except InvalidSelectValueError:
                    pass  # a profile value this build does not offer: keep the current one
            else:
                widget.value = str(value)


class DetailScreen(ModalScreen):
    """One request's numbers as fields, opened by a double click (or `enter`)."""

    BINDINGS: ClassVar = [Binding("escape", "dismiss", "close"), Binding("c", "copy_json", "copy json")]

    def __init__(self, row: dict, index: int) -> None:
        super().__init__()
        self.row = row
        self.index = index

    def compose(self) -> ComposeResult:
        f = self.row
        lines = [
            ("endpoint", f.get("path", "")),
            ("sampling", f.get("sampling", "")),
            ("prompt", (f"{f.get('prompt_tokens', 0)} tokens ({f.get('cached_tokens', 0)} cached"
                        f"{', restored' if f.get('restored') else ''})")),
            ("prefill", f"{f.get('prefill_ms', 0):.0f} ms ({f.get('prefill_tps') or 0:.1f} t/s)"),
            ("generated", (f"{f.get('completion_tokens', 0)} tokens in {f.get('decode_ms', 0):.0f} ms "
                           f"({f.get('decode_tps') or 0:.1f} t/s)")),
            ("drafts", f"{f.get('accepted', 0)} accepted / {f.get('drafted', 0)} proposed"
                       + (f" ({100.0 * f['accepted'] / f['drafted']:.0f} %)" if f.get("drafted") else "")),
            ("stop", str(f.get("stop", "")) + (" (client gone)" if f.get("client_gone") else "")),
        ]
        with Vertical(id="modal"):
                yield Static(f"request #{self.index} · {f.get('time', '')}", classes="modal-title")
                with VerticalScroll(id="modal-body"):
                    for label, value in lines:
                        yield Horizontal(Static(label, classes="detail-label"), Static(str(value)), classes="field")
                yield Horizontal(
                    Button("copy JSON", id="btn-copy", variant="primary"),
                    Button("close", id="btn-close"),
                    id="modal-buttons")

    @on(Button.Pressed, "#btn-close")
    def _close(self) -> None:
        self.dismiss()

    @on(Button.Pressed, "#btn-copy")
    def _copy(self) -> None:
        self.action_copy_json()

    def action_copy_json(self) -> None:
        import json
        payload = {k: v for k, v in self.row.items() if k != "raw"}
        text = json.dumps(payload, default=str)
        self.app.copy_to_clipboard(text)
        self.app.last_copied = text  # for the tests
        self.app.notify("request copied as JSON")


class ProfileScreen(ModalScreen):
    """F7: load, save or delete a profile. Click a row (or enter) to load."""

    BINDINGS: ClassVar = [Binding("escape", "dismiss", "cancel"), Binding("s", "save", "save"),
                          Binding("d", "delete", "delete")]

    def __init__(self, current: dict) -> None:
        super().__init__()
        self.current = current

    def compose(self) -> ComposeResult:
        self.table = DataTable(id="profiles", cursor_type="row", zebra_stripes=True)
        self.table.add_columns("profile", "what it is")
        for name, values in sorted(profiles.load().items()):
            self.table.add_row(name, self._describe(values), key=name)
        with Vertical(id="modal"):
                yield Static("profiles — click a row to load, `s` saves the current form", classes="modal-title")
                with VerticalScroll():
                    yield self.table
                yield Horizontal(Button("load", id="btn-load", variant="primary"), Button("save current", id="btn-save"),
                                 Button("delete", id="btn-delete", variant="error"), Button("cancel", id="btn-cancel"),
                                 id="modal-buttons")

    @staticmethod
    def _describe(values: dict) -> str:
        temp = float(values.get("temperature") or 0)
        sampler = "greedy" if temp <= 0 else f"{temp:g}/{values.get('top_p')}/{values.get('top_k')}"
        parts = [sampler, f"ctx {values.get('ctx')}"]
        if float(values.get("repeat_penalty") or 1) != 1:
            parts.append(f"repeat {float(values['repeat_penalty']):g} n={values.get('repeat_last_n')}")
        if values.get("dflash"):
            parts.append("DFlash2")
        if values.get("kv_k4_layers") == "none":
            parts.append("K8")
        return " · ".join(str(p) for p in parts)

    @on(DataTable.RowSelected, "#profiles")
    def _selected(self, event: DataTable.RowSelected) -> None:
        name = str(event.row_key.value)
        values = profiles.load().get(name)
        if values:
            self.dismiss(values)
            self.app.notify(f"profile loaded: {name}")

    @on(Button.Pressed)
    def _buttons(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss()
        elif event.button.id == "btn-save":
            self.action_save()
        elif event.button.id == "btn-delete":
            self.action_delete()
        elif event.button.id == "btn-load":
            name = self._cursor_name()
            if name:
                self.dismiss(profiles.load().get(name))

    def _cursor_name(self) -> str | None:
        if self.table.cursor_row < 0 or self.table.cursor_row >= len(self.table.rows):
            return None
        return str(self.table.coordinate_to_cell_key((self.table.cursor_row, 0)).row_key.value)

    def action_save(self) -> None:
        name = self._cursor_name() or "my-profile"
        if name in profiles.PRESETS:
            name = "my-profile"  # a preset is a recipe to start from, not to overwrite
        if name in profiles.load() and name not in profiles.PRESETS:
            # a saved profile would be replaced: ask first (#347)
            def answer(overwrite) -> None:
                if overwrite:
                    self._write(name)
            self.app.push_screen(ConfirmOverwrite(name), answer)
            return
        self._write(name)

    def _write(self, name: str) -> None:
        path = profiles.save(name, self.current)
        if path is None:
            # never overwrite a file this build cannot parse (#347)
            self.app.notify(f"cannot save: {profiles.PROFILES_PATH} is not readable JSON", severity="error")
        else:
            self.app.notify(f"saved {name} to {path}")

    def action_delete(self) -> None:
        """`d` (the docs say so) or the delete button."""
        name = self._cursor_name()
        if name and profiles.delete(name):
            self.app.notify(f"deleted {name}")
            self.dismiss("refresh")


class HelpScreen(ModalScreen):
    """`?` (or `h`): what every key does -- the soft keyboards of a phone have
    no F-keys and no visible footer to read."""

    BINDINGS: ClassVar = [Binding("escape", "dismiss", "close"), Binding("question_mark", "dismiss", "close"),
                          Binding("h", "dismiss", "close")]

    KEYS = (("s / F2", "start the server with the form's options"),
            ("x / F3", "stop it (and its process group)"),
            ("r / F5", "restart"),
            ("p / F7", "profiles: load, save, delete"),
            ("c / F9", "the command that will run, with the pre-flight checks"),
            ("y / F10", "copy that command"),
            ("t", "send a small test prompt to the running server"),
            ("w", "open the server's web UI in the browser"),
            ("e", "the log: everything / errors only"),
            ("enter", "details for the selected request (or double click)"),
            ("1..7", "the form's tabs"),
            ("ctrl+l", "clear the log"),
            ("q", "quit (asks first if the server is running)"))

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Static("keys", classes="modal-title")
            with VerticalScroll(id="modal-body"):
                for key, what in self.KEYS:
                    yield Horizontal(Static(key, classes="detail-label"), Static(what), classes="field")
                yield Static("[dim]mouse: tabs, rows, buttons, selectors and the scrollbars all "
                             "take clicks; the wheel scrolls. Shift+drag selects text.[/dim]")
            yield Horizontal(Button("close", id="btn-close", variant="primary"), id="modal-buttons")

    @on(Button.Pressed, "#btn-close")
    def _close(self) -> None:
        self.dismiss()


class ConfirmQuit(ModalScreen):
    """Quitting kills the server (no orphans); ask when one is running."""

    BINDINGS: ClassVar = [Binding("escape", "dismiss", "cancel"), Binding("n", "dismiss", "no")]

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Static("the server is running", classes="modal-title")
            with VerticalScroll(id="modal-body"):
                yield Static("Quitting stops it (the TUI kills what it started).\n\nQuit?")
            yield Horizontal(Button("quit", id="btn-quit", variant="error"), Button("cancel", id="btn-cancel"),
                             id="modal-buttons")

    @on(Button.Pressed)
    def _buttons(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-quit")


class ConfirmOverwrite(ModalScreen):
    """`s` on a saved profile replaces it: ask first (#347). The presets are
    never overwritten, only the user's own profiles."""

    BINDINGS: ClassVar = [Binding("escape", "dismiss", "cancel"), Binding("n", "dismiss", "no")]

    def __init__(self, name: str) -> None:
        super().__init__()
        self.profile = name

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Static(f"the profile {self.profile} exists", classes="modal-title")
            with VerticalScroll(id="modal-body"):
                yield Static(f"Overwrite {self.profile} with the form's values?")
            yield Horizontal(Button("overwrite", id="btn-overwrite", variant="error"),
                             Button("cancel", id="btn-cancel"), id="modal-buttons")

    @on(Button.Pressed)
    def _buttons(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-overwrite")


class CommandScreen(ModalScreen):
    """F9: exactly the argv F2 will exec, the env line, and the pre-flight checks."""

    BINDINGS: ClassVar = [Binding("escape", "dismiss", "close"), Binding("c", "copy", "copy")]

    def __init__(self, values: dict, binary: str, exec_command: str | None,
                 exclude_pids: Iterable[int] = ()) -> None:
        super().__init__()
        self.values = values
        self.binary = binary
        self.exec_command = exec_command
        # the preflight must not call the app's own child "another omph-server"
        self.exclude_pids = set(exclude_pids)
        self.show_defaults = False

    def command(self) -> tuple[dict, list]:
        if self.exec_command:
            return {}, shlex.split(self.exec_command)
        return schema.build_command(self.values, self.binary, include_defaults=self.show_defaults)

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
                yield Static("command", classes="modal-title")
                with VerticalScroll(id="modal-body"):
                    yield Static(id="command-text")
                    yield Horizontal(Switch(value=False, id="sw-defaults"), Static("show defaults"), classes="field")
                    yield Static(id="preflight")
                yield Horizontal(Button("copy", id="btn-copy", variant="primary"), Button("start", id="btn-start"),
                                 Button("close", id="btn-close"), id="modal-buttons")

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        env, argv = self.command()
        text = schema.format_command(env, argv)
        self.query_one("#command-text", Static).update(escape(text))
        lines = []
        for level, label, detail in srv.preflight(self.values, self.exclude_pids):
            mark = {"ok": "[green]ok[/green]", "warn": "[yellow]warn[/yellow]", "err": "[red]err[/red]"}[level]
            lines.append(f"{mark} {label}: {detail}")
        self.query_one("#preflight", Static).update("\n".join(lines))

    @on(Switch.Changed, "#sw-defaults")
    def _toggle(self, event: Switch.Changed) -> None:
        self.show_defaults = event.value
        self._refresh()

    @on(Button.Pressed)
    def _buttons(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-copy":
            self.action_copy()
        elif event.button.id == "btn-start":
            self.dismiss("start")
        else:
            self.dismiss()

    def action_copy(self) -> None:
        env, argv = self.command()
        text = schema.format_command(env, argv)
        self.app.copy_to_clipboard(text)
        self.app.last_copied = text
        self.app.notify("command copied (env line included)")


class OmphTui(App):
    """The application."""

    CSS: ClassVar[str] = """
    Screen { background: $background; }
    #titlebar { height: 1; background: $boost; padding: 0 1; }
    #titlebar Static { padding: 0 1; }
    #chips { width: 1fr; height: 1; }
    #buttons { width: auto; height: 1; }
    #buttons Button { min-width: 9; height: 1; border: none; margin: 0 1; }
    TabbedContent { height: 1fr; }
    #main { height: 1fr; }
    #tabs { width: 55%; border-right: solid $panel; }
    /* Phone over ssh: stack the form above the log, trim the table, keep every
       button reachable. Toggled from on_resize (NARROW_WIDTH). */
    Screen.narrow #main { layout: vertical; }
    Screen.narrow #tabs { width: 100%; height: 55%; border-right: none; border-bottom: solid $panel; }
    Screen.narrow #log-pane { width: 100%; height: 45%; }
    Screen.narrow #table { height: 9; }
    Screen.narrow .field-label { width: 34%; }
    FormPane { padding: 0 1; }
    .field { height: auto; }
    .field-label { width: 30%; height: auto; }
    .field Input, .field Select, .field Checkbox { width: 1fr; }
    #log-pane { width: 1fr; border-left: solid $panel; }
    #log-pane Static, #preflight { padding: 0 1; }
    #log { height: 1fr; }
    #table { height: 12; border-top: solid $panel; }
    #stats { height: 1; background: $boost; padding: 0 1; }
    /* A modal has to fit the terminal it is drawn in: a title, a scrolling body
       and a row of buttons that is always on screen (a phone over ssh is 30 rows
       tall, and a button below the last row cannot be tapped). */
    CommandScreen, DetailScreen, ProfileScreen { align: center middle; }
    #modal { width: 90; max-width: 95%; height: 85%; background: $surface; border: thick $accent; padding: 0 1; }
    .modal-title { height: 1; background: $accent 30%; padding: 0 1; }
    #modal-body { height: 1fr; }
    .detail-label { width: 16; }
    #modal-buttons { height: 3; align: center middle; }
    #profiles { height: 1fr; }
    """

    BINDINGS: ClassVar = [
        # The F-keys are the comfortable path on a desktop; a phone's soft
        # keyboard has none, so every action has a letter too (and a button).
        Binding("f2", "start", "start"),
        Binding("f3", "stop", "stop"),
        Binding("f5", "restart", "restart"),
        Binding("f7", "profiles", "profiles"),
        Binding("f8", "export", "export CSV"),
        Binding("f9", "command", "command"),
        Binding("f10", "copy", "copy"),
        Binding("s", "start", "start", show=False),
        Binding("x", "stop", "stop", show=False),
        Binding("r", "restart", "restart", show=False),
        Binding("p", "profiles", "profiles", show=False),
        Binding("c", "command", "command", show=False),
        Binding("y", "copy", "copy", show=False),
        Binding("h", "help", "help", show=False),
        Binding("question_mark", "help", "help", show=False),
        Binding("t", "test_prompt", "test prompt"),
        Binding("w", "open_webui", "web ui"),
        Binding("e", "toggle_errors", "errors"),
        Binding("ctrl+l", "clear_log", "clear log"),
        Binding("enter", "details", "details"),
        Binding("1", "tab('model')", "", show=False),
        Binding("2", "tab('sampling')", "", show=False),
        Binding("3", "tab('drafting')", "", show=False),
        Binding("4", "tab('cache')", "", show=False),
        Binding("5", "tab('vision')", "", show=False),
        Binding("6", "tab('advanced')", "", show=False),
        Binding("7", "tab('tools')", "", show=False),
        Binding("q", "quit", "quit"),
    ]

    def __init__(self, values: dict | None = None, binary: str = "engine/build/omph-server",
                 exec_command: str | None = None, demo_file: str | None = None, poll: float = 0.1,
                 demo_delay: float = 0.05) -> None:
        super().__init__()
        self.values = {**schema.defaults(), **(values or {})}
        self.binary = binary
        self.exec_command = exec_command
        self.demo_file = demo_file
        self.poll = poll
        self.demo_delay = demo_delay
        self.server = srv.ServerProcess()
        import atexit
        atexit.register(self.server.stop)  # a crash or an exit() must not leave it behind (#347)
        self.requests: list[dict] = []
        self.live_row: object | None = None
        self.live_phase: str | None = None
        self.errors_only = False
        self.last_copied = ""
        self.started_at: float | None = None
        self.ready_at: float | None = None
        self.health = "unknown"
        self.stats_text = ""
        self.chips_text = ""
        self.narrow = False
        self.columns = list(COLS)
        self.column_keys: dict[str, object] = {}
        self.demo_thread: threading.Thread | None = None

    # --- layout

    def compose(self) -> ComposeResult:
        with Horizontal(id="titlebar"):
            yield Static("omph-tui", id="app-name")
            yield Static("", id="chips")
            with Horizontal(id="buttons"):
                yield Button("start", id="btn-start", variant="success")
                yield Button("stop", id="btn-stop", variant="error")
                yield Button("restart", id="btn-restart")
        with Horizontal(id="main"):
            with TabbedContent(id="tabs"):
                for group in schema.GROUPS:
                    with TabPane(group.capitalize(), id=f"tab-{group}"):
                        yield FormPane(group)
            with Vertical(id="log-pane"):
                yield Static("log  ·  follow on", id="log-title")
                yield RichLog(id="log", markup=True, wrap=True, highlight=False)
        yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
        yield Static("", id="stats")
        yield Footer()

    def on_resize(self, event) -> None:
        self.apply_layout(event.size.width < NARROW_WIDTH)

    def apply_layout(self, narrow: bool, force: bool = False) -> None:
        changed = narrow != self.narrow
        self.narrow = narrow
        if getattr(self, "table", None) is None or not (changed or force):
            return  # a resize before the widgets exist: on_mount applies it
        self.screen.set_class(narrow, "narrow")
        self._set_columns(narrow)
        self.logview.write(f"[dim]layout: {'narrow' if narrow else 'wide'} "
                           f"({'phone/ssh' if narrow else 'desktop'})[/dim]")
        self.refresh_chips()

    def on_mount(self) -> None:
        self.forms = {g: self.query_one(f"#form-{g}", FormPane) for g in schema.GROUPS}
        for pane in self.forms.values():
            pane.set_values(self.values)
        self.logview = self.query_one("#log", RichLog)
        self.chips = self.query_one("#chips", Static)
        self.stats = self.query_one("#stats", Static)
        self.table = self.query_one("#table", DataTable)
        self._set_columns(self.narrow)
        self.apply_layout(self.size.width < NARROW_WIDTH, force=True)
        self.set_interval(self.poll, self.drain)
        self.set_interval(1.0, self.refresh_chips)
        self.refresh_chips()
        self.logview.write("[dim]ready. F2 starts the server, F9 shows the command, F7 the profiles.[/dim]")
        if self.demo_file:
            self.start_demo()

    # --- form values

    def collect(self) -> dict:
        values = dict(self.values)
        for pane in self.forms.values():
            values.update(pane.values())
        self.values = values
        return values

    def apply(self, values: dict) -> None:
        self.values = {**schema.defaults(), **values}
        for pane in self.forms.values():
            pane.set_values(self.values)
        self.refresh_chips()

    # --- process

    def command(self) -> tuple[dict, list]:
        if self.exec_command:
            return {}, shlex.split(self.exec_command)
        return schema.build_command(self.collect(), self.binary)

    def action_start(self) -> None:
        if self.server.running():
            self.notify("already running", severity="warning")
            return
        env, argv = self.command()
        self.requests.clear()
        self.live_row = None
        self.live_phase = None
        self.table.clear()
        self.ready_at = None
        try:
            self.server.start(argv, env)
        except (OSError, ValueError) as exc:
            self.notify(f"cannot start: {exc}", severity="error")
            self.logview.write(f"[red]cannot start: {exc}[/red]")
            return
        self.started_at = time.monotonic()
        self.logview.write(f"[b]$ {escape(' '.join(argv))}[/b]")
        if env:
            self.logview.write("[dim]env " + escape(" ".join(f"{k}={v}" for k, v in env.items())) + "[/dim]")

    def action_stop(self) -> None:
        if self.demo_thread and self.demo_thread.is_alive():
            self.demo_thread = None
            self.logview.write("[yellow]demo stopped[/yellow]")
            return
        if not self.server.running():
            self.notify("not running")
            return
        self._stop_then()

    def _stop_then(self, done=None) -> None:
        """SIGTERM, then up to 4 s of waiting: off the UI thread, so F3/F5 do not
        freeze the interface while the child goes away."""
        def work() -> None:
            self.server.stop()
            self.call_from_thread(self._stopped, done)
        self.run_worker(work, thread=True, group="stop")

    def _stopped(self, done) -> None:
        if not self.is_running:
            return
        self.logview.write("[yellow]stopped[/yellow]")
        if done is not None:
            done()

    def action_restart(self) -> None:
        if self.server.running():
            self._stop_then(self.action_start)
        else:
            self.action_start()

    # --- events

    def drain(self) -> None:
        while True:
            try:
                kind, payload = self.server.queue.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                self.on_line(payload)
            elif kind == "started":
                self.notify(f"started pid {payload['pid']}")
            elif kind == "exit":
                self.logview.write(f"[yellow]server exited with {payload}[/yellow]" if payload
                               else "[yellow]server exited[/yellow]")
            elif kind == "stopped":
                pass
        self.refresh_chips()

    def on_line(self, line: str) -> None:
        event = logmod.parse_line(line)
        if event.kind == "ready":
            self.ready_at = time.monotonic()
        if event.kind == "request":
            self.finish_request(event)
        elif event.kind == "progress":
            self.update_live(event)
        if event.kind in ("error", "warn") or not self.errors_only:
            style = {"error": "red", "warn": "yellow", "ready": "green", "progress": "cyan", "request": ""}.get(
                event.kind, "dim")
            text = escape(event.text)
            self.logview.write(f"[{style}]{text}[/{style}]" if style else text)

    def finish_request(self, event) -> None:
        table = self.table
        f = dict(event.fields)
        f["time"] = datetime.now(tz=UTC).astimezone().strftime("%H:%M:%S")
        self.requests.append(f)
        index = len(self.requests)
        row = self.row_values(f, index)
        if self.live_row is not None:
            # the row created by the progress lines becomes the final one: same
            # row, the summary's values (a stale row is a bug, not a warning)
            for (_, cell), value in zip(self.columns, row, strict=True):
                self._update_live_cell(cell, value)
            self.live_row = None
        else:
            table.add_row(*row, key=f"r{index}")
        self.live_phase = None

    def update_live(self, event) -> None:
        table = self.table
        f = event.fields
        tokens = f"{(f.get('tokens') or 0):d}"
        tps = f"{f.get('tps') or 0:.1f}"
        if f.get("phase") == "prefill":  # #310: a long prompt shows its progress
            where = f"{f.get('tokens')}/{f.get('total')} ({f.get('tps') or 0:.0f} t/s)"
            if self.live_row is None:
                cells = {"num": str(len(self.requests) + 1),
                         "time": datetime.now(tz=UTC).astimezone().strftime("%H:%M:%S"),
                         "path": "chat", "prompt": f"{(f.get('total') or 0):d}",
                         "prefill": where, "decode_tps": "", "stop": "[cyan]prefilling[/cyan]"}
                self.live_row = table.add_row(*[cells.get(key, "") for _, key in self.columns],
                                              key=f"r{len(self.requests) + 1}")
                self.live_phase = "prefill"
            else:
                self._update_live_cell("prefill", where)
            return
        if self.live_row is None:
            cells = {"num": str(len(self.requests) + 1),
                     "time": datetime.now(tz=UTC).astimezone().strftime("%H:%M:%S"),
                     "path": "chat", "prompt": "…", "prefill": "prefilling…",
                     "out": tokens, "decode_tps": tps,
                     "acc": f"{f.get('accepted_pct') or 0:.0f} %",
                     "stop": "[cyan]in flight[/cyan]"}
            self.live_row = table.add_row(*[cells.get(key, "") for _, key in self.columns],
                                          key=f"r{len(self.requests) + 1}")
            self.live_phase = "decode"
            return
        try:
            self._update_live_cell("out", tokens)
            self._update_live_cell("decode_tps", tps)
            if f.get("accepted_pct") is not None:
                self._update_live_cell("acc", f"{f['accepted_pct']:.0f} %")
            if self.live_phase == "prefill":  # the prefill's row now reports the decode
                self._update_live_cell("stop", "[cyan]in flight[/cyan]")
                self.live_phase = "decode"
        except CellDoesNotExist as exc:
            self.log.debug(f"live row not updated: {exc}")

    def _update_live_cell(self, cell: str, value: str) -> None:
        """A column that this width does not show is simply not there (the
        narrow layout drops half of them)."""
        key = self.column_keys.get(cell)
        if key is not None and self.live_row is not None:
            self.table.update_cell(self.live_row, key, value)

    @staticmethod
    def cell_values(f: dict, index: int | None = None) -> dict:
        """One row's cell values by column key; the visible columns pick from
        this, so trimming for a narrow screen cannot lose the data."""
        drafted, accepted = f.get("drafted") or 0, f.get("accepted") or 0
        return {
            "num": "" if index is None else str(index),
            "time": f.get("time", ""),
            "path": str(f.get("path", "")).replace("/v1/", ""),
            "sampling": logmod.sampling_summary(str(f.get("sampling", ""))),
            "prompt": f"{(f.get('prompt_tokens') or 0):d}",
            "cached": f"{(f.get('cached_tokens') or 0):d}" + (" restored" if f.get("restored") else ""),
            "prefill": f"{((f.get('prefill_ms') or 0) / 1000.0):.2f} s",
            "prefill_tps": f"{f.get('prefill_tps') or 0:.0f}",
            "out": f"{(f.get('completion_tokens') or 0):d}",
            "decode_tps": f"{f.get('decode_tps') or 0:.1f}",
            "drafts": f"{accepted}/{drafted}" if drafted else "—",
            "acc": f"{100.0 * accepted / drafted:.0f} %" if drafted else "—",
            "stop": str(f.get("stop", "")),
        }

    def row_values(self, f: dict, index: int | None = None) -> list[str]:
        cells = self.cell_values(f, index)
        return [cells[key] for _, key in self.columns]

    def _set_columns(self, narrow: bool) -> None:
        """Wide: every column. Narrow (phone over ssh): the seven that answer
        "cache hit? how fast? did it stop?"."""
        self.columns = [c for c in COLS if not narrow or c[1] in NARROW_COLS]
        table = self.table
        table.clear(columns=True)
        keys = table.add_columns(*[name for name, _ in self.columns])
        self.column_keys = {cell: key for (_, cell), key in zip(self.columns, keys, strict=True)}
        for i, row in enumerate(self.requests, 1):
            table.add_row(*self.row_values(row, i), key=f"r{i}")
        self.live_row = None  # rebuilt by the next progress line
        self.live_phase = None

    def refresh_chips(self) -> None:
        if not self.is_mounted or not self.chips.is_mounted:
            return
        running = self.server.running()
        est = srv.vram_estimate(self.collect())
        used, total = srv.vram()
        free = f"{(total - (used or 0)) / 1024 ** 3:.1f}" if total else "?"
        state = "[green]● running[/green]" if running else ("[yellow]● demo[/yellow]" if self.demo_thread
                                                            else "[red]● stopped[/red]")
        uptime = ""
        if running and self.started_at:
            secs = int(time.monotonic() - self.started_at)
            uptime = f" · up {secs // 3600:02d}:{secs % 3600 // 60:02d}:{secs % 60:02d}"
        model = Path(str(self.values.get("model") or "")).name or "no model"
        if self.narrow:
            text = (f"{state} {self.values.get('host')}:{self.values.get('port')} · "
                    f"ctx {self.values.get('ctx')}{uptime}")
        else:
            text = (f"{state} [b]{escape(model)}[/b] · {self.values.get('host')}:{self.values.get('port')} · "
                    f"ctx {self.values.get('ctx')}{uptime} · "
                    f"[yellow]VRAM est. {est:.1f} / {free} GiB free[/yellow]")
        if text != self.chips_text:  # a slow ssh link should not get a redraw per tick
            self.chips_text = text
            self.chips.update(text)
        text = self._stats_text()
        if text != self.stats_text:
            self.stats_text = text
            self.stats.update(text)

    def _stats_text(self) -> str:
        total_out = sum(int(r.get("completion_tokens") or 0) for r in self.requests)
        total_prompt = sum(int(r.get("prompt_tokens") or 0) for r in self.requests)
        avg = [float(r.get("decode_tps") or 0) for r in self.requests if r.get("decode_tps")]
        head = (f"requests: {len(self.requests)} · prompt {total_prompt / 1000:.0f}k tok · "
                f"generated {total_out} tok")
        return head + (f" · avg {sum(avg) / len(avg):.1f} t/s decode" if avg else "")

    # --- actions

    def action_tab(self, group: str) -> None:
        self.query_one(TabbedContent).active = f"tab-{group}"

    def action_clear_log(self) -> None:
        self.logview.clear()

    def action_toggle_errors(self) -> None:
        self.errors_only = not self.errors_only
        self.notify("log: errors only" if self.errors_only else "log: everything")

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_quit(self) -> None:
        if not self.server.running():
            self.exit()
            return

        def answer(quit_now) -> None:
            if quit_now:
                self.exit()

        self.push_screen(ConfirmQuit(), answer)

    def action_profiles(self) -> None:
        def loaded(values) -> None:
            if isinstance(values, dict):
                self.apply(values)
                self.logview.write("[dim]profile applied; F9 shows the command[/dim]")
        self.push_screen(ProfileScreen(self.collect()), loaded)

    def action_command(self) -> None:
        def chosen(what) -> None:
            if what == "start":
                self.action_start()
        own = {self.server.proc.pid} if self.server.running() and self.server.proc else set()
        self.push_screen(CommandScreen(self.collect(), self.binary, self.exec_command, own), chosen)

    def action_copy(self) -> None:
        env, argv = self.command()
        text = schema.format_command(env, argv)
        self.copy_to_clipboard(text)
        self.last_copied = text
        self.notify("command copied")

    def action_export(self, path: str | None = None) -> None:
        if not self.requests:
            self.notify("no requests yet", severity="warning")
            return
        if path is None:
            path = "omph-tui-requests-" + datetime.now(tz=UTC).astimezone().strftime("%Y%m%d-%H%M%S") + ".csv"
        out = Path(path)
        # the cell keys, so every column a row has is in the file; the formatted
        # cells plus the numbers the bench results use
        fields = (["#"] + [key for _, key in COLS if key != "num"]
                  + ["prompt_tokens", "cached_tokens", "prefill_ms", "completion_tokens", "decode_ms",
                     "drafted", "accepted", "restored", "client_gone"])
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for i, row in enumerate(self.requests, 1):
                cells = self.cell_values(row, i)
                writer.writerow({
                    "#": i,
                    **{k: cells.get(k, "") for k in fields if k in cells},
                    "prompt_tokens": row.get("prompt_tokens"),
                    "cached_tokens": row.get("cached_tokens"),
                    "prefill_ms": row.get("prefill_ms"),
                    "completion_tokens": row.get("completion_tokens"),
                    "decode_ms": row.get("decode_ms"),
                    "drafted": row.get("drafted"),
                    "accepted": row.get("accepted"),
                    "restored": bool(row.get("restored")),
                    "client_gone": bool(row.get("client_gone")),
                })
        self.notify(f"exported {out}")

    def action_details(self) -> None:
        self.open_details(self.table.cursor_row)

    def action_open_webui(self) -> None:
        """`w`: the server's own web UI (#378) in the browser."""
        if not self.server.running():
            self.notify("start the server first", severity="warning")
            return
        host = str(self.values.get("host") or "127.0.0.1")
        if host in ("0.0.0.0", "::", "[::]"):
            host = "127.0.0.1"  # a wildcard bind is not a connectable address
        url = f"http://{host}:{self.values.get('port')}/"
        self.notify(f"opening {url}")
        webbrowser.open(url)

    def action_test_prompt(self) -> None:
        values = self.collect()
        host, port = values.get("host") or "127.0.0.1", int(values.get("port") or 8080)
        headers = {"content-type": "application/json"}
        key = str(values.get("api_key") or "")
        if key:  # the server rejects an unauthenticated request otherwise
            headers["authorization"] = f"Bearer {key}"
        self.logview.write(f"[dim]test prompt -> http://{host}:{port}/v1/chat/completions[/dim]")

        def work() -> None:
            import json as _json
            import urllib.error
            import urllib.request
            body = _json.dumps({"messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 8,
                                "chat_template_kwargs": {"enable_thinking": False}}).encode()
            req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions", data=body,
                                         headers=headers)
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    data = _json.loads(resp.read())
                text = data["choices"][0]["message"].get("content") or ""
                self.call_from_thread(self.logview.write,
                                      f"[green]test ok[/green]: {escape(text.strip()[:60])!r} "
                                      f"in {time.monotonic() - t0:.1f} s")
            except urllib.error.HTTPError as exc:
                self.call_from_thread(self.logview.write,
                                      f"[red]test failed[/red]: {exc.code} {exc.read().decode()[:200]}")
            except OSError as exc:
                self.call_from_thread(self.logview.write, f"[red]test failed[/red]: {exc}")

        self.run_worker(work, thread=True)

    def start_demo(self) -> None:
        path = Path(self.demo_file or "")
        lines = path.read_text(errors="replace").splitlines()

        def feed() -> None:
            for line in lines:
                self.server.send_line(line)
                time.sleep(self.demo_delay)

        self.demo_thread = threading.Thread(target=feed, daemon=True)
        self.demo_thread.start()
        self.logview.write(f"[yellow]demo: replaying {path.name} ({len(lines)} lines)[/yellow]")

    @on(Button.Pressed)
    def _buttons(self, event: Button.Pressed) -> None:
        action = {"btn-start": self.action_start, "btn-stop": self.action_stop,
                  "btn-restart": self.action_restart}.get(str(event.button.id))
        if action:
            action()

    def open_details(self, row: int) -> None:
        index = row + 1
        if 0 < index <= len(self.requests) and not isinstance(self.screen, DetailScreen):
            self.push_screen(DetailScreen(self.requests[index - 1], index))

    @on(DataTable.RowSelected)
    def _row_selected(self, event: DataTable.RowSelected) -> None:
        self.open_details(event.cursor_row)

    @on(DataTable.CellSelected)
    def _cell_selected(self, event: DataTable.CellSelected) -> None:
        # a click arrives as CellSelected; with cursor_type="row" a keyboard
        # move arrives as RowSelected, so both open the detail (#304)
        self.open_details(event.coordinate.row)

    def on_unmount(self) -> None:
        self.server.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="the .omph to serve")
    ap.add_argument("--dflash", default=None,
                    help="the DFlash2 drafter .omph to fill in (default: the repo's when the file exists; "
                         "empty starts on the MTP head)")
    ap.add_argument("--mmproj", default=None,
                    help="the vision encoder to fill in, run on the CPU (default: the repo's when the file "
                         "exists; empty for no vision)")
    ap.add_argument("--profile", default="coding-agent", help="the profile to start from")
    ap.add_argument("--binary", default=str(HERE.parent / "engine/build/omph-server"))
    ap.add_argument("--exec", dest="exec_command", default=None,
                    help="run this command instead of omph-server (a fake server, for development)")
    ap.add_argument("--demo", default=None, help="replay a saved log file instead of starting anything")
    ap.add_argument("--poll", type=float, default=0.1, help="how often the UI drains the server's lines")
    ap.add_argument("--demo-delay", type=float, default=0.05, help="seconds between replayed lines")
    ap.add_argument("--screenshot", default=None, help="write an SVG screenshot and exit (headless)")
    args = ap.parse_args()

    values = profiles.load().get(args.profile, schema.defaults())
    # --dflash / --mmproj win, then the profile's own file, then the repo's when present
    dflash = resolve_file(args.dflash, values.get("dflash"), default_file(DEFAULT_DFLASH))
    mmproj = resolve_file(args.mmproj, values.get("mmproj"), default_file(DEFAULT_MMPROJ))
    values = {**values, "model": args.model, "dflash": dflash, "mmproj": mmproj}
    app = OmphTui(values, binary=args.binary, exec_command=args.exec_command, demo_file=args.demo,
                  poll=args.poll, demo_delay=args.demo_delay)

    if args.screenshot:
        # headless: a screenshot of the real widgets, for the docs
        async def shoot() -> None:
            async with app.run_test(size=(140, 46)) as pilot:
                if args.demo:
                    await pilot.pause(1.0)
                app.save_screenshot(args.screenshot)
        import asyncio
        asyncio.run(shoot())
        return

    # A terminal hangup or a `kill` must stop the child: the default action of
    # these signals would skip on_unmount and leave the server holding VRAM
    # (#347). PR_SET_PDEATHSIG in server.py covers SIGKILL, which no handler
    # can.
    def _signal(signum, _frame) -> None:
        app.exit()

    for _sig in (signal.SIGHUP, signal.SIGTERM):
        try:
            signal.signal(_sig, _signal)
        except (ValueError, OSError):
            pass
    app.run()


if __name__ == "__main__":
    main()
