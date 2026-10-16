"""AppRun's two lookups, and what it must not touch (#534).

The AppImage's own logic is this one shell script, so it is worth driving
directly: a stub AppDir (the real AppRun, a stub interpreter that prints its
argv) run from real directories. No GPU, no model, no AppImage build.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

APPRUN = Path(__file__).resolve().parents[1] / "appimage/AppRun"


def appdir(tmp_path: Path) -> Path:
    """An AppDir whose interpreter prints what it was given."""
    app = tmp_path / "squashfs-root"
    python = app / "usr/lib/omphalos/bin/python3"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\n'
                      'for a in "$@"; do echo "arg:$a"; done\n'
                      'echo "models:${OMPH_MODELS_DIR:-}"\n'
                      'echo "pythonpath:${PYTHONPATH:-}"\n'
                      'echo "usersite:${PYTHONNOUSERSITE:-}"\n')
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    (app / "usr/lib/omphalos/tools").mkdir()
    (app / "usr/lib/omphalos/tools/omph_tui.py").write_text("")
    run = app / "AppRun"
    run.write_text(APPRUN.read_text())
    run.chmod(run.stat().st_mode | stat.S_IXUSR)
    return app


def run_app(app: Path, cwd: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    """The AppRun from `cwd`, in a shell that exports nothing but what a distro
    would (no OMPH_*, no PYTHONPATH)."""
    home = cwd / "home"
    home.mkdir(exist_ok=True)
    e = {"PATH": "/usr/bin:/bin", "HOME": str(home), "LC_ALL": "C", **env}
    return subprocess.run([str(app / "AppRun"), *args], cwd=cwd, env=e,
                          capture_output=True, text=True, check=False)


def test_the_checkout_supplies_both(tmp_path) -> None:
    """Run from a checkout: its models/ and its engine/build/omph-server."""
    app = appdir(tmp_path)
    here = tmp_path / "omphalos"
    (here / "models").mkdir(parents=True)
    (here / "engine/build").mkdir(parents=True)
    server = here / "engine/build/omph-server"
    server.write_text("#!/bin/sh\n")
    server.chmod(server.stat().st_mode | stat.S_IXUSR)

    p = run_app(app, here, "--profile", "coding-agent")
    assert p.returncode == 0, p.stderr
    assert f"arg:--binary\narg:{server}\n" in p.stdout
    assert f"models:{here / 'models'}\n" in p.stdout
    # ... and the caller's args are all still there, after the injected pair
    assert "arg:--profile\narg:coding-agent\n" in p.stdout


def test_outside_a_checkout_a_data_dir_is_created(tmp_path) -> None:
    """No models/ and no server to find: the data directory is created, and no
    --binary is invented (the preflight names what is missing)."""
    app = appdir(tmp_path)
    here = tmp_path / "elsewhere"
    here.mkdir()
    data = tmp_path / "xdg"

    p = run_app(app, here, "--demo", "log.txt", XDG_DATA_HOME=str(data))
    assert p.returncode == 0, p.stderr
    assert (data / "omphalos/models").is_dir()
    assert f"models:{data / 'omphalos/models'}\n" in p.stdout
    assert "arg:--binary" not in p.stdout
    assert "arg:--demo\narg:log.txt\n" in p.stdout


def test_the_server_in_the_path_is_used(tmp_path) -> None:
    app = appdir(tmp_path)
    here = tmp_path / "elsewhere"
    here.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    server = bindir / "omph-server"
    server.write_text("#!/bin/sh\n")
    server.chmod(server.stat().st_mode | stat.S_IXUSR)

    p = run_app(app, here, PATH=f"{bindir}:/usr/bin:/bin")
    assert p.returncode == 0, p.stderr
    assert f"arg:--binary\narg:{server}\n" in p.stdout


def test_an_explicit_choice_is_left_alone(tmp_path) -> None:
    """--binary wins, and --exec (a fake server) asks for no binary at all."""
    app = appdir(tmp_path)
    here = tmp_path / "omphalos"
    (here / "models").mkdir(parents=True)
    (here / "engine/build").mkdir(parents=True)
    server = here / "engine/build/omph-server"
    server.write_text("#!/bin/sh\n")
    server.chmod(server.stat().st_mode | stat.S_IXUSR)

    p = run_app(app, here, "--binary", "/opt/omph-server")
    assert p.returncode == 0, p.stderr
    assert "arg:--binary\narg:/opt/omph-server\n" in p.stdout
    assert str(server) not in p.stdout

    for other in (("--binary=/opt/omph-server",), ("--exec", "cat"), ("--exec=cat",)):
        p = run_app(app, here, *other)
        assert p.returncode == 0, p.stderr
        assert str(server) not in p.stdout, other


def test_the_models_dir_can_be_set_from_outside(tmp_path) -> None:
    app = appdir(tmp_path)
    here = tmp_path / "omphalos"
    (here / "models").mkdir(parents=True)

    p = run_app(app, here, OMPH_MODELS_DIR="/srv/models")
    assert "models:/srv/models\n" in p.stdout


def test_the_shell_cannot_shadow_the_bundle(tmp_path) -> None:
    """A stale PYTHONPATH (or a user site) must not reach the bundled Textual."""
    app = appdir(tmp_path)
    here = tmp_path / "elsewhere"
    here.mkdir()

    p = run_app(app, here, PYTHONPATH="/tmp/old-textual", PYTHONHOME="/tmp/old-python")
    assert p.returncode == 0, p.stderr
    assert "pythonpath:\n" in p.stdout
    assert "usersite:1\n" in p.stdout


def test_a_broken_bundle_says_so(tmp_path) -> None:
    app = appdir(tmp_path)
    here = tmp_path / "elsewhere"
    here.mkdir()
    os.remove(app / "usr/lib/omphalos/bin/python3")

    p = run_app(app, here)
    assert p.returncode == 1
    assert "no bundled interpreter" in p.stderr
