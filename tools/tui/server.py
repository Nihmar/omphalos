"""Spawn and supervise one omph-server (#304).

The rules the process handling follows, learned the hard way during the #287
runs: the child gets its own session, so a `SIGTERM` reaches the whole group;
the TUI always kills what it started (two orphan servers held 12 GB of VRAM
each until they were hunted down); and the queue the reader thread fills is the
only thing the UI thread touches, so nothing is called across threads.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

KIB_PER_TOKEN = 26 * 1024        # the default K/Q8 + V/Q4 cache, #58's 0.87 GB at 32k
MTP_MIB = 352                     # the MTP block's weights
DFLASH_MIB = 1100                 # the DFlash2 drafter's weights (Q4_K_M)


def read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def other_servers() -> list[int]:
    """Pids of omph-server processes that are not ours (a stray one eats VRAM)."""
    out = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if (entry / "comm").read_text().strip() == "omph-server":
                out.append(int(entry.name))
        except OSError:
            continue
    return out


def port_free(host: str, port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def vram() -> tuple[int | None, int | None]:
    """(used, total) bytes from sysfs, for the estimate in the title bar."""
    for card in sorted(Path("/sys/class/drm").glob("card*/device")):
        total = read(card / "mem_info_vram_total")
        used = read(card / "mem_info_vram_used")
        if total:
            return (int(used) if used else None, int(total))
    return (None, None)


def vram_estimate(values: dict) -> float:
    """A rough GiB figure for the settings of a form: the weights plus the KV
    the context needs plus the drafter. An estimate, not a promise."""
    model = str(values.get("model") or "")
    path = Path(model)
    weights = path.stat().st_size / 1024 ** 3 if model and path.is_file() else 11.3
    ctx = int(values.get("ctx") or 8192)
    kv = ctx * KIB_PER_TOKEN / 1024 ** 3
    extra = 0.0
    if not values.get("no_mtp"):
        extra += MTP_MIB / 1024
    if values.get("dflash"):
        extra += DFLASH_MIB / 1024
    return weights + kv + extra


def preflight(values: dict) -> list[tuple[str, str, str]]:
    """(level, label, detail) checks to show before starting: level is
    'ok', 'warn' or 'err'."""
    checks: list[tuple[str, str, str]] = []
    model = str(values.get("model") or "")
    if not model:
        checks.append(("err", "model", "no .omph chosen"))
    elif not Path(model).exists():
        checks.append(("err", "model", f"{model} does not exist"))
    elif model.endswith(".gguf"):
        checks.append(("err", "model", "the engine loads .omph only: run omph-convert"))
    else:
        checks.append(("ok", "model", f"{Path(model).name} ({Path(model).stat().st_size / 1024 ** 3:.2f} GiB)"))
    for key, flag in (("dflash", "--dflash"), ("mmproj", "--mmproj")):
        path = str(values.get(key) or "")
        if path and not Path(path).exists():
            checks.append(("err", key, f"{path} does not exist"))
        elif path:
            checks.append(("ok", key, Path(path).name))
    host, port = str(values.get("host") or "127.0.0.1"), int(values.get("port") or 8080)
    checks.append(("ok", "port", f"{host}:{port} free") if port_free(host, port)
                  else ("err", "port", f"{host}:{port} is in use"))
    others = other_servers()
    checks.append(("warn", "servers", f"another omph-server is running: {others}") if others
                  else ("ok", "servers", "no other omph-server"))
    est = vram_estimate(values)
    used, total = vram()
    if total:
        free = (total - (used or 0)) / 1024 ** 3
        level = "err" if est > free else ("warn" if est > free - 0.5 else "ok")
        checks.append((level, "vram", f"est. {est:.1f} GiB, {free:.1f} GiB free of {total / 1024 ** 3:.1f}"))
    else:
        checks.append(("warn", "vram", f"est. {est:.1f} GiB (no sysfs reading)"))
    return checks


class ServerProcess:
    """One child, its lines in a queue, and a stop that cannot leave it behind."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.queue: queue.Queue = queue.Queue()
        self.started_at: float | None = None
        self.argv: list[str] = []
        self.env: dict[str, str] = {}
        self._stopping = False

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, argv: list[str], env: dict[str, str] | None = None) -> None:
        self.stop()
        self.argv = list(argv)
        self.env = dict(env or {})
        full_env = {**os.environ, **self.env}
        self.proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            env=full_env, start_new_session=True,
        )
        self.started_at = time.monotonic()
        self._stopping = False
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.queue.put(("started", {"pid": self.proc.pid, "argv": self.argv, "env": self.env}))

    def _reader(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self.queue.put(("line", line.rstrip("\n")))
        code = proc.wait()
        self.queue.put(("exit", code))

    def stop(self, timeout: float = 4.0) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            self.proc = None
            return
        self._stopping = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + timeout
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
        self.proc = None
        self.queue.put(("stopped", {"pid": proc.pid}))

    def send_line(self, line: str) -> None:
        """Feed a line as if the server had printed it (used by --demo)."""
        self.queue.put(("line", line))
