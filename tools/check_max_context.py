"""The largest context that is actually usable on the 16 GB card (#237).

A candidate context is only usable if a full prefill plus a decode actually
fits and runs, not only if the server loads: the driver evicts the desktop's
VRAM to make room, so "it loaded" can mean "it ran by pushing the compositor
out". This starts omph-server with --cache-ram 0 (no host checkpoints, the
configuration of the load sweep), sends one long prompt and a few generated
tokens, and reports:

  * VRAM and GTT after load (device total; the desktop included);
  * VRAM and GTT peaks over the request, and the peak during the decode
    phase (the server logs the summary line with the prefill/decode split);
  * prefill t/s and decode ms/token from the response's timings;
  * the stop reason, so a context-full stop is visible.

    uv run python check_max_context.py --ctx 196608 --kv k4 --spec mtp
    uv run python check_max_context.py --ctx 172032 --kv k4 --spec dflash \\
        --dflash ../models/Qwen3.8-27B-DFlash2-Q4_K_M.omph --generate 256

One row per invocation is appended to bench/results/max-context-237.txt (a
-- comment line with the conditions). tools/niah.py is the retrieval check
(one haystack at the same --ctx) for the positions past 128k.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import llama_pin
from niah import free_port, wait_health
from omph_model import omph_file

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "bench/results/max-context-237.txt"


def read_sysfs(name: str) -> int | None:
    for card in sorted(Path("/sys/class/drm").glob("card*/device")):
        try:
            return int((card / name).read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def mib(v: int | None) -> float:
    return (v or 0) / 1024 ** 2


class PeakVram(threading.Thread):
    """Poll VRAM / GTT while the request runs, keeping the maxima."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop = False
        self.vram = 0
        self.gtt = 0

    def run(self) -> None:
        while not self.stop:
            self.vram = max(self.vram, read_sysfs("mem_info_vram_used") or 0)
            self.gtt = max(self.gtt, read_sysfs("mem_info_gtt_used") or 0)
            time.sleep(0.1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    ap.add_argument("--text", default=str(ROOT / "models/datasets/wikitext-2-raw/wiki.train.raw"))
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--kv", choices=["k4", "mixed", "k8"], default="k4",
                    help="k4: K and V Q4 on every layer (#237's K4/V4); mixed: the default K4/K8 set; "
                         "k8: every layer K8")
    ap.add_argument("--spec", choices=["mtp", "dflash", "none"], default="mtp")
    ap.add_argument("--dflash", default=str(ROOT / "models/Qwen3.8-27B-DFlash2-Q4_K_M.omph"))
    ap.add_argument("--tokens", type=int, default=0, help="prompt tokens (0: ctx - 1024)")
    ap.add_argument("--generate", type=int, default=256)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    llama_ref = llama_pin.describe(llama_pin.load()["commit"])  # the reference (#393)

    prompt_tokens = args.tokens or (args.ctx - 1024)
    if prompt_tokens + args.generate >= args.ctx:
        sys.exit(f"prompt {prompt_tokens} + generate {args.generate} does not fit --ctx {args.ctx}")

    # The prompt: the first `prompt_tokens` ids of the wikitext.
    tokenize = f"{args.omph}/omph-tokenize"
    text = Path(args.text).read_text()
    cut = min(len(text), int(prompt_tokens * 4.3))
    ids: list[int] = []
    while cut <= len(text):  # the chars/token rate varies: grow until it is enough
        ids = [int(x) for x in subprocess.run([tokenize, args.model], input=text[:cut], capture_output=True,
                                              text=True, check=True).stdout.split()]
        if len(ids) >= prompt_tokens or cut == len(text):
            break
        cut = min(len(text), int(cut * 1.5))
    ids = ids[:prompt_tokens]
    if len(ids) < prompt_tokens:
        sys.exit(f"the text gave {len(ids)} tokens, wanted {prompt_tokens}")

    env = dict(os.environ)
    if args.kv == "k4":
        env["OMPH_KV_K4"] = "1"
    elif args.kv == "k8":
        env["OMPH_KV_K4_LAYERS"] = "none"
    cmd = [f"{args.omph}/omph-server", omph_file(args.model), "--port", str(free_port()), "--ctx", str(args.ctx),
           "--cache-ram", "0"]
    port = cmd[cmd.index("--port") + 1]
    if args.spec == "dflash":
        cmd += ["--dflash", args.dflash]
    elif args.spec == "none":
        cmd += ["--no-mtp"]
    log = open(f"/tmp/check_max_context-{args.ctx}.log", "w")  # noqa: SIM115
    peak = PeakVram()
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        wait_health(url, proc)
        time.sleep(1.0)
        load_vram, load_gtt = read_sysfs("mem_info_vram_used") or 0, read_sysfs("mem_info_gtt_used") or 0
        peak.vram, peak.gtt = load_vram, load_gtt
        peak.start()
        body = {"prompt": ids, "max_tokens": args.generate, "temperature": 0, "stream": False}
        req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            res = json.loads(r.read())
        wall = time.time() - t0
        peak.stop = True
        peak.join(timeout=2)
        tm = res.get("timings", {})
        prefill_ms = tm.get("prompt_ms", 0.0)
        decode_ms = tm.get("predicted_ms", 0.0)
        used = res.get("usage", {})
        log_line = Path(f"/tmp/check_max_context-{args.ctx}.log").read_text().strip().splitlines()
        summary = next((ln for ln in reversed(log_line) if "tokens in" in ln), "")
        stop = summary.split("stop: ")[-1].split(" (client gone)")[0] if "stop: " in summary else "?"
        cond = (f"ctx {args.ctx}, {args.kv}, {args.spec}, prompt {len(ids)}, generate {args.generate}"
                + (f", {args.note}" if args.note else ""))
        row = (f"{cond}: load {mib(load_vram):.0f} MiB, peak {mib(peak.vram):.0f} MiB "
               f"(gtt {mib(load_gtt):.0f} -> {mib(peak.gtt):.0f}), "
               f"prefill {prefill_ms / 1000:.1f} s ({1000.0 * tm.get('prompt_n', 0) / prefill_ms:.0f} t/s), "
               f"decode {decode_ms / max(1, tm.get('predicted_n', 0)):.1f} ms/token, "
               f"prompt+generated {used.get('prompt_tokens')}+{used.get('completion_tokens')}, stop {stop}, "
               f"wall {wall:.0f} s; {llama_ref}")
        print(row, flush=True)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            out.write_text("# #237: the largest usable context, one line per configuration (01_desktop included).\n"
                           "# check_max_context.py: VRAM/GTT after load and peak over one prompt + decode;\n"
                           "# --cache-ram 0 (no host checkpoints), greedy.\n"
                           "# The llama.cpp reference of each row is its trailing label (bench/llama.cpp.pin, #393).\n")
        with out.open("a") as f:
            f.write(row + "\n")
    finally:
        peak.stop = True
        proc.terminate()
        proc.wait(60)
        log.close()


if __name__ == "__main__":
    main()
