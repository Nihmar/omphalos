"""Pelican riding a bicycle: one SVG prompt on omphalos and llama.cpp (#229).

The prompt is rendered once with the GGUF's chat template by omph-tokenize
(thinking on, reasoning effort xhigh: the template's default) and sent as the
same raw prompt to either engine's completion endpoint, in four configurations
per engine: temperature 0 and 1, each without and with MTP speculative
decoding. --kv picks the KV cache: k8q4 (the default: omphalos' own mix,
llama.cpp q8_0 / q4_0) or k4q4 (omphalos OMPH_KV_K4=1, llama.cpp q4_0 / q4_0);
k4q4 runs are tagged so. The generation runs until the model stops (context 64K, no thinking
budget). Temperature 1 samples with Qwen's recommended top_k 20, top_p 0.95,
min_p 0 and a fixed seed on both engines; temperature 0 is greedy.

    uv run python pelican.py --llama-server <llama.cpp>/build-hip/bin/llama-server \\
        --out ../bench/results/pelican-229

Per run: <engine>-<mtp|plain>[-k4q4]-t<T>.svg (the answer's SVG), .png (its first
frame, rsvg-convert), .txt (thinking and answer) and .json (tokens, times,
speeds, peak VRAM); README.md summarizes them. Peak VRAM is the GPU's
mem_info_vram_used (sysfs) sampled every 0.2 s during the request, with the
idle level before the server started.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import threading
import time
import typing
from pathlib import Path

from niah import free_port, post, wait_health
from omph_model import omph_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROMPT = ("Write SVG code that displays a 2D animation of a pelican riding a bicycle. "
          "No additional testing is required.")
THINK_END = "</think>"
SEED = 229


def vram_used() -> int:
    vals = []
    for f in glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"):
        try:
            vals.append(int(Path(f).read_text()))
        except OSError:
            pass
    return max(vals) if vals else 0


def wait_vram_idle(limit: int = 2_000_000_000) -> int:
    """ROCm can hold VRAM for a moment after a process exits."""
    for _ in range(120):
        v = vram_used()
        if v < limit:
            return v
        time.sleep(1)
    return vram_used()


class PeakVram:
    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, vram_used())
            time.sleep(0.2)

    def __enter__(self) -> typing.Self:
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join()


def render(tokenize: str, model: str) -> str:
    req = {"messages": [{"role": "user", "content": PROMPT}], "add_generation_prompt": True}
    return subprocess.run([tokenize, model, "--chat"], input=json.dumps(req), capture_output=True, text=True,
                          check=True).stdout


def complete(engine: str, url: str, prompt: str, max_tokens: int, temp: float) -> tuple[str, dict]:
    samp = {"temperature": temp, "seed": SEED}
    if temp > 0:
        samp.update({"top_k": 20, "top_p": 0.95, "min_p": 0.0})
    if engine == "omphalos":
        r = post(url + "/v1/completions", {"prompt": prompt, "max_tokens": max_tokens, **samp}, timeout=14400)
        return r["choices"][0]["text"], r["timings"]
    if temp == 0:
        samp.update({"top_k": 1})
    r = post(url + "/completion", {"prompt": prompt, "n_predict": max_tokens, "cache_prompt": False, **samp},
             timeout=14400)
    return r["content"], r["timings"]


def extract_svg(answer: str) -> str:
    blocks = re.findall(r"```(?:svg|xml|html)?\s*\n(.*?)```", answer, re.DOTALL)
    for b in reversed(blocks):
        m = re.search(r"<svg\b.*</svg>", b, re.DOTALL)
        if m:
            return m.group(0)
    m = re.search(r"<svg\b.*</svg>", answer, re.DOTALL)
    return m.group(0) if m else ""


def config_name(engine: str, mtp: bool, kv: str) -> str:
    return f"{engine}-{'mtp' if mtp else 'plain'}" + ("-k4q4" if kv == "k4q4" else "")


def server_cmd(engine: str, args, port: int, mtp: bool) -> list[str]:
    if engine == "omphalos":
        cmd = [f"{args.omph}/omph-server", omph_file(args.model), "--port", str(port), "--ctx", str(args.ctx),
               "--cache-ram", "0"]  # no prompt checkpoints: every run prefills
        return cmd + ([] if mtp else ["--no-mtp"])
    cmd = [args.llama_server, "-m", args.model, "--port", str(port), "-c", str(args.ctx), "-ngl", "999", "-fa",
           "on", "-ctk", "q4_0" if args.kv == "k4q4" else "q8_0", "-ctv", "q4_0", "-np", "1", "--no-webui"]
    if mtp:  # the fork's MTP flags (bench/m0_mtp_ab.sh), 3 drafts as omphalos
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-p-min", "0.1",
                "--cache-type-k-draft", "q4_0", "--cache-type-v-draft", "q4_0"]
    return cmd


def run_config(engine: str, mtp: bool, args, prompt: str, prompt_n: int, out: Path, rows: list) -> None:
    name = config_name(engine, mtp, args.kv)
    env = {**os.environ, **({"OMPH_KV_K4": "1"} if engine == "omphalos" and args.kv == "k4q4" else {})}
    idle = wait_vram_idle()
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    with open(out / f"{name}.server.log", "w") as log:
        proc = subprocess.Popen(server_cmd(engine, args, port, mtp), stdout=log, stderr=subprocess.STDOUT,
                                env=env)
        try:
            wait_health(url, proc)
            for temp in args.temps:
                tag = f"{name}-t{temp:g}"
                if args.only and tag not in args.only:
                    continue
                t0 = time.time()
                with PeakVram() as pv:
                    text, tm = complete(engine, url, prompt, args.ctx - prompt_n - 64, temp)
                wall = time.time() - t0
                answer = text.partition(THINK_END)[2]
                svg = extract_svg(answer)
                (out / f"{tag}.txt").write_text(text)
                png = ""
                if svg:
                    (out / f"{tag}.svg").write_text(svg)
                    r = subprocess.run(["rsvg-convert", "-w", "800", "-o", str(out / f"{tag}.png"),
                                        str(out / f"{tag}.svg")], capture_output=True, text=True, check=False)
                    png = f"{tag}.png" if r.returncode == 0 else ""
                row = {
                    "run": tag, "engine": engine, "mtp": mtp, "kv": args.kv, "temperature": temp,
                    "tokens": tm["predicted_n"], "thinking_closed": THINK_END in text,
                    "total_s": round(wall, 1),
                    "prefill_tps": round(tm["prompt_n"] / tm["prompt_ms"] * 1000, 1) if tm["prompt_ms"] else 0,
                    "decode_tps": round(tm["predicted_n"] / tm["predicted_ms"] * 1000, 2) if tm["predicted_ms"] else 0,
                    "prompt_n": tm["prompt_n"], "peak_vram_mib": round(pv.peak / 2**20),
                    "idle_vram_mib": round(idle / 2**20), "svg": bool(svg), "png": png,
                }
                (out / f"{tag}.json").write_text(json.dumps(row, indent=1) + "\n")
                rows.append(row)
                print(f"{tag}: {row['tokens']} tok in {row['total_s']} s, decode {row['decode_tps']} t/s, "
                      f"prefill {row['prefill_tps']} t/s, peak VRAM {row['peak_vram_mib']} MiB, "
                      f"svg {'yes' if svg else 'NO'}", flush=True)
        finally:
            proc.terminate()
            proc.wait()


def write_readme(out: Path, rows: list, args) -> None:
    rows = sorted(rows, key=lambda r: (r["engine"], r["mtp"], r.get("kv", "k8q4"), r["temperature"]))
    lines = [
        "# Pelican riding a bicycle (#229)", "",
        f"Prompt: \"{PROMPT}\"", "",
        ("Rendered with the model's chat template (thinking on, reasoning effort xhigh, the template's default), "
         "the same raw prompt to omphalos (`omph-server`) and llama.cpp (`llama-server`, HIP), context 64K, no "
         "thinking budget. KV cache k8q4: omphalos' default (K4 on 8 layers, V4), llama.cpp `-ctk q8_0 -ctv "
         "q4_0`; k4q4 runs: omphalos `OMPH_KV_K4=1`, llama.cpp `-ctk q4_0 -ctv q4_0`. Temperature 0 is greedy; temperature 1 samples with top_k 20, top_p 0.95, min_p 0, "
         f"seed {SEED}. MTP: omphalos' speculative decoding (3 drafts), llama.cpp's `--spec-type draft-mtp "
         "--spec-draft-n-max 3`. The images are each SVG's first frame (`rsvg-convert`); open the `.svg` "
         "files in a browser for the animation. `tools/pelican.py` produced everything here."), "",
        "| run | KV | tokens (thinking + answer) | total time | prefill | decode | peak VRAM (idle) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['run']} | {r.get('kv', 'k8q4')} | {r['tokens']}{'' if r['thinking_closed'] else ' (thinking not closed)'} | "
                     f"{r['total_s']} s | {r['prefill_tps']} t/s | {r['decode_tps']} t/s | "
                     f"{r['peak_vram_mib']} MiB ({r['idle_vram_mib']}) |")
    lines.append("")
    for r in rows:
        lines += [f"## {r['run']}", "",
                  (f"{r['tokens']} tokens, {r['total_s']} s, prefill {r['prefill_tps']} t/s, decode "
                   f"{r['decode_tps']} t/s, peak VRAM {r['peak_vram_mib']} MiB"), ""]
        lines.append(f"![{r['run']}]({r['png']})" if r["png"] else "(no SVG in the answer)")
        lines.append("")
    (out / "README.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama-server", required=True)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--temps", default="0,1")
    ap.add_argument("--kv", default="k8q4", choices=["k8q4", "k4q4"])
    ap.add_argument("--mtp", default="0,1", help="0: plain, 1: MTP speculative decoding")
    ap.add_argument("--engines", default="omphalos,llama")
    ap.add_argument("--only", default="", help="only these runs, e.g. omphalos-mtp-t0,llama-plain-t1")
    args = ap.parse_args()
    args.temps = [float(t) for t in args.temps.split(",")]
    args.only = set(filter(None, args.only.split(",")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompt = render(f"{args.omph}/omph-tokenize", args.model)
    prompt_n = len(subprocess.run([f"{args.omph}/omph-tokenize", args.model], input=prompt, capture_output=True,
                                  text=True, check=True).stdout.split())
    rows = []
    for f in sorted(out.glob("*.json")):  # resumable: keep the runs already done
        rows.append(json.loads(f.read_text()))
    done = {r["run"] for r in rows}
    for engine in args.engines.split(","):
        for mtp in [m == "1" for m in args.mtp.split(",")]:
            name = config_name(engine, mtp, args.kv)
            todo = [t for t in args.temps
                    if f"{name}-t{t:g}" not in done and (not args.only or f"{name}-t{t:g}" in args.only)]
            if todo:
                saved = args.temps
                args.temps = todo
                run_config(engine, mtp, args, prompt, prompt_n, out, rows)
                args.temps = saved
                write_readme(out, rows, args)
    write_readme(out, rows, args)


if __name__ == "__main__":
    main()
