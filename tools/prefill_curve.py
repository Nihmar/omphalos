"""Prefill speed curve: omphalos vs llama.cpp, K4/V4 and K8/V4, with and without MTP (#232).

Each point is a raw completion of N wikitext-2 tokens (token ids, no chat
template) and one generated token, sent to the engine's server with prompt
caching off; the speed is the server's own prefill time (`timings.prompt_ms`,
the whole prompt: with MTP loaded it includes the draft block's pass). Every
repeat takes its tokens from another place in the corpus, so no prefix can be
reused, and the server must report all N tokens as prefilled. After a server
starts, one 2k-token request warms it up. Median of --repeats runs per point.

    uv run python prefill_curve.py --llama-server <llama.cpp>/build-hip/bin/llama-server \\
        --out ../bench/results/prefill-curve-232.csv

Configurations run in the order: kv (k4q4 first, the target), engine, MTP on
then off. The CSV gets one row per request (resumable: points already in it
are skipped); a .txt next to it holds the median table.
"""

import argparse
import csv
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from niah import VramPeak, free_port, post, wait_health
from pelican import config_name, server_cmd, wait_vram_idle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIELDS = ["engine", "kv", "mtp", "length", "repeat", "prompt_n", "prompt_ms", "tps", "vram_peak_mib"]


def corpus_tokens(tokenize: str, model: str, text: Path, need: int, cache: Path) -> list[int]:
    if cache.exists():
        ids = [int(t) for t in cache.read_text().split()]
        if len(ids) >= need:
            return ids
    raw = text.read_text()[: need * 6]  # ~4.3 characters per token on wikitext
    ids = [int(t) for t in subprocess.run([tokenize, model], input=raw, capture_output=True, text=True,
                                          check=True).stdout.split()]
    if len(ids) < need:
        raise SystemExit(f"the corpus gave {len(ids)} tokens, {need} needed")
    cache.write_text(" ".join(map(str, ids)))
    return ids


def prefill(engine: str, url: str, ids: list[int]) -> dict:
    if engine == "omphalos":
        tm = post(url + "/v1/completions", {"prompt": ids, "max_tokens": 1, "temperature": 0})["timings"]
    else:
        tm = post(url + "/completion", {"prompt": ids, "n_predict": 1, "temperature": 0, "top_k": 1,
                                        "cache_prompt": False})["timings"]
    return tm


def summarize(rows: list[dict], path: Path, header: str) -> None:
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        groups.setdefault((r["kv"], r["engine"], r["mtp"], int(r["length"])), []).append(float(r["tps"]))
    lengths = sorted({k[3] for k in groups})
    configs = sorted({k[:3] for k in groups}, key=lambda c: (c[0] != "k4q4", c[1] != "omphalos", c[2] != "1"))
    lines = [header, "", "median prefill t/s (runs)", "",
             f"{'kv':5} {'engine':9} {'mtp':4}" + "".join(f"{n:>13}" for n in lengths)]
    for c in configs:
        cells = []
        for n in lengths:
            v = groups.get((*c, n))
            cells.append(f"{statistics.median(v):8.0f} ({len(v)})" if v else f"{'-':>13}")
        lines.append(f"{c[0]:5} {c[1]:9} {'on' if c[2] == '1' else 'off':4}" + "".join(f"{s:>13}" for s in cells))
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[2:]), flush=True)


def plot(rows: list[dict], path: Path) -> None:
    """Median prefill t/s against prompt length (log scale), one panel per
    cache: omphalos blue, llama.cpp orange; MTP loaded solid, not dashed."""
    med: dict[tuple, dict[int, float]] = {}
    for r in rows:
        med.setdefault((r["kv"], r["engine"], r["mtp"]), {}).setdefault(int(r["length"]), []).append(float(r["tps"]))
    med = {k: {n: statistics.median(v) for n, v in d.items()} for k, d in med.items()}
    lengths = sorted({n for d in med.values() for n in d})
    if not lengths:
        return
    ymax = max(v for d in med.values() for v in d.values()) * 1.1
    color = {"omphalos": "#2a78d6", "llama": "#eb6834"}
    W, H, L, R, T, B, gap = 920, 400, 60, 24, 64, 48, 40
    pw = (W - L - R - gap) / 2
    lo, hi = math.log(lengths[0]), math.log(lengths[-1]) + (len(lengths) == 1)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>',
           f'<text x="{L}" y="22" font-size="15" fill="#0b0b0b">Prefill speed (median t/s) vs prompt length</text>']
    lx = L
    for eng, name in (("omphalos", "omphalos"), ("llama", "llama.cpp")):
        for mtp, label in (("1", "MTP loaded"), ("0", "no MTP")):
            dash = "" if mtp == "1" else ' stroke-dasharray="6 4"'
            svg.append(f'<line x1="{lx}" y1="42" x2="{lx + 24}" y2="42" stroke="{color[eng]}" stroke-width="2"{dash}/>'
                       f'<text x="{lx + 30}" y="46" fill="#52514e">{name}, {label}</text>')
            lx += 190
    for i, kv in enumerate(("k4q4", "k8q4")):
        x0 = L + i * (pw + gap)
        y0, ph = T + 20, H - T - 20 - B

        def X(n: float, x0: float = x0) -> float:
            return x0 + (math.log(n) - lo) / (hi - lo) * pw

        def Y(v: float, y0: float = y0, ph: float = ph) -> float:
            return y0 + ph - v / ymax * ph

        svg.append(f'<text x="{x0}" y="{T + 8}" fill="#0b0b0b">{"K4/V4" if kv == "k4q4" else "K8/V4"}</text>')
        step = 200 if ymax > 600 else 100
        for v in range(0, int(ymax) + 1, step):
            svg.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x0 + pw}" y2="{Y(v):.1f}" stroke="#e8e7e3"/>')
            if i == 0:
                svg.append(f'<text x="{x0 - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#52514e">{v}</text>')
        for n in lengths:
            t = f"{n // 1024}k" if n >= 1024 and n % 1024 == 0 else (f"{n / 1000:g}k" if n >= 1000 else str(n))
            svg.append(f'<text x="{X(n):.1f}" y="{y0 + ph + 18}" text-anchor="middle" fill="#52514e">{t}</text>')
        for (k, eng, mtp), d in sorted(med.items()):
            if k != kv:
                continue
            pts = " ".join(f"{X(n):.1f},{Y(v):.1f}" for n, v in sorted(d.items()))
            dash = "" if mtp == "1" else ' stroke-dasharray="6 4"'
            svg.append(f'<polyline points="{pts}" fill="none" stroke="{color[eng]}" stroke-width="2"{dash}/>')
            for n, v in sorted(d.items()):
                svg.append(f'<circle cx="{X(n):.1f}" cy="{Y(v):.1f}" r="4" fill="{color[eng]}" stroke="#fcfcfb" '
                           f'stroke-width="2"><title>{eng} {kv} MTP {"on" if mtp == "1" else "off"}, {n} tokens: '
                           f'{v:.0f} t/s</title></circle>')
    svg.append(f'<text x="{W / 2}" y="{H - 8}" text-anchor="middle" fill="#52514e">prompt tokens (log scale)</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama-server", required=True)
    ap.add_argument("--lengths", default="512,2048,8192,16384,32768,65536,100000")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--kvs", default="k4q4,k8q4")
    ap.add_argument("--engines", default="omphalos,llama")
    ap.add_argument("--mtps", default="1,0")
    ap.add_argument("--ctx", type=int, default=106496)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    ap.add_argument("--text", default=str(ROOT / "models/datasets/wikitext-2-raw/wiki.train.raw"))
    ap.add_argument("--header", default="", help="first line of the .txt summary (conditions, commits)")
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]
    out = Path(a.out)
    txt = out.with_suffix(".txt")
    stride = max(lengths)
    ids = corpus_tokens(f"{a.omph}/omph-tokenize", a.model, Path(a.text), stride * a.repeats + 2048,
                        ROOT / "models/datasets/wiki-train-tokens.txt")
    rows = list(csv.DictReader(out.open())) if out.exists() else []
    done = {(r["engine"], r["kv"], r["mtp"], r["length"], r["repeat"]) for r in rows}
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for kv in a.kvs.split(","):
            for engine in a.engines.split(","):
                for mtp in a.mtps.split(","):
                    todo = [(n, r) for n in lengths for r in range(a.repeats)
                            if (engine, kv, mtp, str(n), str(r)) not in done]
                    if not todo:
                        continue
                    for ctx in (a.ctx, 69632):  # llama.cpp with MTP and K8 cannot allocate 106k: retry at 68k
                        todo = [(n, r) for n, r in todo if n < ctx - 16]
                        args = SimpleNamespace(omph=a.omph, model=a.model, ctx=ctx, kv=kv, llama_server=a.llama_server)
                        port = free_port()
                        url = f"http://127.0.0.1:{port}"
                        cmd = server_cmd(engine, args, port, mtp == "1")
                        env = {**os.environ, **({"OMPH_KV_K4": "1"} if engine == "omphalos" and kv == "k4q4" else {})}
                        wait_vram_idle()
                        name = config_name(engine, mtp == "1", kv)
                        log = open(out.with_name(f"{out.stem}.{name}.server.log"), "a")  # noqa: SIM115
                        vram = VramPeak()
                        vram.start()
                        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                        try:
                            try:
                                wait_health(url, proc)
                            except SystemExit as e:
                                print(f"{name}: the server did not start at context {ctx} ({e})", flush=True)
                                continue
                            try:
                                prefill(engine, url, ids[-2048:])  # warm-up, text no point uses
                            except OSError as e:  # llama.cpp with MTP and K8 runs out of VRAM at 106k
                                print(f"{name}: the server failed on the warm-up at context {ctx} ({e})", flush=True)
                                continue
                            for n, r in todo:
                                vram.reset()
                                try:
                                    tm = prefill(engine, url, ids[r * stride: r * stride + n])
                                except OSError as e:  # e.g. out of memory at the longest lengths
                                    print(f"{name} {n} #{r}: failed ({e})", flush=True)
                                    if proc.poll() is not None:
                                        break
                                    continue
                                if tm["prompt_n"] != n:
                                    raise SystemExit(f"{name} {n}: {tm['prompt_n']} tokens prefilled, {n} sent")
                                row = {"engine": engine, "kv": kv, "mtp": mtp, "length": n, "repeat": r,
                                       "prompt_n": tm["prompt_n"], "prompt_ms": round(tm["prompt_ms"], 1),
                                       "tps": round(n / tm["prompt_ms"] * 1000, 1), "vram_peak_mib": vram.reset()}
                                w.writerow(row)
                                f.flush()
                                rows.append({k: str(v) for k, v in row.items()})
                                print(f"{name} {n:>6} #{r}: {row['tps']:7.1f} t/s ({row['prompt_ms'] / 1000:.1f} s), "
                                      f"VRAM peak {row['vram_peak_mib']} MiB", flush=True)
                        finally:
                            proc.terminate()
                            proc.wait(60)
                            vram.stop.set()
                            log.close()
                            time.sleep(2)
                        break
                    summarize(rows, txt, a.header)
                    plot(rows, out.with_suffix(".svg"))
    summarize(rows, txt, a.header)
    plot(rows, out.with_suffix(".svg"))


if __name__ == "__main__":
    main()
