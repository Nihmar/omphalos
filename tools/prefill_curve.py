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
                    args = SimpleNamespace(omph=a.omph, model=a.model, ctx=a.ctx, kv=kv, llama_server=a.llama_server)
                    port = free_port()
                    url = f"http://127.0.0.1:{port}"
                    cmd = server_cmd(engine, args, port, mtp == "1")
                    env = {**os.environ, **({"OMPH_KV_K4": "1"} if engine == "omphalos" and kv == "k4q4" else {})}
                    wait_vram_idle()
                    name = config_name(engine, mtp == "1", kv)
                    log = open(out.with_name(f"{out.stem}.{name}.server.log"), "w")  # noqa: SIM115
                    vram = VramPeak()
                    vram.start()
                    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        wait_health(url, proc)
                        try:
                            prefill(engine, url, ids[-2048:])  # warm-up, text no point uses
                        except OSError as e:  # llama.cpp with MTP and K8 runs out of VRAM at 106k
                            print(f"{name}: the server failed on the warm-up ({e}), skipped", flush=True)
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
                    summarize(rows, txt, a.header)
    summarize(rows, txt, a.header)


if __name__ == "__main__":
    main()
