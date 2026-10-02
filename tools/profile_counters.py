"""Hardware counters per kernel class (#203 phase 1.2), from one rocprofv3 pass per counter.

    for c in GRBM_GUI_ACTIVE VALUBusy ...; do
        rocprofv3 --pmc $c --kernel-include-regex "gemv_|gemm_|..." -f csv -d DIR/$c -o run -- omph-run ...
    done   # one counter per pass: groups can exceed what the hardware collects in one
    uv run python profile_counters.py DIR

Note (#203): on gfx1200 with rocprofiler-sdk 1.1 only GRBM_GUI_ACTIVE, SQ_WAVES and
SQ_BUSY_CYCLES return data; the other counters read 0 (the columns print 0).

DIR/<counter>/run_counter_collection.csv per counter. Each pass splits at its last
prefill kernel (a GEMM or the prefill attention) into prefill and decode; the
classes are profile_kernels.py's. Percent metrics (VALUBusy, LdsUtil, ...) are
averaged weighted by kernel time; counts are per launch; derived: the clock
(GRBM_GUI_ACTIVE over the kernel time) and the DRAM read bandwidth (the GL2C
read requests to memory by size, over the kernel time).
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from profile_kernels import classify

PERCENT = {"VALUBusy", "ValuPipeIssueUtil", "LdsUtil", "LDSBankConflict", "OccupancyPercent",
           "MeanOccupancyPerActiveCU", "MemUnitBusy", "L0CacheHit", "L2CacheHit"}


def key_of(name: str) -> str:
    cls, t = classify(name)
    k = f"{cls} {t}" if t else cls
    nt = re.search(r"gemv_\w+?(?:_ntg)?_kernel<(\d+)", name)
    if cls == "gemv" and nt and int(nt.group(1)) > 1:
        k += f" x{nt.group(1)}"
    return k


def load(path: Path):
    """(phase, class) -> [value sum, time sum in us, launches, VGPRs, LDS bytes] for one counter."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    disp = {}
    for r in rows:  # one row per dispatch and counter (several for some derived ones: summed)
        d = disp.setdefault(r["Dispatch_Id"], [r, 0.0])
        d[1] += float(r["Counter_Value"])
    order = sorted(disp.values(), key=lambda d: int(d[0]["Start_Timestamp"]))
    last = max((i for i, d in enumerate(order) if re.search(r"gemm_|attention_wmma", d[0]["Kernel_Name"])), default=-1)
    out = defaultdict(lambda: [0.0, 0.0, 0, 0, 0])
    for i, (r, v) in enumerate(order):
        k = ("prefill" if i <= last else "decode", key_of(r["Kernel_Name"]))
        us = (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3
        o = out[k]
        o[0] += v * us if path.parent.name in PERCENT else v
        o[1] += us
        o[2] += 1
        o[3] = int(r["VGPR_Count"])
        o[4] = int(r["LDS_Block_Size"])
    return out


def main() -> None:
    root = Path(sys.argv[1])
    data = {}
    for p in sorted(root.glob("*/run_counter_collection.csv")):
        data[p.parent.name] = load(p)
    keys = set()
    for d in data.values():
        keys |= set(d)
    base = data.get("GRBM_GUI_ACTIVE") or next(iter(data.values()))

    def per_launch(c, k):
        o = data.get(c, {}).get(k)
        return o[0] / o[2] if o and o[2] else None

    def pct(c, k):
        o = data.get(c, {}).get(k)
        return o[0] / o[1] if o and o[1] else None

    def f(v, fmt):
        return format(v, fmt) if v is not None else "-"

    cols = ["launches", "ms", "MHz", "VGPR", "LDS KB", "occ%", "VALU%", "issue%", "LDS%", "bank%", "mem%",
            "L2hit%", "DRAM GB/s", "valu/wave", "lds/wave", "vmem/wave", "wait%", "dep%"]
    print(" ".join(f"{c:>9s}" for c in ["class"] + cols))
    for phase in ("prefill", "decode"):
        print(f"--- {phase}")
        ks = sorted((k for k in keys if k[0] == phase), key=lambda k: -(base.get(k, [0, 0])[1]))
        for k in ks:
            o = base.get(k)
            if not o or o[1] < 1.0:
                continue
            us = o[1]
            clk = per_launch("GRBM_GUI_ACTIVE", k)
            rd = None
            if "GL2C_EA_RDREQ_128B_sum" in data:
                b = sum((data[c].get(k, [0])[0]) * s for c, s in
                        (("GL2C_EA_RDREQ_128B_sum", 128), ("GL2C_EA_RDREQ_64B_sum", 64), ("GL2C_EA_RDREQ_32B_sum", 32))
                        if c in data)
                rd = b / (us * 1e3)
            waves = per_launch("SQ_WAVES", k)
            wave_cyc = per_launch("SQ_WAVE_CYCLES", k)

            def per_wave(c, k=k, waves=waves):
                v = per_launch(c, k)
                return v / waves if v is not None and waves else None

            wait = per_launch("SQ_WAIT_INST_ANY", k)
            dep = per_launch("WAVE_DEP_WAIT", k)
            vals = [o[2], us / 1e3, clk / (us / o[2]) if clk else None, o[3], o[4] / 1024,
                    pct("OccupancyPercent", k), pct("VALUBusy", k), pct("ValuPipeIssueUtil", k), pct("LdsUtil", k),
                    pct("LDSBankConflict", k), pct("MemUnitBusy", k), pct("L2CacheHit", k), rd,
                    per_wave("SQ_INSTS_VALU"), per_wave("SQ_INSTS_LDS"), per_wave("SQ_INSTS_TEX_LOAD"),
                    100 * wait / wave_cyc if wait and wave_cyc else None,
                    100 * dep / wave_cyc if dep and wave_cyc else None]
            fmts = ["9d", "9.2f", "9.0f", "9d", "9.1f", "9.0f", "9.0f", "9.0f", "9.0f", "9.1f", "9.0f", "9.0f",
                    "9.0f", "9.0f", "9.0f", "9.0f", "9.0f", "9.0f"]
            print(f"{k[1][:30]:30s} " + " ".join(f(v, fm) for v, fm in zip(vals, fmts)))


if __name__ == "__main__":
    main()
