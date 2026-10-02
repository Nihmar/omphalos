"""Per-step breakdown of a decode from a kernel trace (#203).

    rocprofv3 --kernel-trace -f csv -d DIR -o run -- ../engine/build/omph-run <model> <ids> x.f32 \\
        --last-logits --gemv --generate 33 --gen-out g.txt [--draft-mtp 3]
    uv run python profile_step.py DIR/run_kernel_trace.csv [--steps N]

The decode starts after the last prefill kernel (a GEMM or the prefill
attention). Per step (N, default: the number of argmax launches): the kernel
time by group, the GEMVs against their bandwidth ceiling (the bytes of every
tensor of that type, read once per step, at 318 GB/s), and the time the GPU
sat idle between kernels (launch and host overhead).
"""

import argparse
import csv
import re
from collections import defaultdict

# MiB read per step by each GEMV type: every tensor of the type once
# (omph-gemv-bench --all-of-type; Q4_K plus the 682 MiB output head, which the
# bench leaves out), the MTP block's Q6_K excluded
GEMV_MIB = {"IQ3_S": 3474.0, "IQ3_XXS": 1979.1, "IQ4_XS": 2882.0, "Q4_K": 826.9 + 682.0, "IQ2_XS": 211.0,
            "IQ2_XXS": 94.1, "IQ2_S": 374.8, "Q2_K": 229.7}
BW = 318.3e9

GROUPS = [
    (r"gemv_iq3s", "gemv IQ3_S"), (r"gemv_iq3_", "gemv IQ3_XXS"), (r"gemv_iq4", "gemv IQ4_XS"),
    (r"gemv_q4k", "gemv Q4_K"), (r"gemv_iq2xxs", "gemv IQ2_XXS"), (r"gemv_iq2xs", "gemv IQ2_XS"),
    (r"gemv_iq2s", "gemv IQ2_S"), (r"gemv_q2k", "gemv Q2_K"), (r"gemv_q6k", "gemv Q6_K (MTP)"),
    (r"gemv_bf16", "gemv BF16"), (r"gemv", "gemv other"),
    (r"attention_dec|attention_gqa_merge|attention_gqa_kernel", "attention (decode)"),
    (r"attn_prep|kv_ring", "attention prep / KV write"),
    (r"gdn|delta|conv", "delta-net"), (r"argmax", "argmax"),
    (r"rms|norm", "norms"), (r"swiglu|silu|gate|add|mul|scale|copy|embed", "elementwise"),
]


def group(name: str) -> str:
    for pat, g in GROUPS:
        if re.search(pat, name):
            return g
    n = re.sub(r"\(.*", "", re.sub(r"omph::kernels::(\(anonymous namespace\)::)?", "", name)).replace("void ", "")
    return "other: " + re.sub(r"<.*", "", n)[:40]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()
    with open(args.trace) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["Start_Timestamp"]))
    last = max(i for i, r in enumerate(rows) if re.search(r"gemm_q|gemm_f16|attention_wmma", r["Kernel_Name"]))
    dec = rows[last + 1:]
    steps = args.steps or sum(1 for r in dec if "argmax" in r["Kernel_Name"])
    tot = defaultdict(float)
    cnt = defaultdict(int)
    busy = 0.0
    end = 0
    for r in dec:
        s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
        tot[group(r["Kernel_Name"])] += (e - s) / 1e6
        cnt[group(r["Kernel_Name"])] += 1
        busy += (max(e, end) - max(s, end)) / 1e6 if e > end else 0.0
        end = max(end, e)
    span = (int(dec[-1]["End_Timestamp"]) - int(dec[0]["Start_Timestamp"])) / 1e6
    kern = sum(tot.values())
    print(f"decode: {len(dec)} kernels over {steps} steps ({len(dec) / steps:.0f} launches per step)")
    print(f"per step: span {span / steps:.2f} ms, GPU busy {busy / steps:.2f} ms, idle {(span - busy) / steps:.2f} ms, "
          f"kernel time {kern / steps:.2f} ms")
    print(f"{'group':30s} {'ms/step':>8s} {'launches':>9s} {'ceiling':>8s} {'% of ceiling':>12s}")
    for g, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        ms = v / steps
        t = g.replace("gemv ", "")
        ceil = GEMV_MIB[t] * 1048576 / BW * 1e3 if t in GEMV_MIB else None
        print(f"{g:30s} {ms:8.2f} {cnt[g] / steps:9.1f} " +
              (f"{ceil:8.2f} {100 * ceil / ms:11.0f} %" if ceil else ""))
    gem = sum(v for g, v in tot.items() if g.replace("gemv ", "") in GEMV_MIB) / steps
    gceil = sum(GEMV_MIB.values()) * 1048576 / BW * 1e3
    print(f"GEMVs of the main model: {gem:.2f} ms/step vs {gceil:.2f} ms at 318 GB/s ({100 * gceil / gem:.0f} %)")


if __name__ == "__main__":
    main()
