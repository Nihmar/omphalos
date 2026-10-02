"""Per-kernel time split of an omph-run kernel trace (#167): the prefill (up to
the last GEMM / WMMA-attention kernel) and the decode after it.

    rocprofv3 --kernel-trace -f csv -d DIR -o run -- ../engine/build/omph-run <model> <ids.txt> x.f32 \
        --last-logits --gemv --generate 12 --gen-out g.txt --draft-mtp 3
    uv run python profile_trace.py DIR/run_kernel_trace.csv 12   # 12: generated tokens
"""
import csv
import re
import sys
from collections import defaultdict


def group(name: str) -> str:
    n = re.sub(r"omph::kernels::(\(anonymous namespace\)::)?", "", name)
    n = n.replace("void ", "")
    n = re.sub(r"\(.*", "", n)                 # drop the argument list
    if "attention_gqa_kernel" in n or "attention_wmma" in n:
        return n                                    # keep the template (Q8, TQ)
    return re.sub(r"<.*", "", n)


def summarize(rows, title, per=None):
    tot = defaultdict(float)
    cnt = defaultdict(int)
    for r in rows:
        g = group(r["Kernel_Name"])
        tot[g] += (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e6
        cnt[g] += 1
    all_ms = sum(tot.values())
    attn = sum(v for k, v in tot.items() if "attention" in k)
    print(f"== {title}: {len(rows)} kernels, {all_ms:.1f} ms of kernel time, attention {attn:.1f} ms "
          f"({100 * attn / all_ms:.1f} %)" + (f", per generated token {all_ms / per:.1f} ms" if per else ""))
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:14]:
        print(f"   {v:9.2f} ms {100 * v / all_ms:5.1f} %  {cnt[k]:6d}x  {k[:90]}")


with open(sys.argv[1]) as f:
    rows = list(csv.DictReader(f))
rows.sort(key=lambda r: int(r["Start_Timestamp"]))
last = max(i for i, r in enumerate(rows) if re.search(r"gemm_q|gemm_f16|attention_wmma", r["Kernel_Name"]))
summarize(rows[:last + 1], "prefill")
summarize(rows[last + 1:], "decode", per=int(sys.argv[2]) if len(sys.argv) > 2 else None)
