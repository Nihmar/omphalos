"""Per-kernel-class profile of an omph-run kernel trace against measured ceilings (#203).

    rocprofv3 --kernel-trace -f csv -d DIR -o run -- ../engine/build/omph-run <model.omph> <ids> x.f32 \\
        --last-logits --gemv --generate 65 --gen-out g.txt [--draft-mtp 3]
    uv run python profile_kernels.py DIR/run_kernel_trace.csv <model.gguf> --prompt 512 [--ctx 512] [--gen 64]

The trace splits at the last prefill kernel (a GEMM or the prefill attention;
the load's uploads before the first kernel are left out):
the prefill is reported in totals, the decode per step (one argmax launch per
step) and per generated token (--gen). Each class of kernels gets its ceiling
from bench/results/ceilings-203.txt and the model's tensors:

    GEMV of type T      launches x the mean bytes of a T tensor / DRAM 318.3 GB/s
    GEMM of type T      2 x the elements of every T tensor x the prompt tokens / f16 WMMA 88.7 TFLOPS
    prefill attention   causal QK + PV FLOPs of the 16 attention layers / 88.7 TFLOPS
    decode attention    the KV bytes at --ctx (K8 or K4 per layer, V4) / 318.3 GB/s

and the loss (time - ceiling). The other classes (delta-net, norms, ...) have
no ceiling here: their time is the loss. "idle" is the GPU time between kernels.
"""

import argparse
import csv
import re
from collections import defaultdict

DRAM = 318.3e9        # B/s, bench/bw_membench.hip
WMMA_F16 = 88.7e12    # FLOP/s, bench/gemm/wmma_rates.hip (random operands, 160 W cap)
TYPES = {"IQ3_S": "iq3s", "IQ3_XXS": "iq3xxs", "IQ4_XS": "iq4xs", "Q4_K": "q4k", "IQ2_XXS": "iq2xxs",
         "IQ2_XS": "iq2xs", "IQ2_S": "iq2s", "Q2_K": "q2k", "IQ1_M": "iq1m", "Q6_K": "q6k", "BF16": "bf16"}
# kernel name -> (class, GGUF type or None); first match wins
CLASSES = [
    (r"gemv_iq3s", "gemv", "IQ3_S"), (r"gemv_iq3_kernel", "gemv", "IQ3_XXS"), (r"gemv_iq4", "gemv", "IQ4_XS"),
    (r"gemv_q4k", "gemv", "Q4_K"), (r"gemv_iq2xxs", "gemv", "IQ2_XXS"), (r"gemv_iq2xs", "gemv", "IQ2_XS"),
    (r"gemv_iq2s", "gemv", "IQ2_S"), (r"gemv_q2k", "gemv", "Q2_K"), (r"gemv_iq1_m", "gemv", "IQ1_M"),
    (r"gemv_q6k", "gemv", "Q6_K"), (r"gemv_bf16", "gemv", "BF16"), (r"gemv", "gemv", None),
    (r"gemm_iq3s", "gemm", "IQ3_S"), (r"DecIq3sTile|DecIq3s\b", "gemm", "IQ3_S"), (r"DecIq3xxs", "gemm", "IQ3_XXS"),
    (r"DecIq4xs", "gemm", "IQ4_XS"), (r"DecQ4k", "gemm", "Q4_K"), (r"DecIq2xxs", "gemm", "IQ2_XXS"),
    (r"DecIq2xs", "gemm", "IQ2_XS"), (r"DecIq2s", "gemm", "IQ2_S"), (r"DecQ2k", "gemm", "Q2_K"),
    (r"DecIq1m", "gemm", "IQ1_M"), (r"DecQ6k", "gemm", "Q6_K"), (r"gemm", "gemm", None),
    (r"dequant", "dequant", None),
    (r"attention_wmma", "attention prefill", None),
    (r"attention_gqa_merge", "attention merge", None),
    (r"attention_dec|attention_gqa", "attention decode", None),
    (r"attn_prep|kv_ring", "attention prep / KV write", None),
    (r"delta|gdn|conv", "delta-net", None),
    (r"rms|norm", "norms", None), (r"swiglu", "swiglu", None), (r"argmax|sample|topk|softmax", "argmax / sampling", None),
    (r"rocclr_copy|rocclr_fill", "copies / fills", None),
]


def classify(name: str) -> tuple[str, str | None]:
    for pat, cls, typ in CLASSES:
        if re.search(pat, name):
            return cls, typ
    n = re.sub(r"\(.*", "", re.sub(r"omph::kernels::(\(anonymous namespace\)::)?", "", name)).replace("void ", "")
    return "other: " + re.sub(r"<.*", "", n)[:30], None


def model_stats(path: str) -> tuple[dict, dict, dict, dict]:
    """Per GGUF type: tensor count and total bytes (the GEMVs: no token_embd, no MTP block), and
    total elements (the GEMMs: the output head neither, a prefill computes it for one row)."""
    from gguf import GGUFReader

    r = GGUFReader(path)
    last = max(int(t.name.split(".")[1]) for t in r.tensors if t.name.startswith("blk."))
    cnt, nbytes, elems, sizes = defaultdict(int), defaultdict(float), defaultdict(float), defaultdict(list)
    for t in r.tensors:
        if t.name == "token_embd.weight" or t.name.startswith(f"blk.{last}.") or len(t.shape) < 2:
            continue
        typ = t.tensor_type.name
        cnt[typ] += 1
        nbytes[typ] += int(t.n_bytes)
        sizes[typ].append(int(t.n_bytes))
        if t.name != "output.weight":
            elems[typ] += int(t.n_elements)
    return cnt, nbytes, elems, sizes


def gemv_shapes(dec, sizes, steps):
    """Each GEMV type's launches by tensor: the k-th launch of a type in a step is
    always the same tensor, so the per-position mean is a tensor's time; positions
    and tensors are paired by rank (the time follows the bytes), then grouped by size."""
    per = defaultdict(list)
    for r in dec:
        cls, t = classify(r["Kernel_Name"])
        if cls == "gemv" and t in sizes:
            per[t].append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3)
    print("\n== GEMV time by tensor size (decode; bandwidth at the tensor's bytes)")
    print(f"{'type':8s} {'MiB':>7s} {'tensors':>7s} {'us':>8s} {'GB/s':>6s} {'eff':>5s} {'loss us/step':>12s}")
    for t, us in sorted(per.items(), key=lambda kv: -sum(kv[1])):
        k = len(sizes[t])
        if len(us) < k:
            continue
        us = us[len(us) % k:]  # a stray launch (e.g. the head at the prefill's end) first
        pos = sorted(sum(us[i::k]) / (len(us) // k) for i in range(k))
        groups = defaultdict(list)
        for b, u in zip(sorted(sizes[t]), pos):
            groups[b].append(u)
        for b, g in sorted(groups.items(), key=lambda kv: -kv[0]):
            m = sum(g) / len(g)
            bw = b / (m * 1e3)
            print(f"{t:8s} {b / 2**20:7.1f} {len(g):7d} {m:8.1f} {bw:6.1f} {100 * bw * 1e9 / DRAM:4.0f}% "
                  f"{len(g) * (m - b / DRAM * 1e6):12.1f}")


def scan(rows):
    tot, n, typ = defaultdict(float), defaultdict(int), {}
    busy, end = 0.0, 0
    for r in rows:
        s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
        cls, t = classify(r["Kernel_Name"])
        key = f"{cls} {t}" if t else cls
        nt = re.search(r"gemv_\w+?(?:_ntg)?_kernel<(\d+)", r["Kernel_Name"])
        if cls == "gemv" and nt and int(nt.group(1)) > 1:
            key += f" x{nt.group(1)}"  # the multi-token (verification) GEMVs
        tot[key] += (e - s) / 1e6
        n[key] += 1
        typ[key] = (cls, t)
        if e > end:
            busy += (e - max(s, end)) / 1e6
            end = e
    span = (int(rows[-1]["End_Timestamp"]) - int(rows[0]["Start_Timestamp"])) / 1e6
    return tot, n, typ, busy, span


def table(title, tot, n, typ, ceil_fn, div, unit, busy, span):
    print(f"\n== {title}: span {span / div:.2f} {unit}, GPU busy {busy / div:.2f}, idle {(span - busy) / div:.2f} "
          f"({100 * (span - busy) / span:.1f} %), kernel time {sum(tot.values()) / div:.2f}")
    print(f"{'class':34s} {'launches':>9s} {unit:>9s} {'%':>5s} {'ceiling':>9s} {'eff':>5s} {'loss':>9s}")
    rows = []
    for k, v in tot.items():
        c = ceil_fn(k, typ[k], n[k])
        rows.append((v - (c or 0.0), k, v, c))
    kern = sum(tot.values())
    for loss, k, v, c in sorted(rows, reverse=True):
        print(f"{k:34s} {n[k] / div:9.1f} {v / div:9.3f} {100 * v / kern:5.1f} "
              + (f"{c / div:9.3f} {100 * c / v:4.0f}% " if c else f"{'':9s} {'':5s} ") + f"{loss / div:9.3f}")
    print(f"{'idle (between kernels)':34s} {'':9s} {(span - busy) / div:9.3f} {'':5s} {'':9s} {'':5s} {(span - busy) / div:9.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("model", help="the GGUF (tensor sizes and shapes)")
    ap.add_argument("--prompt", type=int, required=True, help="prompt tokens")
    ap.add_argument("--ctx", type=int, default=0, help="context during the decode (default: --prompt)")
    ap.add_argument("--gen", type=int, default=0, help="generated tokens (per-token figures)")
    ap.add_argument("--k4-layers", type=int, default=8, help="attention layers with the K4 cache (mix A: 8)")
    args = ap.parse_args()
    cnt, nbytes, elems, sizes = model_stats(args.model)
    with open(args.trace) as f:
        rows = [r for r in csv.DictReader(f) if "wave_size_kernel" not in r["Kernel_Name"]]
    rows.sort(key=lambda r: int(r["Start_Timestamp"]))
    last = max(i for i, r in enumerate(rows) if re.search(r"gemm_|attention_wmma", r["Kernel_Name"]))
    # the prefill starts at its first kernel: the load before it only uploads (copies / fills)
    first = next(i for i, r in enumerate(rows) if not re.search(r"rocclr_copy|rocclr_fill", r["Kernel_Name"]))
    pre, dec = rows[first: last + 1], rows[last + 1:]

    T = args.prompt
    heads, hd, layers = 24, 256, 16
    attn_flops = 4.0 * heads * hd * layers * T * (T + 1) / 2   # QK^T and PV, causal

    def pre_ceil(k, ct, launches):
        cls, t = ct
        if cls == "gemm" and t in elems:
            return 2.0 * elems[t] * T / WMMA_F16 * 1e3
        if cls == "attention prefill":
            return attn_flops / WMMA_F16 * 1e3
        return None

    ctx = args.ctx or T
    kv_k = (layers - args.k4_layers) * 34 / 32 + args.k4_layers * 18 / 32   # bytes per element, K
    kv_bytes = ctx * 4 * hd * (kv_k + layers * 18 / 32)                       # 4 KV heads, V in Q4

    def dec_ceil(k, ct, launches):
        cls, t = ct
        if cls == "gemv" and t in nbytes:
            return launches * nbytes[t] / cnt[t] / DRAM * 1e3
        return None

    tot, n, typ, busy, span = scan(pre)
    table(f"prefill, {T} tokens (totals)", tot, n, typ, pre_ceil, 1.0, "ms", busy, span)
    if dec:
        steps = sum(1 for r in dec if "argmax" in r["Kernel_Name"]) or 1
        tot, n, typ, busy, span = scan(dec)
        att = sum(v for k, v in tot.items() if k.startswith(("attention decode", "attention merge"))) / steps
        print(f"\n(decode attention: {att:.3f} ms per step against ~{kv_bytes / DRAM * 1e3:.3f} ms of KV reads at "
              f"{ctx} tokens)")
        table(f"decode, {steps} steps (per step)", tot, n, typ, dec_ceil, steps, "ms/step", busy, span)
        if any(n[k] % steps == 0 for k in n):
            gemv_shapes(dec, sizes, steps)
        if args.gen:
            print(f"\nper generated token: {span / args.gen:.2f} ms ({span / steps:.2f} ms per step, "
                  f"{args.gen / steps:.2f} tokens per step)")
    gceil = sum(nbytes.values()) / DRAM * 1e3
    print(f"\nall main-model weights once: {sum(nbytes.values()) / 2**30:.2f} GiB -> {gceil:.2f} ms at 318.3 GB/s")


if __name__ == "__main__":
    main()
