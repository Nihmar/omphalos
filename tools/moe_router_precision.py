"""How often the f16 router changes the MoE expert selection (#544).

omphalos computes the router logits from f16 copies of ffn_gate_inp and of the
FFN input (moe_router / gemm_f16, #494, #509) and the shared expert's gate from
the f16 input (moe_finish, #522); llama.cpp computes both in f32. This takes
llama.cpp's own router inputs (attn_post_norm-N) from a golden dump and
compares, per layer and token, the top-k selection and its renormalized
weights with f32 and with f16-rounded operands.

    tools/native/dump_tensors -m <model.gguf> -p "$(cat prompt.txt)" -o <dir> -c 1024 \\
        --only attn_post_norm,ffn_moe_logits
    uv run python moe_router_precision.py <model.gguf> <dir>
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from gguf import GGUFReader


def load_dump(d: Path) -> dict[str, np.ndarray]:
    """The router inputs and logits as [rows, ne0]; a prompt longer than one
    ubatch dumps a name once per ubatch, in order, so the rows are concatenated."""
    parts: dict[str, list[np.ndarray]] = {}
    for line in (d / "index.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["type"] != "f32" or not re.fullmatch(r"(attn_post_norm|ffn_moe_logits)-\d+", e["name"]):
            continue
        a = np.fromfile(d / e["file"], dtype=np.float32).reshape(-1, e["ne"][0])
        parts.setdefault(e["name"], []).append(a)
    return {n: np.concatenate(p) for n, p in parts.items()}


def topk(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The selection as moe_topk makes it: softmax over all, the k largest (ties
    to the lower id), renormalized to sum 1. Rows are tokens."""
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    ids = np.argsort(-p, axis=1, kind="stable")[:, :k]
    w = np.take_along_axis(p, ids, axis=1)
    return ids, w / w.sum(axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("dump")
    a = ap.parse_args()

    r = GGUFReader(a.model)
    k = int(r.fields["qwen35moe.expert_used_count"].contents())
    w_router = {t.name: np.asarray(t.data, dtype=np.float32) for t in r.tensors
                if t.name.endswith(".ffn_gate_inp.weight") or t.name.endswith(".ffn_gate_inp_shexp.weight")}
    dump = load_dump(Path(a.dump))

    layers = sorted(int(n.rsplit("-", 1)[1]) for n in dump if n.startswith("attn_post_norm-"))
    # each variant's selections against the exact (f64) router: llama.cpp's own
    # f32 logits (the dump) are the control, the f16 operands what omphalos did
    # before #547, f16 weights with the input as two f16 halves (hi + lo) after
    names = ("llama.cpp f32 (dumped)", "f16 weights, f16 input", "f16 weights, hi+lo input")
    flips = [0, 0, 0]
    flip_w = [[], [], []]  # the exact weight of the expert a flip dropped
    between = tokens = 0
    worst_w = worst_gate = 0.0
    for il in layers:
        x = dump[f"attn_post_norm-{il}"]
        w = w_router[f"blk.{il}.ffn_gate_inp.weight"]
        exact = x.astype(np.float64) @ w.astype(np.float64).T
        w16 = w.astype(np.float16).astype(np.float32)
        hi = x.astype(np.float16).astype(np.float32)
        lo = (x - hi).astype(np.float16).astype(np.float32)
        ids_e, w_e = topk(exact, k)
        sel = [topk(dump[f"ffn_moe_logits-{il}"].reshape(exact.shape), k), topk(hi @ w16.T, k),
               topk(hi @ w16.T + lo @ w16.T, k)]
        for t in range(x.shape[0]):
            tokens += 1
            ref = set(ids_e[t])
            for v in range(3):
                got = set(sel[v][0][t])
                if got != ref:
                    flips[v] += 1
                    dropped = list(ref - got)
                    flip_w[v] += [float(w_e[t][list(ids_e[t]).index(e)]) for e in dropped]
            if set(sel[0][0][t]) != set(sel[1][0][t]):
                between += 1
            elif set(sel[1][0][t]) == ref:
                o_e = np.argsort(ids_e[t])
                o_t = np.argsort(sel[1][0][t])
                worst_w = max(worst_w, float(np.abs(w_e[t][o_e] - sel[1][1][t][o_t]).max()))
        wg = w_router.get(f"blk.{il}.ffn_gate_inp_shexp.weight")
        if wg is not None:
            g_e = 1 / (1 + np.exp(-(x.astype(np.float64) @ wg.reshape(-1).astype(np.float64))))
            g_t = 1 / (1 + np.exp(-(x.astype(np.float16).astype(np.float32) @ wg.reshape(-1))))
            worst_gate = max(worst_gate, float(np.abs(g_e - g_t).max()))

    print(f"{len(layers)} layers x {tokens // max(len(layers), 1)} tokens = {tokens} selections of {k}")
    for v in range(3):
        fw = flip_w[v]
        print(f"{names[v]:>26}: {flips[v]} selections differ from the exact one "
              f"({100 * flips[v] / max(tokens, 1):.3f} %), dropped expert weight "
              f"median {np.median(fw) if fw else 0:.4f} max {max(fw) if fw else 0:.4f}")
    print(f"f16 input vs llama.cpp f32: {between} selections differ ({100 * between / max(tokens, 1):.3f} %)")
    print(f"same selection: max |weight diff| (f16 vs exact) {worst_w:.3g}")
    print(f"shared expert gate: max |sigmoid diff| (f16 input vs exact) {worst_gate:.3g}")

if __name__ == "__main__":
    main()
