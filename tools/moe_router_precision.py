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
    picks = changed_tok = tokens = 0
    worst_w = worst_logit = worst_gate = 0.0
    swaps = []  # (layer, token, probability gap at the cut of the reference)
    for il in layers:
        x = dump[f"attn_post_norm-{il}"]
        w = w_router[f"blk.{il}.ffn_gate_inp.weight"]
        ref = x @ w.T
        if f"ffn_moe_logits-{il}" in dump:  # the dump's own logits: the recomputation is llama.cpp's
            worst_logit = max(worst_logit, float(np.abs(ref - dump[f"ffn_moe_logits-{il}"].reshape(ref.shape)).max()))
        x16 = x.astype(np.float16).astype(np.float32)
        test = x16 @ w.astype(np.float16).astype(np.float32).T
        ids_r, w_r = topk(ref, k)
        ids_t, w_t = topk(test, k)
        for t in range(x.shape[0]):
            same = set(ids_r[t]) == set(ids_t[t])
            tokens += 1
            picks += k
            if not same:
                changed_tok += 1
                p = np.sort(np.exp(ref[t] - ref[t].max()))[::-1]
                p /= p.sum()
                swaps.append((il, t, float(p[k - 1] - p[k])))
            else:
                o_r = np.argsort(ids_r[t])
                o_t = np.argsort(ids_t[t])
                worst_w = max(worst_w, float(np.abs(w_r[t][o_r] - w_t[t][o_t]).max()))
        wg = w_router.get(f"blk.{il}.ffn_gate_inp_shexp.weight")
        if wg is not None:
            g_r = 1 / (1 + np.exp(-(x @ wg.reshape(-1))))
            g_t = 1 / (1 + np.exp(-(x16 @ wg.reshape(-1))))
            worst_gate = max(worst_gate, float(np.abs(g_r - g_t).max()))

    print(f"{len(layers)} layers x {tokens // max(len(layers), 1)} tokens = {tokens} selections of {k}")
    print(f"dumped vs recomputed f32 logits: max |diff| {worst_logit:.3g}")
    print(f"selections that differ (f16 vs f32): {changed_tok} / {tokens} ({100 * changed_tok / max(tokens, 1):.3f} %)")
    for il, t, gap in swaps[:20]:
        print(f"  layer {il} token {t}: the reference's k-th / (k+1)-th probability gap {gap:.2e}")
    print(f"same selection: max |weight diff| {worst_w:.3g}")
    print(f"shared expert gate: max |sigmoid diff| {worst_gate:.3g}")


if __name__ == "__main__":
    main()
