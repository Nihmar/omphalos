"""Images in omphalos vs llama.cpp (#160).

For one image and a question, omphalos generates a greedy answer; then both
engines are teacher-forced through that answer and the logits of every
position are compared (KL(llama.cpp || omphalos), top-1 agreement). The same
question without the image is the text-only baseline. Two ablations of the
image positions (OMPH_TEST_MROPE=swap: h and w exchanged; flat: sequential
1D positions) must score clearly worse than the real layout, which shows the
comparison sees the M-RoPE positions.

    uv run python check_vision.py --image <file.jpg> \
        --dumper native/dump_mtmd_logits   # tools/native/build.sh <llama.cpp-dir>

The reference runs llama.cpp's text model on the GPU (f16 KV); both encode
the image on the CPU with mtmd.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from compare_logits import N_VOCAB, log_softmax

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMAGE_PAD = 248056
FAILURES: list[str] = []


def check(cond: bool, what: str) -> None:
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILURES.append(what)


def run(cmd: list[str], env: dict | None = None, stdin: str | None = None) -> str:
    r = subprocess.run(cmd, input=stdin, capture_output=True, text=True, check=False,
                       env={**os.environ, **(env or {})})
    if r.returncode != 0:
        sys.exit(f"{cmd[0]} failed:\n{r.stderr[-2000:]}")
    return r.stdout


def kl_stats(ref: Path, test: Path) -> tuple[float, float, float]:
    a = np.fromfile(ref, dtype=np.float32).reshape(-1, N_VOCAB)
    b = np.fromfile(test, dtype=np.float32).reshape(-1, N_VOCAB)
    n = min(len(a), len(b))
    la, lb = log_softmax(a[:n]), log_softmax(b[:n])
    kl = (np.exp(la) * (la - lb)).sum(axis=-1)
    top1 = float((a[:n].argmax(-1) == b[:n].argmax(-1)).mean())
    return float(kl.mean()), float(np.percentile(kl, 99)), top1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--mmproj", default=str(ROOT / "models/mmproj-Qwen3.8-27B-BF16.gguf"))
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", default="Describe this image in two sentences.")
    ap.add_argument("--dumper", default=str(HERE / "native/dump_mtmd_logits"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    args = ap.parse_args()

    gen, tokenize = f"{args.omph}/omph-generate", f"{args.omph}/omph-tokenize"
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        results = {}
        for kind in ("text", "image"):
            content = ([{"type": "image"}, {"type": "text", "text": args.question}] if kind == "image"
                       else args.question)
            req = json.dumps({"messages": [{"role": "user", "content": content}],
                              "add_generation_prompt": True, "enable_thinking": False})
            img = ["--mmproj", args.mmproj, "--image", args.image] if kind == "image" else []
            ids = run([gen, args.model, "--chat", "--max", "96", "--ids", *img], stdin=req).split("\n")[0]
            (t / "force.ids").write_text(ids)
            # the same prompt for llama.cpp: mtmd adds the vision tags itself
            marker = "<__media__>" if kind == "image" else ""
            (t / "prompt.txt").write_text(f"<|im_start|>user\n{marker}{args.question}<|im_end|>\n"
                                          "<|im_start|>assistant\n<think>\n\n</think>\n\n")
            run([args.dumper, args.model, args.mmproj, str(t / "prompt.txt"), str(t / "ref.f32"),
                 str(t / "ref.tok"), *([args.image] if kind == "image" else [])],
                env={"DUMP_TAIL": str(t / "force.ids")})
            ours = [int(x) for x in run([tokenize, args.model, "--chat-ids"], stdin=req).split()]
            ref_tok = [int(x) for x in (t / "ref.tok").read_text().split()]
            n_img = ref_tok.count(-1)
            expanded = [y for x in ours for y in ([-1] * n_img if x == IMAGE_PAD else [x])]
            check(expanded == ref_tok, f"{kind}: the same {len(ref_tok)} prompt tokens as llama.cpp")
            variants = {"": {}, " f32 KV": {"OMPH_KV_F32": "1"}}
            if kind == "image":
                variants |= {" swap": {"OMPH_TEST_MROPE": "swap"}, " flat": {"OMPH_TEST_MROPE": "flat"}}
            for name, env in variants.items():
                out = t / "ours.f32"
                run([gen, args.model, "--chat", "--force", str(t / "force.ids"), "--logits-out", str(out), *img],
                    env=env, stdin=req)
                results[kind + name] = kl_stats(t / "ref.f32", out)
                kl, p99, top1 = results[kind + name]
                print(f"     {kind + name:12s} KL mean {kl:.6f}  p99 {p99:.6f}  top-1 {100 * top1:.2f} % "
                      f"({len(ids.split())} positions)")
    base, img = results["text"][0], results["image"][0]
    check(img < 0.005, f"image KL {img:.6f} within the engine's quantization noise (text {base:.6f})")
    for ab in ("swap", "flat"):
        check(results[f"image {ab}"][0] > 3 * img, f"the {ab} ablation is clearly worse ({results[f'image {ab}'][0]:.6f})")
    if FAILURES:
        sys.exit(f"{len(FAILURES)} check(s) failed")
    print("vision checks passed")


if __name__ == "__main__":
    main()
