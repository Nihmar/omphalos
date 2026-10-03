"""How much would n-gram (prompt lookup) drafts add to MTP? Offline, on real outputs (#199).

1. Generate: each scenario's chat request goes through omph-server (greedy, MTP on); the
   prompt ids, the output text re-tokenized and the server's MTP counts are saved to
   --out/<scenario>.json (kept on reruns). The #229 pelican outputs join as they are.
2. Simulate: walk each output as a speculative decoder would. At a step that predicts
   position i, the longest n-gram (n_min .. n_max tokens) ending at i - 1 that occurred
   earlier in prompt + output proposes the tokens that followed its latest occurrence (up to
   K, only tokens already known); the step accepts their prefix that matches the real
   continuation, plus one token. Steps without a proposal are MTP steps worth the scenario's
   measured MTP tokens per step (server counts).

    uv run python ngram_coverage.py --out ../bench/results/ngram-199

Reports per scenario: the share of output tokens an n-gram draft would supply, its steps
against MTP alone, and the step count of the combined policy (n-gram when it proposes at
least --min-prop tokens, MTP otherwise) as a ratio to MTP alone. Steps, not time: a 16-token
verification costs more than a 4-token one (#170), measured separately.
"""

import argparse
import gzip
import json
import subprocess
from pathlib import Path

from niah import free_port, post, wait_health
from omph_model import omph_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EDIT = ("Here is a file of a project:\n\n```{lang}\n{src}```\n\n{task} Reply with the complete updated file in "
        "one code block, nothing else.")
HE = ("Complete the following Python function. Reply with the complete function (with any imports it needs) "
      "in a single ```python code block.\n\n```python\n{}```")


def scenarios() -> dict[str, dict]:
    py = (HERE / "pelican.py").read_text()
    cc = (ROOT / "engine/src/runtime/options.cc").read_text()
    wiki = (ROOT / "models/datasets/wikitext-2-raw/wiki.test.raw").read_text()[20000:27000]
    s = {
        "edit-py": {"msg": EDIT.format(lang="python", src=py, task="Add a --runs N option that repeats every "
                                       "configuration N times and reports the median of each speed."),
                    "think": False, "max": 8000},
        "edit-py-think": {"msg": EDIT.format(lang="python", src=py, task="Add a --runs N option that repeats "
                                             "every configuration N times and reports the median of each speed."),
                          "think": True, "max": 12000},
        "edit-cc": {"msg": EDIT.format(lang="cpp", src=cc, task="Add an integer option OMPH_GDN_CHUNK (default "
                                       "64), parsed like OMPH_STAGE_MIB, in the struct's field order."),
                    "think": False, "max": 6000},
        "quote": {"msg": "Summarize the following text in six bullet points, each quoting its key sentence "
                         "verbatim.\n\n" + wiki, "think": False, "max": 2000},
        "prose": {"msg": "Write a 700-word short story about a lighthouse keeper who finds a message in a bottle.",
                  "think": False, "max": 2000},
    }
    with gzip.open(ROOT / "models/datasets/human-eval/data/HumanEval.jsonl.gz", "rt") as f:
        problems = [json.loads(line) for line in f]
    for p in problems[::16]:  # 11 problems across the set, thinking on
        s["he-" + p["task_id"].split("/")[1]] = {"msg": HE.format(p["prompt"]), "think": True, "max": 4096}
    return s


def tokenize(tool: str, model: str, text: str) -> list[int]:
    out = subprocess.run([tool, model, "--no-parse-special"], input=text, capture_output=True, text=True,
                         check=True).stdout
    return [int(x) for x in out.split()]


def generate(args, out: Path) -> None:
    tok = f"{args.omph}/omph-tokenize"
    todo = {k: v for k, v in scenarios().items() if not (out / f"{k}.json").exists()}
    if todo:
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        proc = subprocess.Popen([f"{args.omph}/omph-server", omph_file(args.model), "--port", str(port), "--ctx",
                                 "32768", "--cache-ram", "0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_health(url, proc)
            for name, sc in todo.items():
                req = {"messages": [{"role": "user", "content": sc["msg"]}], "add_generation_prompt": True,
                       "enable_thinking": sc["think"]}
                ids = [int(x) for x in subprocess.run([tok, args.model, "--chat-ids"], input=json.dumps(req),
                                                      capture_output=True, text=True, check=True).stdout.split()]
                r = post(url + "/v1/completions", {"prompt": ids, "max_tokens": sc["max"], "temperature": 0},
                         timeout=7200)
                text, tm = r["choices"][0]["text"], r["timings"]
                row = {"prompt": ids, "output": tokenize(tok, args.model, text), "predicted_n": tm["predicted_n"],
                       "draft_n": tm["draft_n"], "draft_accepted": tm["draft_n_accepted"]}
                (out / f"{name}.json").write_text(json.dumps(row))
                print(f"{name}: {tm['predicted_n']} tokens, MTP {tm['draft_n_accepted']} / {tm['draft_n']}",
                      flush=True)
        finally:
            proc.terminate()
            proc.wait()
    for f in sorted((ROOT / "bench/results/pelican-229").glob("omphalos-mtp-t*.json")):  # #229, MTP counts unknown
        name = "pelican-" + f.stem.split("-")[-1]
        if not (out / f"{name}.json").exists():
            req = {"messages": [{"role": "user", "content": "Write SVG code that displays a 2D animation of a "
                                 "pelican riding a bicycle. No additional testing is required."}],
                   "add_generation_prompt": True}
            ids = [int(x) for x in subprocess.run([tok, args.model, "--chat-ids"], input=json.dumps(req),
                                                  capture_output=True, text=True, check=True).stdout.split()]
            text = f.with_suffix(".txt").read_text()
            (out / f"{name}.json").write_text(json.dumps({"prompt": ids, "output": tokenize(tok, args.model, text)}))


def simulate(prompt: list[int], output: list[int], k: int, n_min: int, n_max: int, min_prop: int,
             mtp_tps: float) -> dict:
    ids = prompt + output
    last: dict[tuple, int] = {}  # n-gram -> the index right after its latest occurrence
    known = 0

    def learn(upto: int) -> None:  # every n-gram ending before index upto
        nonlocal known
        for end in range(max(known, n_min), upto + 1):
            for n in range(n_min, n_max + 1):
                if end - n >= 0:
                    last[tuple(ids[end - n:end])] = end
        known = upto + 1

    i = len(prompt)
    ng_tokens = ng_steps = mtp_steps = 0.0
    while i < len(ids):
        learn(i - 1)  # n-grams whose continuation is already known
        prop = []
        for n in range(n_max, n_min - 1, -1):
            p = last.get(tuple(ids[i - n:i]))
            if p is not None:
                prop = ids[p:min(p + k, i)]
                break
        if len(prop) >= min_prop:
            a = 0
            while a < len(prop) and i + a < len(ids) and ids[i + a] == prop[a]:
                a += 1
            ng_tokens += a
            ng_steps += 1
            i += a + 1
        else:
            mtp_steps += 1 / mtp_tps  # an MTP step covers mtp_tps tokens on average
            i += 1
    n = len(output)
    return {"ngram_share": ng_tokens / n, "ngram_steps": ng_steps, "steps": ng_steps + mtp_steps,
            "mtp_only": n / mtp_tps}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    ap.add_argument("--k", default="8,16")
    ap.add_argument("--n-min", default="2,3,4")
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--min-prop", type=int, default=4)
    ap.add_argument("--mtp-tps", type=float, default=3.4, help="MTP tokens per step where the server's are unknown")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generate(args, out)
    print(f"{'scenario':16} {'tokens':>6} {'MTP t/step':>10}  " +
          "  ".join(f"K{k}/n{n}: share steps-vs-MTP" for k in args.k.split(",") for n in args.n_min.split(",")))
    for f in sorted(out.glob("*.json")):
        d = json.loads(f.read_text())
        tps = args.mtp_tps
        if d.get("draft_n"):
            tps = d["predicted_n"] / (d["draft_n"] / 3)  # 3 drafts per step
        cells = []
        for k in [int(x) for x in args.k.split(",")]:
            for n_min in [int(x) for x in args.n_min.split(",")]:
                r = simulate(d["prompt"], d["output"], k, n_min, args.n_max, args.min_prop, tps)
                cells.append(f"{r['ngram_share']:5.0%} {r['steps'] / r['mtp_only']:5.2f}x")
        print(f"{f.stem:16} {len(d['output']):6d} {tps:10.2f}  " + "  ".join(f"{c:>26}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
