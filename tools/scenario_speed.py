"""The #199 scenarios through omph-server, per decoding configuration (#262).

The n-gram drafts (#199) are on by default, so every DFlash2 / MTP measurement
since PR #252 scores drafter + n-gram together. This runs the same scenario
prompts (the #199 set, as bench/results/dflash-head-prefix-245.txt did) against
one server configuration at a time -- spec plain / mtp / dflash, OMPH_NGRAM on
or off -- and records the decode rate and the drafts accepted, so the two
effects can be separated.

    uv run python scenario_speed.py --out ../bench/results/scenario-speed-262.jsonl \\
        --specs mtp,dflash --ngram 0,1 --kv k4q4

Each JSONL row is one scenario run; the console prints the t/s per scenario.
The server is restarted per (spec, ngram) cell; a long scenario's request can
take minutes (thinking on for the he-* prompts).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ngram_coverage import scenarios
from niah import free_port, post, wait_health
from omph_model import omph_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# the seven scenarios of the #245 table, so the cells line up with it
DEFAULT_SCENARIOS = "quote,he-80,he-0,prose,edit-cc,he-144,edit-py"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--specs", default="mtp,dflash", help="plain, mtp, dflash")
    ap.add_argument("--ngram", default="0,1", help="OMPH_NGRAM values to run")
    ap.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    ap.add_argument("--kv", default="k4q4", choices=["k8q4", "k4q4"])
    ap.add_argument("--temp", type=float, default=0.0, help="0: greedy (the identity witness)")
    ap.add_argument("--seed", type=int, default=262, help="sent with temp > 0 (reproducible sampling)")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
    ap.add_argument("--omph", default=str(ROOT / "engine/build"))
    ap.add_argument("--drafter", default=str(ROOT / "models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"),
                    help="the DFlash2 drafter GGUF (omphalos reads the .omph next to it)")
    args = ap.parse_args()

    wanted = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    sets = scenarios()
    missing = [s for s in wanted if s not in sets]
    if missing:
        sys.exit(f"unknown scenario(s) {missing}; the set is {sorted(sets)}")
    tok = f"{args.omph}/omph-tokenize"
    for spec in args.specs.split(","):
        for ngram in args.ngram.split(","):
            port = free_port()
            url = f"http://127.0.0.1:{port}"
            env = {**os.environ, **({"OMPH_KV_K4": "1"} if args.kv == "k4q4" else {}),
                   "OMPH_NGRAM": ngram, "OMPH_NGRAM_MIN": "4"}
            cmd = [f"{args.omph}/omph-server", omph_file(args.model), "--port", str(port), "--ctx", str(args.ctx),
                   "--cache-ram", "0"]
            if spec == "plain":
                cmd += ["--no-mtp"]
            elif spec == "dflash":
                cmd += ["--dflash", omph_file(args.drafter)]
            log = open(f"{args.out}.{spec}-ngram{ngram}.server.log", "w")  # noqa: SIM115
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_health(url, proc)
                for name in wanted:
                    sc = sets[name]
                    req = {"messages": [{"role": "user", "content": sc["msg"]}], "add_generation_prompt": True,
                           "enable_thinking": sc["think"]}
                    ids = [int(x) for x in subprocess.run([tok, args.model, "--chat-ids"],
                                                          input=json.dumps(req), capture_output=True, text=True,
                                                          check=True).stdout.split()]
                    t0 = time.time()
                    body = {"prompt": ids, "max_tokens": sc["max"], "temperature": args.temp}
                    if args.temp > 0:
                        body |= {"seed": args.seed, "top_k": 20, "top_p": 0.95}
                    r = post(url + "/v1/completions", body, timeout=7200)
                    tm = r["timings"]
                    text = r["choices"][0].get("text") or ""
                    row = {"scenario": name, "spec": spec, "ngram": int(ngram), "kv": args.kv,
                           "prompt_n": tm["prompt_n"], "prompt_ms": round(tm["prompt_ms"], 1),
                           "predicted_n": tm["predicted_n"], "predicted_ms": round(tm["predicted_ms"], 1),
                           "decode_tps": round(1000.0 * tm["predicted_n"] / tm["predicted_ms"], 1)
                           if tm["predicted_ms"] > 0 else 0.0,
                           "drafts": tm.get("draft_n", 0), "accepted": tm.get("draft_n_accepted", 0),
                           "temp": args.temp,
                           # #262: at temperature 0 every configuration must produce
                           # plain greedy's tokens; the hash is the witness.
                           "text_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
                           "wall_s": round(time.time() - t0, 1)}
                    with open(args.out, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"{spec:6s} ngram={ngram} {name:12s}: {row['predicted_n']:5d} tok "
                          f"{row['decode_tps']:6.1f} t/s, drafts {row['accepted']} / {row['drafts']}"
                          f" ({row['wall_s']} s)", flush=True)
            finally:
                proc.terminate()
                proc.wait(60)
                log.close()


if __name__ == "__main__":
    main()
