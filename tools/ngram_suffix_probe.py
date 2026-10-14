"""Would a longest-match (suffix automaton) n-gram beat the current n <= 8 hash, offline (#199 follow-up)?

The engine's n-gram drafts (#199) look up the longest n-gram in [OMPH_NGRAM_MIN, 8] ending
at the previous token whose latest earlier occurrence is known, and propose the up-to-15
tokens that followed it. The suffix-automaton policy proposes from the *longest repeated
suffix* instead - unbounded n - and so anchors a repeat on as much text as actually
repeated, not on at most 8 tokens.

`ngram_coverage.py` measured the hash: 0.28-0.36x the MTP steps on edits and quotes (84-92 %
of tokens copied), and the K16 columns halved the steps of K8 - the chains are *capped* by
the verification budget, so a longer anchor is the one thing left that could help. Its
"verification cost" caveat is obsolete: every weight type now verifies 16 tokens in one
pass (#178, #244, #253, #258, #275), so a K16 step costs a K4 step.

    uv run python ngram_suffix_probe.py [--out ../bench/results/ngram-199] [--k 15] [--min-prop 4]

Reads the recorded scenarios (prompt + real output, no server needed) and prints, per
scenario, the share of output tokens each policy supplies and its verification steps as a
ratio to MTP alone - the same columns and the same acceptance rule as `ngram_coverage.py`,
so the rows are comparable.
"""

import argparse
import json
from pathlib import Path

from ngram_coverage import simulate  # the hash policy, unchanged

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


class Sam:
    """Incremental suffix automaton over ids, with the first end position of each state's
    class. `longest_repeat()` returns the longest suffix of the text so far that occurs at
    least twice, and the end of its first occurrence: a state's endpos set contains the
    current end of text plus an earlier one exactly when it is reachable by suffix links
    from the last state."""

    def __init__(self) -> None:
        self.len = [0]
        self.link = [-1]
        self.next: list[dict[int, int]] = [{}]
        self.first = [-1]  # end position of the first occurrence
        self.last = 0

    def extend(self, c: int, pos: int) -> None:
        cur = len(self.len)
        self.len.append(self.len[self.last] + 1)
        self.link.append(-1)
        self.next.append({})
        self.first.append(pos)
        p = self.last
        while p >= 0 and c not in self.next[p]:
            self.next[p][c] = cur
            p = self.link[p]
        if p < 0:
            self.link[cur] = 0
        else:
            q = self.next[p][c]
            if self.len[p] + 1 == self.len[q]:
                self.link[cur] = q
            else:
                clone = len(self.len)
                self.len.append(self.len[p] + 1)
                self.link.append(self.link[q])
                self.next.append(dict(self.next[q]))
                self.first.append(self.first[q])  # the class's earliest end, kept by the clone
                while p >= 0 and self.next[p].get(c) == q:
                    self.next[p][c] = clone
                    p = self.link[p]
                self.link[q] = clone
                self.link[cur] = clone
        self.last = cur

    def longest_repeat(self) -> tuple[int, int] | None:
        """(length, end of the first occurrence) of the longest repeated suffix, or None."""
        v = self.link[self.last]
        if v <= 0:  # the root: nothing repeated
            return None
        return self.len[v], self.first[v]


def simulate_sam(prompt: list[int], output: list[int], k: int, min_prop: int, mtp_tps: float) -> dict:
    """The same walk as ngram_coverage.simulate, with the longest-match proposal."""
    ids = prompt + output
    sam = Sam()
    for pos in range(len(prompt)):
        sam.extend(ids[pos], pos)
    i = len(prompt)
    ng_tokens = ng_steps = mtp_steps = 0.0
    while i < len(ids):
        if i > len(prompt):  # learn through i - 1 only: a match must end at the previous token
            sam.extend(ids[i - 1], i - 1)
        rep = sam.longest_repeat()
        prop: list[int] = []
        if rep is not None:
            n, end = rep
            if n >= min_prop:  # a 1-token anchor is coincidence, not a repeat
                p = end + 1  # the token right after the first occurrence
                prop = ids[p:min(p + k, i)]
        if len(prop) >= min_prop:
            a = 0
            while a < len(prop) and i + a < len(ids) and ids[i + a] == prop[a]:
                a += 1
            ng_tokens += a
            ng_steps += 1
            i += a + 1
        else:
            mtp_steps += 1 / mtp_tps
            i += 1
    n_out = len(output)
    return {"ngram_share": ng_tokens / n_out, "ngram_steps": ng_steps, "steps": ng_steps + mtp_steps,
            "mtp_only": n_out / mtp_tps}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "bench/results/ngram-199"))
    ap.add_argument("--k", type=int, default=15, help="draft budget (the engine's 15)")
    ap.add_argument("--min-prop", type=int, default=4, help="OMPH_NGRAM_MIN")
    ap.add_argument("--n-max", type=int, default=8, help="the hash policy's longest n-gram (OMPH_NGRAM)")
    ap.add_argument("--mtp-tps", type=float, default=3.4)
    args = ap.parse_args()

    rows = []
    for f in sorted(Path(args.out).glob("*.json")):
        d = json.loads(f.read_text())
        tps = args.mtp_tps
        if d.get("draft_n"):
            tps = d["predicted_n"] / (d["draft_n"] / 3)
        h = simulate(d["prompt"], d["output"], args.k, 4, args.n_max, args.min_prop, tps)
        s = simulate_sam(d["prompt"], d["output"], args.k, args.min_prop, tps)
        rows.append((f.stem, len(d["output"]), h, s))

    print(f"K{args.k}, min_prop {args.min_prop}, hash n_max {args.n_max}; steps as a ratio to MTP alone (lower is better)")
    print(f"{'scenario':16} {'tokens':>6} {'hash share':>10} {'hash x':>7} | {'SAM share':>9} {'SAM x':>6} | {'d share':>7} {'d x':>6}")
    for name, n, h, s in rows:
        print(f"{name:16} {n:6} {h['ngram_share']:9.0%} {h['steps'] / h['mtp_only']:7.2f} | "
              f"{s['ngram_share']:8.0%} {s['steps'] / s['mtp_only']:6.2f} | "
              f"{s['ngram_share'] - h['ngram_share']:+6.0%} {s['steps'] / h['steps'] - 1:+6.1%}")
    mh = sum(r[2]["steps"] for r in rows)
    ms = sum(r[3]["steps"] for r in rows)
    th = sum(r[2]["ngram_share"] * r[1] for r in rows)
    ts = sum(r[3]["ngram_share"] * r[1] for r in rows)
    tot = sum(r[1] for r in rows)
    print(f"\ntotal ({tot} tokens): hash {mh:.0f} steps, SAM {ms:.0f} steps ({ms / mh - 1:+.1%}); "
          f"tokens supplied {th / tot:.0%} vs {ts / tot:.0%}")
    wins = [r[0] for r in rows if r[3]["steps"] < r[2]["steps"] * 0.98]
    losses = [r[0] for r in rows if r[3]["steps"] > r[2]["steps"] * 1.02]
    print(f"clear wins: {', '.join(wins) or 'none'}; clear losses: {', '.join(losses) or 'none'}")


if __name__ == "__main__":
    main()
