#!/usr/bin/env python3
"""#452: what a DFlash2 draft tree would have been worth, from the probe.

`OMPH_DFLASH_PROBE=1 omph-run ... --dflash <drafter>` prints one `dfl_probe` line
per verified block whose chain was rejected at some position:

    dfl_probe,<step>,<pos0>,<kept>,<drafts_accepted>,<accepted>,<next>,<i>:<second>:<margin> ...

`kept` drafts of the chain were accepted, so the first rejected position is the
one at list index `kept`; the target's own token there is `<next>`, and a tree
would have offered `<second>` (the selector's best successor next to the chain's
pick) as its sibling. This turns those lines into

    P(the target picked the second candidate | the chain's first was rejected)

per position, plus the margin (selector score of the first minus the second) at
the rejections, which is what a margin-driven tree shape would key on. With
`OMPH_DFLASH_DEBUG=1` in the same run, the per-position reached/kept counters are
reported too (the denominator: how often the chain even got that far).

    uv run python dflash_tree_probe.py run.log [run2.log ...]

Lines must come from a run *without* sampling (greedy): under sampling the
"second candidate" is not what a tree would compare against.
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

PROBE = re.compile(r"^dfl_probe,(\d+),(\d+),(\d+),(-?\d+),(\d+),(-?\d+),(.*)$")
KEEP = re.compile(r"per-position reached/kept:(.*)")


def parse(path: Path) -> tuple[list[dict], dict[int, tuple[int, int]]]:
    probes: list[dict] = []
    keep: dict[int, tuple[int, int]] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = PROBE.match(line.strip())
        if m is not None:
            step, pos0, kept, _da, accepted, nxt, rest = m.groups()
            cands = {}
            for field in rest.split():
                i, second, margin = field.split(":")
                cands[int(i)] = (int(second), float(margin))
            probes.append({"step": int(step), "pos0": int(pos0), "kept": int(kept),
                           "accepted": int(accepted), "next": int(nxt), "cands": cands})
            continue
        k = KEEP.search(line)
        if k is not None:
            for field in k.group(1).split():
                n, frac = field.split(":")
                kept_n, reached = (int(x) for x in frac.split("/"))
                keep[int(n)] = (kept_n, reached)
    return probes, keep


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    probes: list[dict] = []
    keep: dict[int, tuple[int, int]] = {}
    for p in sys.argv[1:]:
        pr, kp = parse(Path(p))
        probes.extend(pr)
        for n, (a, b) in kp.items():
            ka, kb = keep.get(n, (0, 0))
            keep[n] = (ka + a, kb + b)
    if not probes:
        sys.exit("no dfl_probe lines: run with OMPH_DFLASH_PROBE=1 and a --dflash decode")
    print(f"{len(probes)} rejected blocks in {len(sys.argv) - 1} run(s)")
    print(f"\n{'position':>8} {'rejected':>8} {'#2 taken':>8} {'P(#2)':>7} {'margin median':>13} "
          f"{'margin p90':>10} {'chain reached (kept/all)':>24}")
    total_hit = total = 0
    for i in range(20):
        rows = [p for p in probes if p["kept"] == i and i in p["cands"]]
        if not rows:
            continue
        hits = sum(1 for p in rows if p["cands"][i][0] == p["next"])
        margins = sorted(p["cands"][i][1] for p in rows)
        med = statistics.median(margins)
        p90 = margins[min(len(margins) - 1, int(0.9 * len(margins)))]
        total_hit += hits
        total += len(rows)
        chain = f"{keep.get(i + 1, ('?', '?'))[0]}/{keep.get(i + 1, ('?', '?'))[1]}" if keep else "-"
        print(f"{i + 1:>8} {len(rows):>8} {hits:>8} {hits / len(rows):>7.3f} {med:>13.3f} {p90:>10.3f} "
              f"{chain:>24}")
    print(f"\noverall P(target = candidate #2 | chain rejected) = {total_hit}/{total} = "
          f"{total_hit / max(total, 1):.3f}")
    print("a tree pays only where that is well above the cost of verifying the extra row")


if __name__ == "__main__":
    main()
