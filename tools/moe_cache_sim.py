"""The MoE expert cache's policies replayed on traced decode picks (#550).

omphalos keeps, per MoE layer, a calibrated fixed set of experts in VRAM plus
32 LRU slots over pinned host copies of the rest; a miss is read in place from
host memory and admitted into a slot on its second miss within 8 steps (#512,
#522: Runner::moe_cache_begin). llama.cpp's cache (#542, v0.7.0) shares one LRU
among the layers of the same expert layout. This replays `omph-run --moe-trace`
files through:

    today      per layer: the fixed set + 32 LRU slots (the engine's policy)
    pool       the same fixed sets, the LRU slots shared by the layers of a layout
    pool-fix   the fixed sets chosen across the layers of a layout too (equal count)
    llama      no fixed set, one LRU over all the slots of a layout, admit on the
               first miss (every expert in host memory: llama.cpp's shape)

at the engine's capacity (CAP experts per layer, CAP - 32 fixed), and prints
the share of picks read from host memory per trace.

    uv run python moe_cache_sim.py ../models/Ornith-1.5-35B-A3B-Q4_K_M.gguf \\
        ../models/Ornith-1.5-35B-A3B-Q4_K_M.omph.experts trace1.bin trace2.bin --cap 185
"""
import argparse
import re
from collections import OrderedDict, defaultdict

import numpy as np
from gguf import GGUFReader

SLOTS = 32       # Runner::kMoeCacheSlots
ADMIT_STEPS = 8  # Runner::kMoeAdmitSteps


def read_trace(path: str) -> tuple[list[int], list[np.ndarray]]:
    """The blocks of the logged tables and per step a [tables, T * topk] array."""
    raw = np.fromfile(path, dtype=np.int32)
    tabs, topk = int(raw[0]), int(raw[1])
    blks = [int(b) for b in raw[2:2 + tabs]]
    steps = []
    i = 2 + tabs
    while i < raw.size:
        t = int(raw[i])
        n = tabs * t * topk
        steps.append(raw[i + 1:i + 1 + n].reshape(tabs, t * topk))
        i += 1 + n
    return blks, steps


def read_counts(path: str) -> dict[int, np.ndarray]:
    out = {}
    with open(path) as f:
        for line in f:
            v = [int(x) for x in line.split()]
            out[v[0]] = np.array(v[1:], dtype=np.int64)
    return out


def layouts(model: str) -> dict[int, tuple]:
    """Each MoE layer's expert layout (the types of its expert tensors): only
    layers of one layout can share slots."""
    kinds = defaultdict(dict)
    for t in GGUFReader(model).tensors:
        m = re.fullmatch(r"blk\.(\d+)\.(ffn_(?:gate|up|down)_exps)\.weight", t.name)
        if m:
            kinds[int(m[1])][m[2]] = t.tensor_type.name
    return {b: tuple(sorted(k.items())) for b, k in kinds.items()}


class Pool:
    """LRU slots shared by `members` (layer keys (blk, e)), with the engine's
    admission rule (a miss enters on its second miss within ADMIT_STEPS steps)
    or on the first miss."""

    def __init__(self, n_slots: int, initial: list, admit_first: bool):
        self.lru = OrderedDict((k, 0) for k in initial[:n_slots])  # key -> last step used
        self.n = n_slots
        self.admit_first = admit_first
        self.missed = {}

    def step(self, clock: int, picks: list, fixed: set) -> int:
        misses = 0
        chosen = set(picks)
        for k in picks:
            if k in self.lru:
                self.lru[k] = clock
                self.lru.move_to_end(k)
        for k in picks:
            if k in fixed or k in self.lru:
                continue
            misses += 1
            prev = self.missed.get(k, 0)
            self.missed[k] = clock
            if self.n == 0 or (not self.admit_first and (prev == 0 or clock - prev > ADMIT_STEPS)):
                continue
            # the least recently used slot whose expert this step did not choose
            victim = next((v for v in self.lru if v not in chosen), None)
            if victim is None:
                continue
            del self.lru[victim]
            self.lru[k] = clock
        return misses


def simulate(policy: str, steps: list, blks: list, order: dict, lay: dict, cap: int) -> tuple[float, np.ndarray]:
    fixed_n = 0 if policy == "llama" else cap - SLOTS
    groups = defaultdict(list)
    for b in blks:
        groups[lay[b]].append(b)
    fixed = set()
    pools = {}  # blk -> Pool
    if policy == "today":
        for b in blks:
            fixed |= {(b, int(e)) for e in order[b][:fixed_n]}
            pools[b] = Pool(SLOTS, [(b, int(e)) for e in order[b][fixed_n:cap]], False)
    else:
        for members in groups.values():
            if policy == "pool-fix":
                # the group's fixed count goes to its most chosen (layer, expert) pairs
                score = [(-int(cnt[e]), b, int(e)) for b in members for cnt in [counts_of[b]] for e in range(cnt.size)]
                score.sort()
                fix = [(b, e) for _, b, e in score[:fixed_n * len(members)]]
                rest = [(b, e) for _, b, e in score[fixed_n * len(members):]]
            else:
                fix = [(b, int(e)) for b in members for e in order[b][:fixed_n]]
                # the slots start with each layer's next experts, interleaved
                rest = [(b, int(order[b][fixed_n + i])) for i in range(cap - fixed_n) for b in members]
            fixed |= set(fix)
            pool = Pool((cap - fixed_n) * len(members), rest, policy == "llama")
            for b in members:
                pools[b] = pool
    per_layer = np.zeros(len(blks))
    total = misses = 0
    for clock, st in enumerate(steps, start=1):
        if policy == "today":
            for i, b in enumerate(blks):
                m = pools[b].step(clock, [(b, int(e)) for e in st[i]], fixed)
                per_layer[i] += m
                misses += m
        else:
            for members in groups.values():
                pool = pools[members[0]]
                picks = [(b, int(e)) for b in members for e in st[blks.index(b)]]
                misses += pool.step(clock, picks, fixed)
        total += st.size
    return 100.0 * misses / max(total, 1), per_layer / max(len(steps) * steps[0].shape[1], 1) * 100


counts_of: dict[int, np.ndarray] = {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="the GGUF (expert layouts)")
    ap.add_argument("counts", help="<model.omph>.experts, the calibration counts")
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--cap", type=int, default=185, help="experts per layer in VRAM (the engine prints it)")
    ap.add_argument("--policies", default="today,pool,pool-fix,llama")
    a = ap.parse_args()

    lay = layouts(a.model)
    counts_of.update(read_counts(a.counts))
    order = {b: np.argsort(-c, kind="stable") for b, c in counts_of.items()}
    pols = a.policies.split(",")
    print(f"cap {a.cap} per layer ({a.cap - SLOTS} fixed + {SLOTS} LRU today); "
          f"{len(set(lay.values()))} expert layouts")
    print("trace".ljust(28) + "".join(p.rjust(10) for p in pols) + "   (% of picks read from host)")
    for path in a.traces:
        blks, steps = read_trace(path)
        row = []
        spread = None
        for p in pols:
            rate, per_layer = simulate(p, steps, blks, order, lay, a.cap)
            row.append(rate)
            if p == "today":
                spread = per_layer
        print(path.rsplit("/", 1)[-1][:27].ljust(28) + "".join(f"{r:10.2f}" for r in row) +
              f"   ({len(steps)} steps)")
        if spread is not None:
            print(" " * 28 + f"today per layer: min {spread.min():.2f} median {np.median(spread):.2f} "
                  f"max {spread.max():.2f} %")


if __name__ == "__main__":
    main()
