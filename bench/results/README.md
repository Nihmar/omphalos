# bench/results

One file per measurement, named after the issue that asked for it. PLAN.md §17
is the methodology: fixed conditions, warm-up discarded, median of ≥5 runs.

Every result must name both commits:

- the **engine** commit the binaries were built from (a `commit` column in the
  CSVs, a `# engine:` line in the `.txt`/`.log` files), and
- the **llama.cpp reference** it was compared against (`llama_commit`, or the
  full `llama.cpp: <tag> @ <hash>` line), read from `bench/llama.cpp.pin`
  (`tools/llama_pin.py`, #393).

## Files written before the pin (#393)

Files from before the pin that carry no `llama_commit` were measured against
whatever llama.cpp checkout was on the machine — usually the RX 9060 XT fork's
`build-hip`. A few name a commit (e.g. `m0-kl-baseline.txt`: `build-hip @
6c7a87f7e`); the rest are **pre-pin, unknown commit**. They are not comparable
with post-pin numbers, and a re-run that matters should be repeated at the pin
and written with its `llama_commit` (docs/llama-refresh.md).

The reference itself: `docs/llama-refresh.md`; the pin: `bench/llama.cpp.pin`;
the refresh checklist and the scheduled reminder:
`.github/ISSUE_TEMPLATE/llama-refresh.md`, `.github/workflows/llama-pin.yml`.
