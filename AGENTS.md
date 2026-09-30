# AGENTS.md

Working instructions for AI coding agents in **omphalos**. These rules are mandatory.
[PLAN.md](./PLAN.md) is the source of truth for the design and its rationale; this file is the source of truth for *how work is done* in this repository.

## What this project is

**omphalos** is a from-scratch, highly optimized inference engine for a single model on a single GPU: `Qwen3.8-27B` (GSQ-RCO GGUF) on an AMD Radeon RX 9060 XT 16 GB (RDNA4, `gfx1200`, ROCm, Linux/CachyOS).

Priorities, in order: **1) VRAM savings — 2) decode speed — 3) prefill speed**. Vision runs on CPU and its latency is irrelevant. The engine is deliberately not generic: do not add generality it does not need.

### Non-negotiable constraints

- Weights stay **bit-exact** with the source GGUF: lossless reorder/re-layout only. **Never re-quantize.**
- Vision (`mmproj`) runs on **CPU**, never in VRAM.
- Python runs **only through `uv`** (`uv run` / `uvx`). Never `sudo pip`, never `--break-system-packages`, never the system Python.
- MTP must **not** slow down prefill.
- ROCm / HIP (`gfx1200`) is the reference backend.

## Language policy

- **Everything in the repository is written in English**: code, identifiers, comments, commit messages, branch names, issue and PR titles/bodies, documentation, file names.
- Conversation with the maintainer may be in Italian. The conversational language never leaks into artifacts: if a discussion happens in Italian, every commit, issue and document that comes out of it is still written in English.

## Git workflow

- **One branch per feature/fix; never commit directly to `main`.**
  - Naming: `<type>/<issue-number>-<short-slug>` — e.g. `feat/12-wmma-prefill-gemm`, `fix/31-deltanet-snapshot-offset`. If no issue exists yet, create one first (see below).
  - Types: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `chore`.
- **Commit frequently.** Many small, focused commits, each leaving the tree buildable; never batch unrelated changes into one commit.
- **Conventional Commits** in the imperative mood, in English: `feat: ...`, `fix: ...`, `docs: ...`, `perf: ...`.
- Reference the issue in commits: `Refs #<n>` while in progress, `Closes #<n>` when the work completes. Merge back to `main` through a PR (`gh pr create`) that closes the issue.
- Every AI-authored commit ends with the authoring agent's co-author trailer — for Command Code:

  ```
  Co-authored-by: CommandCodeBot <noreply@commandcode.ai>
  ```

  Pass the message via HEREDOC: `git commit -F - <<'EOF' ... EOF`.
- **Never commit large artifacts** (model files, converted weights, profiler dumps). `models/` is local-only and git-ignored. Benchmark results are the exception: CSV files under `bench/`, tagged with the engine commit hash (PLAN.md §17).

## GitHub issues — work tracking and decision log

`gh` is installed and authenticated (account **Nihmar**; repo `Nihmar/omphalos`). Use it directly — no additional setup is needed.

- **Track all work in issues.** Before starting anything non-trivial, create or find its issue: `gh issue create --title "..." --body "..."`. One issue per feature, fix, or milestone-sized unit; mention the relevant PLAN.md milestone ( §18) in the body.
- **Document decisions in the issue, not only in code.** When a design or implementation decision is made, add a comment with: context, alternatives considered, chosen option, rationale (`gh issue comment <n>`). The issue thread is the project's decision log; code comments and PR descriptions point back to it.
- When a decision changes the design, **update PLAN.md in the same change** and say so in the issue.
- When a `[verify]` item from PLAN.md ( §19) is resolved, record the finding in the relevant issue and update PLAN.md.

## Toolchain

- Engine: **C++20 + HIP**, CMake ≥ 3.21 (`CMAKE_HIP_ARCHITECTURES=gfx1200`).
- Tooling, reference implementation, validation: **Python via `uv`** + NumPy.
- Profiling and reference builds: `rocprofv3`, `hipblaslt-bench`, `rocminfo` / `amd-smi`, llama.cpp (HIP/Vulkan/CPU), `gguf-dump` (PLAN.md §6–§7).
- Once build/test/bench entry points exist, their exact commands are documented here.

## Working agreements

- PLAN.md is the source of truth for architecture; deviations are proposed in an issue first.
- Every performance claim comes with a measurement following PLAN.md §17 (fixed conditions, median of ≥5 runs, results in `bench/`).
- "Done" means: tree builds, relevant tests pass (once tests exist), results reported as measured — never claimed untested.
- Keep each change scoped to its issue; keep refactors separate from features.
