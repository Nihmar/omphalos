#!/usr/bin/env bash
# Milestone 0 — rocprofv3 kernel trace of prefill with and without MTP.
#
# Drives a single-turn llama-cli run (fixed prompt file, 8 tokens generated)
# twice — spec off and draft-mtp n=2 — with identical build, model and flags,
# and keeps the rocprofv3 per-kernel summary of each in bench/results/.
# Raw traces land in bench/traces/ (git-ignored).
#
# usage: m0_trace_prefill.sh <llama-build-dir> <model.gguf>
set -euo pipefail

build="${1:?usage: m0_trace_prefill.sh <llama-build-dir> <model.gguf>}"
model="${2:?}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cli="$build/bin/llama-cli"
[[ -x "$cli" ]] || { echo "missing $cli" >&2; exit 1; }

mkdir -p "$root/bench/traces" "$root/bench/results"

prompt="$(cat "$root/bench/prompts/continue-code.txt")"
common=(-m "$model" -ngl 99 -fa on -c 4096 -n 8 -t 8 -st -p "$prompt")

echo "== trace: spec off =="
rocprofv3 --kernel-trace -f csv -S -o "$root/bench/traces/m0-prefill-off" -- \
  "$cli" "${common[@]}" > "$root/bench/results/m0-trace-prefill-off-summary.txt" 2>&1 || true

echo "== trace: spec on (draft-mtp n=2) =="
rocprofv3 --kernel-trace -f csv -S -o "$root/bench/traces/m0-prefill-mtp" -- \
  "$cli" "${common[@]}" --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.1 \
  > "$root/bench/results/m0-trace-prefill-mtp-summary.txt" 2>&1 || true

ls -la "$root/bench/traces/" || true
echo "summaries: bench/results/m0-trace-prefill-{off,mtp}-summary.txt"
