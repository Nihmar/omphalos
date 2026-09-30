#!/usr/bin/env bash
# Milestone 0 — llama.cpp baseline suite for omphalos (PLAN.md §7, §17).
#
# usage: m0_llama_bench.sh <build-dir> <label> <model.gguf> [outdir]
#
#   e.g. m0_llama_bench.sh ~/Projects/llama.cpp-RX9060XT-16GB/build-hip hip \
#                          models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
#
# Notes:
#   - llama-bench cannot drive MTP; MTP A/B runs go through llama-server instead.
#   - The CSV rows carry build commit, flags and model path (self-contained provenance);
#     run conditions are recorded in the tracking issue.
set -euo pipefail

build="${1:?usage: m0_llama_bench.sh <build-dir> <label> <model.gguf> [outdir]}"
label="${2:?}"
model="${3:?}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
outdir="${4:-$root/bench/results}"
mkdir -p "$outdir"

bench="$build/bin/llama-bench"
[[ -x "$bench" ]] || { echo "missing $bench" >&2; exit 1; }

ts="$(date +%Y%m%d-%H%M%S)"
csv="$outdir/m0-llama-bench-$label-$ts.csv"
log="$outdir/m0-llama-bench-$label-$ts.log"
tmp="$(mktemp)"

{
  echo "# omphalos M0 llama.cpp baselines"
  echo "# label: $label"
  echo "# build: $build"
  echo "# model: $model"
  echo "# date:  $(date -Is)"
  echo "# note:  desktop session active during run"
} > "$log"

wait_vram() {
  # ROCm can hold VRAM briefly after a process exits; wait before the next load.
  local i used
  for i in $(seq 1 60); do
    used=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | sort -n | tail -1 || true)
    if [[ "${used:-0}" -lt 2000000000 ]]; then return 0; fi
    sleep 1
  done
}

run() {
  wait_vram
  echo "+ llama-bench $*" | tee -a "$log"
  "$bench" -m "$model" -ngl 99 -fa on -r 5 -o csv "$@" > "$tmp" 2>> "$log"
  if [[ -s "$csv" ]]; then tail -n +2 "$tmp" >> "$csv"; else cat "$tmp" >> "$csv"; fi
}

# PLAN §7: prefill 512 / decode 128, decode depths 0 / 4k / 16k (f16 KV)
run -p 512 -n 128 -d 0,4096,16384
# PLAN §7: KV quantization baselines
run -p 512 -n 128 -ctk q8_0 -ctv q8_0 -d 16384,32768
run -p 512 -n 128 -ctk q8_0 -ctv q4_0 -d 16384,32768

rm -f "$tmp"
echo "results: $csv" | tee -a "$log"
