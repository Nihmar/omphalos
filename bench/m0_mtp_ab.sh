#!/usr/bin/env bash
# Milestone 0 — MTP A/B on llama-server: decode t/s, draft acceptance and
# prefill impact, using the fork's production-style spec flags
# (preset.final.ini [math-38-27b], benches/rx9060xt/bench-mtp-ab.sh).
#
# usage: m0_mtp_ab.sh <build-dir> <model.gguf> [ctx]
#
# Configs: off / draft-mtp n=2 / draft-mtp n=4.
# Client: bench/m0_mtp_client.py, fixed prompts in bench/prompts/.
set -euo pipefail

build="${1:?usage: m0_mtp_ab.sh <build-dir> <model.gguf> [ctx]}"
model="${2:?}"
ctx="${3:-16384}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
outdir="$root/bench/results"
mkdir -p "$outdir"
srv="$build/bin/llama-server"
[[ -x "$srv" ]] || { echo "missing $srv" >&2; exit 1; }

ts="$(date +%Y%m%d-%H%M%S)"
csv="$outdir/m0-mtp-ab-$ts.csv"
port=8192

wait_vram() {
  # ROCm can hold VRAM briefly after a process exits; wait before the next load.
  local i used
  for i in $(seq 1 60); do
    used=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | sort -n | tail -1 || true)
    if [[ "${used:-0}" -lt 2000000000 ]]; then return 0; fi
    sleep 1
  done
}

run_config() {
  local name="$1"; shift
  wait_vram
  local log="$outdir/m0-mtp-ab-$name-$ts.srv.log"
  echo "== config: $name $* ==" | tee -a "$log"
  "$srv" -m "$model" -ngl 99 -fa on -c "$ctx" -np 1 -ctk q8_0 -ctv q8_0 \
    --device ROCm0 --no-mmproj-offload -t 8 -tb 8 \
    --cache-type-k-draft q4_0 --cache-type-v-draft q4_0 \
    --port "$port" "$@" >> "$log" 2>&1 &
  local pid=$!
  local up=0
  for _ in $(seq 1 300); do
    if curl -sf -o /dev/null "http://127.0.0.1:$port/health"; then up=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if [[ "$up" -eq 1 ]]; then
    uv run --no-project "$root/bench/m0_mtp_client.py" "$port" "$name" "$csv"
  else
    echo "server did not come up for '$name' (log: $log)" >&2
  fi
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

run_config "off"
run_config "mtp-n2" --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.1
run_config "mtp-n4" --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.1

echo "results: $csv"
