#!/usr/bin/env bash
# Milestone 0 — hipBLASLt GEMM ceiling (fp16 + int8) at the model's projection
# shapes (M = 512).
#
# The distro rocBLAS is built without hipBLASLt (~3.7 TFLOPS flat, unusable);
# hipBLASLt itself is installed with gfx1200 Tensile kernels but ships no bench
# client, so we link the library directly (see bench/hipblaslt_gemm_bench.hip).
#
# usage: m0_gemm_hipblaslt.sh [outdir]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
outdir="${1:-$root/bench/results}"
mkdir -p "$outdir"

bin="$root/bench/build/hipblaslt_gemm_bench"
if [[ ! -x "$bin" ]]; then
  echo "missing $bin — build it with:" >&2
  echo "  hipcc -O3 --offload-arch=gfx1200 bench/hipblaslt_gemm_bench.hip -o bench/build/hipblaslt_gemm_bench -L/opt/rocm/lib -lhipblaslt -Wl,-rpath,/opt/rocm/lib" >&2
  exit 1
fi

ts="$(date +%Y%m%d-%H%M%S)"
csv="$outdir/m0-gemm-hipblaslt-$ts.csv"
echo "label,m,n,k,dtype,median_ms,best_ms,worst_ms,tflops,workspace_mb" > "$csv"

# model: hidden 5120, ffn 17408, qkv 8192, vocab 248320
shapes=(
  "ffn_gate_up 512 17408 5120"
  "ffn_down 512 5120 17408"
  "attn_qkv 512 8192 5120"
  "lm_head 512 248320 5120"
)

for dtype in fp16 int8; do
  for shape in "${shapes[@]}"; do
    # shellcheck disable=SC2086
    "$bin" $shape 20 "$dtype" | tee -a "$csv"
  done
done

echo "results: $csv"
