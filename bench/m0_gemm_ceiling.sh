#!/usr/bin/env bash
# Milestone 0 — FP16 GEMM ceiling at the model's projection shapes (M = 512).
#
# rocBLAS is what llama.cpp links on ROCm, so these numbers are the realistic
# compute ceiling for prefill; hipblaslt-bench is not shipped by the distro.
# PLAN.md §7 / §17.
#
# usage: m0_gemm_ceiling.sh [outdir]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
outdir="${1:-$root/bench/results}"
mkdir -p "$outdir"

bench=/opt/rocm/bin/rocblas-bench
[[ -x "$bench" ]] || { echo "missing $bench" >&2; exit 1; }

ts="$(date +%Y%m%d-%H%M%S)"
out="$outdir/m0-gemm-rocblas-$ts.log"

# m n k label  (model: hidden 5120, ffn 17408, qkv 8192, vocab 248320)
shapes=(
  "512 17408 5120 ffn_gate_up"
  "512 5120 17408 ffn_down"
  "512 8192 5120 attn_qkv"
  "512 248320 5120 lm_head"
)

{
  echo "# omphalos M0 — rocBLAS FP16 GEMM ceiling (compute f32), M = 512"
  echo "# date: $(date -Is)"
  echo "# gpu:  gfx1200 / RX 9060 XT 16GB / ROCm 7.2.4"
} > "$out"

for shape in "${shapes[@]}"; do
  read -r m n k label <<< "$shape"
  echo "== $label: m=$m n=$n k=$k ==" | tee -a "$out"
  "$bench" -f gemm -r f16_r --compute_type f32_r \
    -m "$m" -n "$n" -k "$k" \
    --transposeA N --transposeB N 2>&1 | tee -a "$out"
done

echo "results: $out"
