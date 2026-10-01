#!/usr/bin/env bash
# M5 baseline (PLAN.md §13.8): llama.cpp's own KV-quantization error at long
# context — KL of -ctk q8_0 -ctv q4_0 (and q8_0/q8_0) against its f16 KV, one
# chunk of CTX tokens of wikitext-2 test. llama-perplexity scores the second
# half of the chunk. This is the budget omphalos' quantized KV is held to (#58).
#
# usage: bench/m5_llama_kv_kl.sh <llama.cpp-build-bin-dir> <ctx> [workdir]
# The f16 base file is n_vocab * ctx/2 * 2 bytes (8.1 GB at 32k): keep workdir
# on a disk, not a tmpfs.
set -euo pipefail
BIN=${1:?usage: m5_llama_kv_kl.sh <llama.cpp-build-bin-dir> <ctx> [workdir]}
CTX=${2:?usage: m5_llama_kv_kl.sh <llama.cpp-build-bin-dir> <ctx> [workdir]}
WORK=${3:-/var/tmp/omphalos-llama-kl}
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL=$root/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
TEXT=$root/models/datasets/wikitext-2-raw/wiki.test.raw
mkdir -p "$WORK"
# the base file is ~8 GB at 32k: never leave it behind, even on a failed run
trap 'rm -f "$WORK/base-$CTX.kld"' EXIT
common=(-m "$MODEL" -f "$TEXT" -c "$CTX" -b 512 -ub 512 --chunks 1 -ngl 99 -fa on)

"$BIN/llama-perplexity" "${common[@]}" -ctk f16 -ctv f16 \
    --kl-divergence-base "$WORK/base-$CTX.kld" > "$WORK/base-$CTX.log" 2>&1
for kv in "q8_0 q4_0" "q8_0 q8_0" "q4_0 q4_0"; do
    set -- $kv
    "$BIN/llama-perplexity" "${common[@]}" -ctk "$1" -ctv "$2" \
        --kl-divergence-base "$WORK/base-$CTX.kld" --kl-divergence \
        > "$WORK/kl-$CTX-$1-$2.log" 2>&1
    echo "== ctx $CTX  K $1  V $2"
    grep -E "Mean    KLD|Median  KLD|99.0%   KLD|Same top p" "$WORK/kl-$CTX-$1-$2.log" || true
done
