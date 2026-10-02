#!/usr/bin/env bash
# Build the golden-tensor dumper against a llama.cpp checkout.
#
# usage: build.sh <llama.cpp-dir> [build-subdir]
#   build-subdir: the llama.cpp build whose libraries to link (default build-hip)
set -euo pipefail

llama_dir="${1:?usage: build.sh <llama.cpp-dir> [build-subdir]}"
build="${2:-build-hip}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -d "$llama_dir/$build/bin" ]] || { echo "missing $llama_dir/$build/bin" >&2; exit 1; }

g++ -O2 -std=c++17 "$here/dump_tensors.cpp" \
    -I"$llama_dir/include" -I"$llama_dir/ggml/include" \
    -L"$llama_dir/$build/bin" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$llama_dir/$build/bin" \
    -o "$here/dump_tensors"

echo "built $here/dump_tensors"

# the image reference (#160): llama.cpp + mtmd, the text model on the GPU
g++ -O2 -std=c++17 "$here/dump_mtmd_logits.cpp" \
    -I"$llama_dir/include" -I"$llama_dir/ggml/include" -I"$llama_dir/tools/mtmd" \
    -L"$llama_dir/$build/bin" -lmtmd -lllama -lggml -lggml-base \
    -Wl,-rpath,"$llama_dir/$build/bin" \
    -o "$here/dump_mtmd_logits"

echo "built $here/dump_mtmd_logits"
