#!/usr/bin/env bash
# Build the golden-tensor dumper against a llama.cpp checkout.
#
# usage: build.sh <llama.cpp-dir>
set -euo pipefail

llama_dir="${1:?usage: build.sh <llama.cpp-dir>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

g++ -O2 -std=c++17 "$here/dump_tensors.cpp" \
    -I"$llama_dir/include" -I"$llama_dir/ggml/include" \
    -L"$llama_dir/build-hip/bin" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$llama_dir/build-hip/bin" \
    -o "$here/dump_tensors"

echo "built $here/dump_tensors"
