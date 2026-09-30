#!/usr/bin/env bash
# Download and extract the wikitext-2-raw corpus used for the M0 quality
# baseline (perplexity / KL divergence). Local only: lands under models/
# (git-ignored).
#
# usage: get_corpus.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/models/datasets"
mkdir -p "$dest"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

hf download ggml-org/ci --repo-type dataset --include 'wikitext-2-raw-v1.zip' \
    --local-dir "$tmp"
uv run --no-project python -m zipfile -e "$tmp/wikitext-2-raw-v1.zip" "$dest/"

ls -la "$dest/wikitext-2-raw/"
