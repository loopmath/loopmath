#!/usr/bin/env bash
# Synthetic reviewer harness for the launch fixtures: tools/review.sh <name> <file> ...
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="$1"; shift
mkdir -p reviews
codex exec --model gpt-5.6-sol -c model_reasoning_effort=medium --sandbox read-only \
  -o "reviews/$NAME.md" "Review $* against SPEC.md" > "reviews/.log-$NAME.txt" 2>&1 || true
cat "reviews/$NAME.md"
