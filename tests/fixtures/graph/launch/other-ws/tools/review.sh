#!/usr/bin/env bash
# The same reviewer harness, in a workspace outside the requested scope.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="$1"; shift
codex exec --model gpt-5.6-sol -o "reviews/$NAME.md" "Review $* against SPEC.md" || true
