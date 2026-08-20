#!/usr/bin/env bash
# Compare one baseline result with one FlashGen RVQD benchmark result.
# Usage: ./flashgen_compare_rvqd.sh BASELINE.json RVQD.json [extra arguments]

set -euo pipefail

WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 BASELINE.json RVQD.json [extra arguments]" >&2
  exit 2
fi

python "${WORKSPACE_ROOT}/FlashGen/benchmark.py" compare \
  --baseline "$1" \
  --rvqd "$2" \
  --max-wer-increase 0.01 \
  --min-rtf-reduction 0.10 \
  "${@:3}"
