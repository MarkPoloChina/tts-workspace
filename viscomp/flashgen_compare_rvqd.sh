#!/usr/bin/env bash
# Compare the latest baseline/RVQD 1/4/8-concurrency service benchmark sweeps.

set -euo pipefail

WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

python "${WORKSPACE_ROOT}/FlashGen/benchmark.py" rvqd-latency-compare \
  --baseline-dir "${WORKSPACE_ROOT}/results/rvqd-service-latency/baseline" \
  --rvqd-dir "${WORKSPACE_ROOT}/results/rvqd-service-latency/rvqd" \
  --output-dir "${WORKSPACE_ROOT}/results/rvqd-service-latency/comparisons" \
  --concurrencies 1 4 8 \
  --expected-requests 50
