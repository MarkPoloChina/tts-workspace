#!/usr/bin/env bash
# Render direct Qwen3-TTS Pad/RVQD/Full audio comparisons without vLLM-Omni.
# Usage: ./flashgen_visualize_rvqd.sh ARTIFACT_OR_WEIGHTS [extra arguments]

set -euo pipefail

WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 ARTIFACT_OR_WEIGHTS [extra arguments]" >&2
  exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES=7

python "${WORKSPACE_ROOT}/FlashGen/benchmark.py" rvqd-visualize \
  --config "${WORKSPACE_ROOT}/FlashGen/flashgen/configs/qwen3_tts_rvqd.yaml" \
  --artifact "$1" \
  --model "${WORKSPACE_ROOT}/Qwen3-TTS-12Hz-1.7B-Base" \
  --seed-tts-root "${WORKSPACE_ROOT}/seedtts_testset" \
  --output-dir "${WORKSPACE_ROOT}/results/rvqd-visualization" \
  --num-samples 5 \
  --seed 42 \
  --pad-codec-id 0 \
  --student-dtype model \
  "${@:2}"
