#!/usr/bin/env bash
# Benchmark one already-running baseline or FlashGen RVQD vLLM-Omni service.

set -euo pipefail

WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ASCEND_RT_VISIBLE_DEVICES=7

python "${WORKSPACE_ROOT}/FlashGen/benchmark.py" rvqd \
  --label baseline \
  --host 127.0.0.1 \
  --port 18091 \
  --model "${WORKSPACE_ROOT}/Qwen3-TTS-12Hz-1.7B-Base" \
  --seed-tts-dataset-path "${WORKSPACE_ROOT}/seedtts_testset" \
  --seed-tts-locale zh \
  --num-prompts 50 \
  --max-concurrency 8 \
  --num-warmups 0 \
  --seed 42 \
  --seed-tts-wer-eval 1 \
  --seed-tts-sim-eval 1 \
  --seed-tts-utmos-eval 0 \
  --seed-tts-eval-device npu:0 \
  --seed-tts-whisper-model "${WORKSPACE_ROOT}/whisper-large-v3" \
  --seed-tts-paraformer-model "${WORKSPACE_ROOT}/paraformer-zh" \
  --seed-tts-sim-device npu:0 \
  --seed-tts-wavlm-model "${WORKSPACE_ROOT}/wavlm-base-plus" \
  --seed-tts-utmos-device npu:0 \
  --seed-tts-utmos-model "${WORKSPACE_ROOT}/utmos/utmos.jit" \
  --seed-tts-wer-save-items \
  --result-dir "${WORKSPACE_ROOT}/results/rvqd" \
  --summary-jsonl "${WORKSPACE_ROOT}/results/rvqd/summary.jsonl" \
  --max-seed-tts-mean-wer 100.0 \
  "$@"
