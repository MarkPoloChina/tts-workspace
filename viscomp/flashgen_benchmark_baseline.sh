#!/usr/bin/env bash
# Benchmark one already-running baseline vLLM-Omni service at 1/4/8 concurrency.

set -euo pipefail

WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

for MAX_CONCURRENCY in 1 4 8; do
  echo "[FlashGen][RVQD][BENCH   ] baseline latency sweep | concurrency=${MAX_CONCURRENCY} requests=50 seed=42"
  python "${WORKSPACE_ROOT}/FlashGen/benchmark.py" rvqd \
    --label "baseline-c${MAX_CONCURRENCY}" \
    --host 127.0.0.1 \
    --port 18091 \
    --model "${WORKSPACE_ROOT}/Qwen3-TTS-12Hz-1.7B-Base" \
    --seed-tts-dataset-path "${WORKSPACE_ROOT}/seedtts_testset" \
    --seed-tts-locale zh \
    --num-prompts 50 \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --num-warmups 0 \
    --seed 42 \
    --metric-percentiles 50,90,99 \
    --save-detailed \
    --seed-tts-wer-eval 0 \
    --seed-tts-sim-eval 0 \
    --seed-tts-utmos-eval 0 \
    --result-dir "${WORKSPACE_ROOT}/results/rvqd-service-latency/baseline/concurrency_${MAX_CONCURRENCY}" \
    --summary-jsonl "${WORKSPACE_ROOT}/results/rvqd-service-latency/baseline/summary.jsonl"
done
