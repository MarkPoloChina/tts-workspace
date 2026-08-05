#!/usr/bin/env bash
# Launch Qwen3-TTS fixed-K RVQ-tail distillation on Ascend.
#
# Seed 42 and the first 50% split are intentionally fixed inside the Python
# entry point. Override the remaining settings through environment variables,
# then append any extra Python arguments directly to this command.

set -euo pipefail

WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_OMNI_ROOT="${WORKSPACE_ROOT}/vllm-omni"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-${WORKSPACE_ROOT}/Qwen3-TTS-12Hz-1.7B-Base}"
SEED_TTS_ROOT="${SEED_TTS_ROOT:-${WORKSPACE_ROOT}/seedtts_testset}"
DISTILLATION_OUTPUT_DIR="${DISTILLATION_OUTPUT_DIR:-${WORKSPACE_ROOT}/distillation_output}"
TRAIN_DEVICE="${TRAIN_DEVICE:-npu:0}"
TRAIN_DTYPE="${TRAIN_DTYPE:-bfloat16}"
TRAIN_ATTN_IMPLEMENTATION="${TRAIN_ATTN_IMPLEMENTATION:-sdpa}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4096}"
TRAIN_ROLLOUT_BATCH_SIZE="${TRAIN_ROLLOUT_BATCH_SIZE:-4}"
TRAIN_TRACE_EXTRACTION_FRAME_BATCH_SIZE="${TRAIN_TRACE_EXTRACTION_FRAME_BATCH_SIZE:-4096}"
TRAIN_STUDENT_DTYPE="${TRAIN_STUDENT_DTYPE:-float32}"
TRAIN_TRACE_CACHE_DTYPE="${TRAIN_TRACE_CACHE_DTYPE:-bfloat16}"
TRAIN_GRADIENT_ACCUMULATION_STEPS="${TRAIN_GRADIENT_ACCUMULATION_STEPS:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-2e-4}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-0.01}"
TRAIN_MAX_GRAD_NORM="${TRAIN_MAX_GRAD_NORM:-1.0}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
DISTILLATION_TRUNCATION_K="${DISTILLATION_TRUNCATION_K:-8}"
DISTILLATION_STATE_SIZE="${DISTILLATION_STATE_SIZE:-384}"
DISTILLATION_TEMPERATURE="${DISTILLATION_TEMPERATURE:-2.0}"
DISTILLATION_CE_WEIGHT="${DISTILLATION_CE_WEIGHT:-1.0}"
DISTILLATION_KL_WEIGHT="${DISTILLATION_KL_WEIGHT:-1.0}"
DISTILLATION_HIDDEN_WEIGHT="${DISTILLATION_HIDDEN_WEIGHT:-0.1}"
VALIDATION_ROWS_PER_LOCALE="${VALIDATION_ROWS_PER_LOCALE:-16}"
VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-4096}"
ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
ROLLOUT_SUBTALKER_DO_SAMPLE="${ROLLOUT_SUBTALKER_DO_SAMPLE:-false}"
ROLLOUT_NON_STREAMING_MODE="${ROLLOUT_NON_STREAMING_MODE:-false}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-50}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.9}"
ROLLOUT_REPETITION_PENALTY="${ROLLOUT_REPETITION_PENALTY:-1.05}"
ROLLOUT_SUBTALKER_TOP_K="${ROLLOUT_SUBTALKER_TOP_K:-50}"
ROLLOUT_SUBTALKER_TOP_P="${ROLLOUT_SUBTALKER_TOP_P:-1.0}"
ROLLOUT_SUBTALKER_TEMPERATURE="${ROLLOUT_SUBTALKER_TEMPERATURE:-0.9}"
ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-2048}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-7}"
DISTILLATION_LOCALES="${DISTILLATION_LOCALES:-en zh}"

read -r -a LOCALE_ARGS <<< "${DISTILLATION_LOCALES}"
if [[ "${#LOCALE_ARGS[@]}" -eq 0 ]]; then
  echo "DISTILLATION_LOCALES must contain at least one locale (en and/or zh)." >&2
  exit 2
fi

case "${ROLLOUT_DO_SAMPLE}" in
  true|1|yes|on) ROLLOUT_SAMPLE_ARG="--rollout-do-sample" ;;
  false|0|no|off) ROLLOUT_SAMPLE_ARG="--no-rollout-do-sample" ;;
  *) echo "ROLLOUT_DO_SAMPLE must be true or false." >&2; exit 2 ;;
esac
case "${ROLLOUT_SUBTALKER_DO_SAMPLE}" in
  true|1|yes|on) ROLLOUT_SUBTALKER_SAMPLE_ARG="--rollout-subtalker-do-sample" ;;
  false|0|no|off) ROLLOUT_SUBTALKER_SAMPLE_ARG="--no-rollout-subtalker-do-sample" ;;
  *) echo "ROLLOUT_SUBTALKER_DO_SAMPLE must be true or false." >&2; exit 2 ;;
esac
case "${ROLLOUT_NON_STREAMING_MODE}" in
  true|1|yes|on) ROLLOUT_STREAMING_ARG="--rollout-non-streaming-mode" ;;
  false|0|no|off) ROLLOUT_STREAMING_ARG="--no-rollout-non-streaming-mode" ;;
  *) echo "ROLLOUT_NON_STREAMING_MODE must be true or false." >&2; exit 2 ;;
esac

export ASCEND_RT_VISIBLE_DEVICES
export PYTHONPATH="${VLLM_OMNI_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" "${VLLM_OMNI_ROOT}/train_qwen3_tts_tail_distillation.py" \
  --model-path "${MODEL_PATH}" \
  --dataset-root "${SEED_TTS_ROOT}" \
  --output-dir "${DISTILLATION_OUTPUT_DIR}" \
  --locales "${LOCALE_ARGS[@]}" \
  --device "${TRAIN_DEVICE}" \
  --dtype "${TRAIN_DTYPE}" \
  --student-dtype "${TRAIN_STUDENT_DTYPE}" \
  --trace-cache-dtype "${TRAIN_TRACE_CACHE_DTYPE}" \
  --attn-implementation "${TRAIN_ATTN_IMPLEMENTATION}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --rollout-batch-size "${TRAIN_ROLLOUT_BATCH_SIZE}" \
  --trace-extraction-frame-batch-size "${TRAIN_TRACE_EXTRACTION_FRAME_BATCH_SIZE}" \
  --gradient-accumulation-steps "${TRAIN_GRADIENT_ACCUMULATION_STEPS}" \
  --epochs "${TRAIN_EPOCHS}" \
  --learning-rate "${TRAIN_LEARNING_RATE}" \
  --weight-decay "${TRAIN_WEIGHT_DECAY}" \
  --max-grad-norm "${TRAIN_MAX_GRAD_NORM}" \
  --log-every "${TRAIN_LOG_EVERY}" \
  --truncation-k "${DISTILLATION_TRUNCATION_K}" \
  --state-size "${DISTILLATION_STATE_SIZE}" \
  --temperature "${DISTILLATION_TEMPERATURE}" \
  --ce-weight "${DISTILLATION_CE_WEIGHT}" \
  --kl-weight "${DISTILLATION_KL_WEIGHT}" \
  --hidden-weight "${DISTILLATION_HIDDEN_WEIGHT}" \
  --validation-rows-per-locale "${VALIDATION_ROWS_PER_LOCALE}" \
  --validation-batch-size "${VALIDATION_BATCH_SIZE}" \
  "${ROLLOUT_SAMPLE_ARG}" \
  --rollout-top-k "${ROLLOUT_TOP_K}" \
  --rollout-top-p "${ROLLOUT_TOP_P}" \
  --rollout-temperature "${ROLLOUT_TEMPERATURE}" \
  --rollout-repetition-penalty "${ROLLOUT_REPETITION_PENALTY}" \
  "${ROLLOUT_SUBTALKER_SAMPLE_ARG}" \
  --rollout-subtalker-top-k "${ROLLOUT_SUBTALKER_TOP_K}" \
  --rollout-subtalker-top-p "${ROLLOUT_SUBTALKER_TOP_P}" \
  --rollout-subtalker-temperature "${ROLLOUT_SUBTALKER_TEMPERATURE}" \
  --rollout-max-new-tokens "${ROLLOUT_MAX_NEW_TOKENS}" \
  "${ROLLOUT_STREAMING_ARG}" \
  "$@"
