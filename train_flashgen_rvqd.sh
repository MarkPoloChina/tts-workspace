#!/usr/bin/env bash
# Launch Qwen3-TTS RVQD training through FlashGen on Ascend.
#
# FlashGen must be available under FLASHGEN_ROOT. Override settings through
# environment variables, then append additional FlashGen dotted-key overrides
# directly to this command.

set -euo pipefail

WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
FLASHGEN_ROOT="${FLASHGEN_ROOT:-${WORKSPACE_ROOT}/FlashGen}"
RVQD_CONFIG="${RVQD_CONFIG:-${FLASHGEN_ROOT}/flashgen/configs/qwen3_tts_rvqd.yaml}"

MODEL_PATH="${MODEL_PATH:-${WORKSPACE_ROOT}/Qwen3-TTS-12Hz-1.7B-Base}"
SEED_TTS_ROOT="${SEED_TTS_ROOT:-${WORKSPACE_ROOT}/seedtts_testset}"
DISTILLATION_OUTPUT_DIR="${DISTILLATION_OUTPUT_DIR:-${WORKSPACE_ROOT}/distillation_output}"

TRAIN_DEVICE="${TRAIN_DEVICE:-npu:0}"
TRAIN_DTYPE="${TRAIN_DTYPE:-bfloat16}"
TRAIN_STUDENT_DTYPE="${TRAIN_STUDENT_DTYPE:-float32}"
TRAIN_TRACE_CACHE_DTYPE="${TRAIN_TRACE_CACHE_DTYPE:-bfloat16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4096}"
TRAIN_ROLLOUT_BATCH_SIZE="${TRAIN_ROLLOUT_BATCH_SIZE:-4}"
TRAIN_GRADIENT_ACCUMULATION_STEPS="${TRAIN_GRADIENT_ACCUMULATION_STEPS:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-2e-4}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-0.01}"
TRAIN_MAX_GRAD_NORM="${TRAIN_MAX_GRAD_NORM:-1.0}"

DISTILLATION_TRUNCATION_K="${DISTILLATION_TRUNCATION_K:-8}"
DISTILLATION_STATE_SIZE="${DISTILLATION_STATE_SIZE:-384}"
DISTILLATION_TEMPERATURE="${DISTILLATION_TEMPERATURE:-2.0}"
DISTILLATION_CE_WEIGHT="${DISTILLATION_CE_WEIGHT:-1.0}"
DISTILLATION_KL_WEIGHT="${DISTILLATION_KL_WEIGHT:-1.0}"
DISTILLATION_HIDDEN_WEIGHT="${DISTILLATION_HIDDEN_WEIGHT:-0.1}"
DISTILLATION_LOCALES="${DISTILLATION_LOCALES:-[en, zh]}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.5}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-null}"
VALIDATION_ROWS_PER_LOCALE="${VALIDATION_ROWS_PER_LOCALE:-16}"
VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-4096}"
TRAIN_SEED="${TRAIN_SEED:-42}"

# Code Predictor sampling. These values must match the vLLM-Omni service's
# subtalker_sampling_params because the FlashGen plugin validates the contract.
RVQD_DO_SAMPLE="${RVQD_DO_SAMPLE:-true}"
RVQD_TOP_K="${RVQD_TOP_K:-50}"
RVQD_TOP_P="${RVQD_TOP_P:-1.0}"
RVQD_TEMPERATURE="${RVQD_TEMPERATURE:-0.9}"

# Talker sampling used only while collecting Teacher rollouts.
ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-50}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.9}"
ROLLOUT_REPETITION_PENALTY="${ROLLOUT_REPETITION_PENALTY:-1.05}"
ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-2048}"
ROLLOUT_NON_STREAMING_MODE="${ROLLOUT_NON_STREAMING_MODE:-false}"

ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-7}"

if [[ ! -f "${FLASHGEN_ROOT}/train.py" ]]; then
  echo "FlashGen training entry not found: ${FLASHGEN_ROOT}/train.py" >&2
  echo "Set FLASHGEN_ROOT to the FlashGen repository root." >&2
  exit 2
fi
if [[ ! -f "${RVQD_CONFIG}" ]]; then
  echo "RVQD config not found: ${RVQD_CONFIG}" >&2
  exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES

exec "${PYTHON_BIN}" "${FLASHGEN_ROOT}/train.py" rvqd \
  --config "${RVQD_CONFIG}" \
  --models.teacher.init_from "${MODEL_PATH}" \
  --models.teacher.device "${TRAIN_DEVICE}" \
  --models.teacher.dtype "${TRAIN_DTYPE}" \
  --method.truncation_k "${DISTILLATION_TRUNCATION_K}" \
  --method.state_size "${DISTILLATION_STATE_SIZE}" \
  --method.do_sample "${RVQD_DO_SAMPLE}" \
  --method.top_k "${RVQD_TOP_K}" \
  --method.top_p "${RVQD_TOP_P}" \
  --method.temperature "${RVQD_TEMPERATURE}" \
  --method.talker_do_sample "${ROLLOUT_DO_SAMPLE}" \
  --method.talker_top_k "${ROLLOUT_TOP_K}" \
  --method.talker_top_p "${ROLLOUT_TOP_P}" \
  --method.talker_temperature "${ROLLOUT_TEMPERATURE}" \
  --method.repetition_penalty "${ROLLOUT_REPETITION_PENALTY}" \
  --method.max_new_tokens "${ROLLOUT_MAX_NEW_TOKENS}" \
  --method.non_streaming_mode "${ROLLOUT_NON_STREAMING_MODE}" \
  --method.kd_temperature "${DISTILLATION_TEMPERATURE}" \
  --method.ce_weight "${DISTILLATION_CE_WEIGHT}" \
  --method.kl_weight "${DISTILLATION_KL_WEIGHT}" \
  --method.hidden_weight "${DISTILLATION_HIDDEN_WEIGHT}" \
  --training.student_dtype "${TRAIN_STUDENT_DTYPE}" \
  --training.data.seed_tts_root "${SEED_TTS_ROOT}" \
  --training.data.locales "${DISTILLATION_LOCALES}" \
  --training.data.train_fraction "${TRAIN_FRACTION}" \
  --training.data.max_train_rows "${MAX_TRAIN_ROWS}" \
  --training.data.max_validation_rows "${VALIDATION_ROWS_PER_LOCALE}" \
  --training.data.rollout_batch_size "${TRAIN_ROLLOUT_BATCH_SIZE}" \
  --training.data.cache_dtype "${TRAIN_TRACE_CACHE_DTYPE}" \
  --training.data.seed "${TRAIN_SEED}" \
  --training.optimizer.learning_rate "${TRAIN_LEARNING_RATE}" \
  --training.optimizer.weight_decay "${TRAIN_WEIGHT_DECAY}" \
  --training.optimizer.max_grad_norm "${TRAIN_MAX_GRAD_NORM}" \
  --training.loop.batch_size "${TRAIN_BATCH_SIZE}" \
  --training.loop.validation_batch_size "${VALIDATION_BATCH_SIZE}" \
  --training.loop.epochs "${TRAIN_EPOCHS}" \
  --training.loop.gradient_accumulation_steps "${TRAIN_GRADIENT_ACCUMULATION_STEPS}" \
  --training.checkpoint.output_dir "${DISTILLATION_OUTPUT_DIR}" \
  "$@"
