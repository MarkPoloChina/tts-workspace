#!/usr/bin/env bash
# Serve Qwen3-TTS with a FlashGen RVQD artifact through the regular vLLM CLI.
#
# Prerequisite: FlashGen must be installed in the same Python environment as
# vLLM/vLLM-Omni so their plugin entry-point discovery can find flashgen_rvqd.
# DEPLOY_CONFIG must keep vLLM-Omni's own Code Predictor truncation disabled;
# truncation K and Student weights are owned by the FlashGen artifact.

set -euo pipefail

WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_PATH="${MODEL_PATH:-${WORKSPACE_ROOT}/Qwen3-TTS-12Hz-1.7B-Base}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-${WORKSPACE_ROOT}/tts.yaml}"
DISTILLATION_OUTPUT_DIR="${DISTILLATION_OUTPUT_DIR:-${WORKSPACE_ROOT}/distillation_output}"
if [[ -z "${FLASHGEN_RVQD_ARTIFACT:-}" ]]; then
  LATEST_RUN=""
  for CANDIDATE in "${DISTILLATION_OUTPUT_DIR}"/rvq_*; do
    if [[ -f "${CANDIDATE}/artifact/manifest.json" ]]; then
      LATEST_RUN="${CANDIDATE}"
    fi
  done
  FLASHGEN_RVQD_ARTIFACT="${LATEST_RUN:+${LATEST_RUN}/artifact}"
fi

SERVE_HOST="${SERVE_HOST:-0.0.0.0}"
SERVE_PORT="${SERVE_PORT:-18091}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-7}"

FLASHGEN_RVQD_STUDENT_DTYPE="${FLASHGEN_RVQD_STUDENT_DTYPE:-model}"
FLASHGEN_RVQD_ENFORCE_SAMPLING="${FLASHGEN_RVQD_ENFORCE_SAMPLING:-1}"

ENABLE_PROFILING="${ENABLE_PROFILING:-true}"
PROFILER_DIR="${PROFILER_DIR:-${WORKSPACE_ROOT}/profiling}"

if [[ ! -f "${DEPLOY_CONFIG}" ]]; then
  echo "vLLM-Omni deploy config not found: ${DEPLOY_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${FLASHGEN_RVQD_ARTIFACT}/manifest.json" ]]; then
  echo "FlashGen RVQD artifact not found: ${FLASHGEN_RVQD_ARTIFACT}" >&2
  echo "Run train_flashgen_rvqd.sh first or set FLASHGEN_RVQD_ARTIFACT." >&2
  exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES
export FLASHGEN_RVQD_ARTIFACT
export FLASHGEN_RVQD_STUDENT_DTYPE
export FLASHGEN_RVQD_ENFORCE_SAMPLING

# With no allowlist, vLLM discovers FlashGen and vLLM-Ascend automatically.
# If the user already has an allowlist, add FlashGen without removing entries.
if [[ "${VLLM_PLUGINS+x}" == "x" ]]; then
  case ",${VLLM_PLUGINS}," in
    *,flashgen_rvqd,*) ;;
    *) VLLM_PLUGINS="${VLLM_PLUGINS:+${VLLM_PLUGINS},}flashgen_rvqd" ;;
  esac
  export VLLM_PLUGINS
fi

VLLM_ARGS=(
  serve "${MODEL_PATH}"
  --omni
  --host "${SERVE_HOST}"
  --port "${SERVE_PORT}"
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}"
  --deploy-config "${DEPLOY_CONFIG}"
)

case "${ENABLE_PROFILING}" in
  true|1|yes|on)
    PROFILER_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILER_DIR}\",\"torch_profiler_with_stack\":false}"
    VLLM_ARGS+=(--profiler-config "${PROFILER_CONFIG}")
    ;;
  false|0|no|off) ;;
  *)
    echo "ENABLE_PROFILING must be true or false." >&2
    exit 2
    ;;
esac

exec "${VLLM_BIN}" "${VLLM_ARGS[@]}" "$@"
