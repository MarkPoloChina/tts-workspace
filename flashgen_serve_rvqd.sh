#!/usr/bin/env bash
# Serve baseline or FlashGen RVQD Qwen3-TTS through the regular vLLM CLI.
#
# Prerequisite: FlashGen must be installed in the same Python environment as
# vLLM/vLLM-Omni so their plugin entry-point discovery can find flashgen_rvqd.
# A non-empty code_predictor_distillation_weights value in tts.yaml enables
# RVQD. code_predictor_distillation_tail_model selects the gru or transformer
# runtime. Omitting weights starts the full Code Predictor.

set -euo pipefail

WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ASCEND_RT_VISIBLE_DEVICES=7
export FLASHGEN_RVQD_STUDENT_DTYPE=model

# With no allowlist, vLLM discovers FlashGen and vLLM-Ascend automatically.
# If the user already has an allowlist, add FlashGen without removing entries.
if [[ -n "${VLLM_PLUGINS:-}" ]]; then
  case ",${VLLM_PLUGINS}," in
    *,flashgen_rvqd,*) ;;
    *) export VLLM_PLUGINS="${VLLM_PLUGINS},flashgen_rvqd" ;;
  esac
fi

vllm serve "${WORKSPACE_ROOT}/Qwen3-TTS-12Hz-1.7B-Base" \
  --omni \
  --host 0.0.0.0 \
  --port 18091 \
  --allowed-local-media-path / \
  --deploy-config "${WORKSPACE_ROOT}/tts.yaml" \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${WORKSPACE_ROOT}/profiling\",\"torch_profiler_with_stack\":false}" \
  "$@"
