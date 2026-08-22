#!/usr/bin/env bash
# Launch Qwen3-TTS RVQD training through FlashGen on Ascend.
#
# FlashGen and model/data directories are expected below WORKSPACE_ROOT.
# Append dotted-key arguments to override the values below.

set -euo pipefail

WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ASCEND_RT_VISIBLE_DEVICES=7

python "${WORKSPACE_ROOT}/FlashGen/train.py" rvqd \
  --config "${WORKSPACE_ROOT}/FlashGen/flashgen/configs/qwen3_tts_rvqd.yaml" \
  --models.teacher.init_from "${WORKSPACE_ROOT}/Qwen3-TTS-12Hz-1.7B-Base" \
  --models.teacher.device npu:0 \
  --models.teacher.dtype bfloat16 \
  --models.teacher.attn_implementation sdpa \
  --models.teacher.trace_aclgraph true \
  --models.teacher.trace_aclgraph_warmups 3 \
  --method.truncation_k 8 \
  --method.model_dim 512 \
  --method.num_layers 4 \
  --method.num_attention_heads 8 \
  --method.intermediate_size 2048 \
  --method.dropout 0.05 \
  --method.rms_norm_eps 1e-6 \
  --method.do_sample false \
  --method.top_k 50 \
  --method.top_p 1.0 \
  --method.temperature 0.9 \
  --method.talker_do_sample true \
  --method.talker_top_k 50 \
  --method.talker_top_p 1.0 \
  --method.talker_temperature 0.9 \
  --method.repetition_penalty 1.05 \
  --method.max_new_tokens 2048 \
  --method.non_streaming_mode false \
  --method.kd_temperature 2.0 \
  --method.ce_weight 1.0 \
  --method.kl_weight 1.0 \
  --method.hidden_weight 0.1 \
  --training.student_dtype float32 \
  --training.data.seed_tts_root "${WORKSPACE_ROOT}/seedtts_testset" \
  --training.data.locales "[en, zh]" \
  --training.data.train_fraction 0.5 \
  --training.data.max_train_rows null \
  --training.data.max_validation_rows 16 \
  --training.data.rollout_batch_size 4 \
  --training.data.cache_dtype bfloat16 \
  --training.data.seed 42 \
  --training.optimizer.learning_rate 2e-4 \
  --training.optimizer.weight_decay 0.01 \
  --training.optimizer.max_grad_norm 1.0 \
  --training.loop.batch_size 4096 \
  --training.loop.validation_batch_size 4096 \
  --training.loop.epochs 5 \
  --training.loop.gradient_accumulation_steps 1 \
  --training.loop.log_every 10 \
  --training.checkpoint.output_dir "${WORKSPACE_ROOT}/distillation_output" \
  "$@"
