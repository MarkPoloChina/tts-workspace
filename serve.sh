export ASCEND_RT_VISIBLE_DEVICES=7

vllm serve /home/x50058110/tts-aclg/Qwen3-TTS-12Hz-1.7B-Base \
  --omni \
  --port 18091 \
  --allowed-local-media-path / \
  --deploy-config /home/x50058110/tts-aclg/tts.yaml \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/x50058110/tts-aclg/profiling","torch_profiler_with_stack": false}' \
  #   --deploy-config /home/x50058110/tts-aclg/tts.yaml \
#   --deploy-config /home/x50058110/tts-aclg/vllm-omni/vllm_omni/deploy/qwen3_tts.yaml \

