ASCEND_RT_VISIBLE_DEVICES=7

# WARMUP
echo 'WARM UP'
python ../tts.py --port 18091 --num-warmups 0 --num-prompts 1 --max-concurrency 1

python ../vllm-omni/prse.py \
  --label baseline \
  --host 127.0.0.1 \
  --port 18091 \
  --model /home/x50058110/tts-aclg/Qwen3-TTS-12Hz-1.7B-Base \
  --seed-tts-dataset-path /home/x50058110/tts-aclg/seedtts_testset \
  --num-prompts 200 \
  --seed 42 \
  --max-concurrency 1 \
  --seed-tts-locale zh \
  --seed-tts-eval-device npu:0 \
  --seed-tts-whisper-model /home/x50058110/tts-aclg/whisper-large-v3 \
  --seed-tts-paraformer-model /home/x50058110/tts-aclg/paraformer-zh \
  --seed-tts-wer-save-items \
  --seed-tts-utmos-device npu:0 \
  --seed-tts-utmos-model /home/x50058110/tts-aclg/utmos/utmos.jit \
  --seed-tts-sim-device npu:0 \
  --seed-tts-wavlm-model /home/x50058110/tts-aclg/wavlm-base-plus \

python ../vllm-omni/prse.py \
  --label baseline \
  --host 127.0.0.1 \
  --port 18091 \
  --model /home/x50058110/tts-aclg/Qwen3-TTS-12Hz-1.7B-Base \
  --seed-tts-dataset-path /home/x50058110/tts-aclg/seedtts_testset \
  --num-prompts 200 \
  --seed 42 \
  --max-concurrency 4 \
  --seed-tts-locale zh \
  --seed-tts-eval-device npu:0 \
  --seed-tts-whisper-model /home/x50058110/tts-aclg/whisper-large-v3 \
  --seed-tts-paraformer-model /home/x50058110/tts-aclg/paraformer-zh \
  --seed-tts-wer-save-items \
  --seed-tts-utmos-device npu:0 \
  --seed-tts-utmos-model /home/x50058110/tts-aclg/utmos/utmos.jit \
  --seed-tts-sim-device npu:0 \
  --seed-tts-wavlm-model /home/x50058110/tts-aclg/wavlm-base-plus \


python ../vllm-omni/prse.py \
  --label baseline \
  --host 127.0.0.1 \
  --port 18091 \
  --model /home/x50058110/tts-aclg/Qwen3-TTS-12Hz-1.7B-Base \
  --seed-tts-dataset-path /home/x50058110/tts-aclg/seedtts_testset \
  --num-prompts 200 \
  --seed 42 \
  --max-concurrency 8 \
  --seed-tts-locale zh \
  --seed-tts-eval-device npu:0 \
  --seed-tts-whisper-model /home/x50058110/tts-aclg/whisper-large-v3 \
  --seed-tts-paraformer-model /home/x50058110/tts-aclg/paraformer-zh \
  --seed-tts-wer-save-items \
  --seed-tts-utmos-device npu:0 \
  --seed-tts-utmos-model /home/x50058110/tts-aclg/utmos/utmos.jit \
  --seed-tts-sim-device npu:0 \
  --seed-tts-wavlm-model /home/x50058110/tts-aclg/wavlm-base-plus \

python ../vllm-omni/prse.py \
  --label baseline \
  --host 127.0.0.1 \
  --port 18091 \
  --model /home/x50058110/tts-aclg/Qwen3-TTS-12Hz-1.7B-Base \
  --seed-tts-dataset-path /home/x50058110/tts-aclg/seedtts_testset \
  --num-prompts 200 \
  --seed 42 \
  --max-concurrency 1 \
  --seed-tts-locale en \
  --seed-tts-eval-device npu:0 \
  --seed-tts-whisper-model /home/x50058110/tts-aclg/whisper-large-v3 \
  --seed-tts-paraformer-model /home/x50058110/tts-aclg/paraformer-zh \
  --seed-tts-wer-save-items \
  --seed-tts-utmos-device npu:0 \
  --seed-tts-utmos-model /home/x50058110/tts-aclg/utmos/utmos.jit \
  --seed-tts-sim-device npu:0 \
  --seed-tts-wavlm-model /home/x50058110/tts-aclg/wavlm-base-plus \

python ../vllm-omni/prse.py \
  --label baseline \
  --host 127.0.0.1 \
  --port 18091 \
  --model /home/x50058110/tts-aclg/Qwen3-TTS-12Hz-1.7B-Base \
  --seed-tts-dataset-path /home/x50058110/tts-aclg/seedtts_testset \
  --num-prompts 200 \
  --seed 42 \
  --max-concurrency 4 \
  --seed-tts-locale en \
  --seed-tts-eval-device npu:0 \
  --seed-tts-whisper-model /home/x50058110/tts-aclg/whisper-large-v3 \
  --seed-tts-paraformer-model /home/x50058110/tts-aclg/paraformer-zh \
  --seed-tts-wer-save-items \
  --seed-tts-utmos-device npu:0 \
  --seed-tts-utmos-model /home/x50058110/tts-aclg/utmos/utmos.jit \
  --seed-tts-sim-device npu:0 \
  --seed-tts-wavlm-model /home/x50058110/tts-aclg/wavlm-base-plus \


python ../vllm-omni/prse.py \
  --label baseline \
  --host 127.0.0.1 \
  --port 18091 \
  --model /home/x50058110/tts-aclg/Qwen3-TTS-12Hz-1.7B-Base \
  --seed-tts-dataset-path /home/x50058110/tts-aclg/seedtts_testset \
  --num-prompts 200 \
  --seed 42 \
  --max-concurrency 8 \
  --seed-tts-locale en \
  --seed-tts-eval-device npu:0 \
  --seed-tts-whisper-model /home/x50058110/tts-aclg/whisper-large-v3 \
  --seed-tts-paraformer-model /home/x50058110/tts-aclg/paraformer-zh \
  --seed-tts-wer-save-items \
  --seed-tts-utmos-device npu:0 \
  --seed-tts-utmos-model /home/x50058110/tts-aclg/utmos/utmos.jit \
  --seed-tts-sim-device npu:0 \
  --seed-tts-wavlm-model /home/x50058110/tts-aclg/wavlm-base-plus \