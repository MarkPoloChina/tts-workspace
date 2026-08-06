export ASCEND_RT_VISIBLE_DEVICES=7

SEED_TTS_WER_EVAL=1
SEED_TTS_SIM_EVAL=1
SEED_TTS_UTMOS_EVAL=0

# WARMUP
python ../tts.py --port 18091 --num-warmups 0 --num-prompts 1 --max-concurrency 1

python ../vllm-omni/unibench.py \
  --label baseline \
  --host 127.0.0.1 \
  --port 18091 \
  --model /home/x50058110/tts/Qwen3-TTS-12Hz-1.7B-Base \
  --seed-tts-dataset-path /home/x50058110/tts/seedtts_testset \
  --num-prompts 50 \
  --seed 42 \
  --max-concurrency 8 \
  --seed-tts-locale zh \
  --seed-tts-wer-eval "${SEED_TTS_WER_EVAL}" \
  --seed-tts-sim-eval "${SEED_TTS_SIM_EVAL}" \
  --seed-tts-utmos-eval "${SEED_TTS_UTMOS_EVAL}" \
  --seed-tts-eval-device npu:0 \
  --seed-tts-whisper-model /home/x50058110/tts/whisper-large-v3 \
  --seed-tts-paraformer-model /home/x50058110/tts/paraformer-zh \
  --seed-tts-wer-save-items \
  --seed-tts-utmos-device npu:0 \
  --seed-tts-utmos-model /home/x50058110/tts/utmos/utmos.jit \
  --seed-tts-sim-device npu:0 \
  --seed-tts-wavlm-model /home/x50058110/tts/wavlm-base-plus \
