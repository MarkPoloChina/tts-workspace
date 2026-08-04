curl -X POST "http://localhost:18091/start_profile"

sleep 1s

python tts.py --port 18091 --num-warmups 0 --num-prompts 32 --max-concurrency 8

curl -X POST "http://localhost:18091/stop_profile"
