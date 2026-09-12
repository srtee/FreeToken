#!/bin/bash
# sbatch wrapper: run the codec battery or the 30-min soak on the GPU.
# Usage: soak-batch.sh <codec> <tune> [--soak]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
set -u
CODEC="${1:-f16}"
TUNE="${2:-none}"
MODE="${3:-battery}"
cd /home/sherntee/20llms/FreeToken
mkdir -p soak/results
if [ "$MODE" = "--soak" ]; then
  # battery first (baseline numbers for this codec), then the 30-min soak
  ./soak/battery.sh "$CODEC" "$TUNE" > "soak/results/${CODEC}-${TUNE}-battery.txt" 2>&1
  # battery.sh kills its server; start a fresh one for the soak
  MODEL=/home/sherntee/.cache/huggingface/hub/models--bartowski--Qwen2.5-Coder-32B-Instruct-GGUF/snapshots/40b525506a4f98ed425882fa6dfc90cc8139065e/Qwen2.5-Coder-32B-Instruct-IQ3_XXS.gguf
  nohup .venv/bin/ft serve --model-path "$MODEL" --served-model-name soak \
    --num-pages 8192 --cuda-graph-max-bs 0 --host 127.0.0.1 --port 1920 \
    --kv-codec "$CODEC" --kv-codec-tune "$TUNE" \
    > "/tmp/ft_soak_${CODEC}_${TUNE}_soak.log" 2>&1 &
  FTPID=$!
  for i in $(seq 1 60); do
    sleep 5
    curl -s -m 3 http://127.0.0.1:1920/health 2>/dev/null | grep -q '"ok"' && break
  done
  ./soak/soak30.sh
  kill $FTPID 2>/dev/null
else
  ./soak/battery.sh "$CODEC" "$TUNE" 2>&1 | tee "soak/results/${CODEC}-${TUNE}-battery.txt"
fi