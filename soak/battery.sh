#!/bin/bash
# codec battery: $1=codec $2=tune(none|innerq)
set -u
CODEC="$1"; TUNE="${2:-none}"
MODEL=/home/sherntee/.cache/huggingface/hub/models--bartowski--Qwen2.5-Coder-32B-Instruct-GGUF/snapshots/40b525506a4f98ed425882fa6dfc90cc8139065e/Qwen2.5-Coder-32B-Instruct-IQ3_XXS.gguf
PORT=1920
cd /home/sherntee/20llms/FreeToken
LOG=/tmp/ft_soak_${CODEC}_${TUNE}.log
PAGES=8192   # 8192-token KV budget fits alongside 12G weights
EXTRA=""
[ "$CODEC" != "f16" ] && EXTRA="--kv-codec $CODEC"
[ "$TUNE" != "none" ] && EXTRA="$EXTRA --kv-codec-tune $TUNE"
nohup .venv/bin/ft serve --model-path "$MODEL" --served-model-name soak \
  --num-pages $PAGES --cuda-graph-max-bs 0 --host 127.0.0.1 --port $PORT \
  $EXTRA > "$LOG" 2>&1 &
FTPID=$!
for i in $(seq 1 60); do
  sleep 5
  if curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"ok"'; then break; fi
done
if ! curl -s -m 3 http://127.0.0.1:$PORT/health | grep -q '"ok"'; then
  echo "SERVER FAILED for $CODEC"; tail -5 "$LOG"; kill $FTPID 2>/dev/null; exit 1
fi
VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "codec=$CODEC tune=$TUNE vram_mib=$VRAM"
# Battery: (a) 2K prefill + 128 decode; (b) 8K prefill + 128 decode
P2K=$(python3 -c "print(' '.join(['x'*7]*290))")          # ~2k tokens
P8K=$(python3 -c "print(' '.join(['x'*7]*1150))")         # ~8k tokens
for name in 2k 8k; do
  if [ "$name" = 2k ]; then P="$P2K"; else P="$P8K"; fi
  # warm the prefix cache with one un-timed run, then time one run
  curl -s -m 300 http://127.0.0.1:$PORT/v1/completions -H "Content-Type: application/json" \
    -d "$(python3 -c "import json,sys; print(json.dumps({'model':'soak','prompt':sys.stdin.read(),'max_tokens':16,'temperature':0}))" <<< "$P")" > /dev/null
  START=$(date +%s.%N)
  curl -s -m 600 http://127.0.0.1:$PORT/v1/completions -H "Content-Type: application/json" \
    -d "$(python3 -c "import json,sys; print(json.dumps({'model':'soak','prompt':sys.stdin.read(),'max_tokens':128,'temperature':0}))" <<< "$P")" > /tmp/soak_out.json
  END=$(date +%s.%N)
  ELAPSED=$(python3 -c "print(f'{$END-$START:.2f}')")
  echo "codec=$CODEC ctx=$name wall_s=$ELAPSED"
done
kill $FTPID 2>/dev/null; wait $FTPID 2>/dev/null
echo "codec=$CODEC done"
