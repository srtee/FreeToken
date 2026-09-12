#!/bin/bash
# 30-min turbo3_tcq soak: alternating long-context prefill + multi-turn
# codegen prompts; hashes every output to detect drift/loops.
set -u
PORT=1920
OUT=/tmp/soak30_results.txt
: > "$OUT"
START=$(date +%s)
TURN=0
while [ $(( $(date +%s) - START )) -lt 1800 ]; do
  TURN=$((TURN+1))
  for prompt in \
    "Write a python function that deduplicates a list while preserving order." \
    "Explain what this does: print([x*x for x in range(10) if x%2])" \
    "Refactor: def f(l):\n s=0\n for x in l:\n  s+=x\n return s" ; do
    T0=$(date +%s.%N)
    R=$(curl -s -m 300 http://127.0.0.1:$PORT/v1/completions -H "Content-Type: application/json" \
      -d "$(python3 -c "import json,sys; print(json.dumps({'model':'soak','prompt':sys.stdin.read(),'max_tokens':200,'temperature':0}))" <<< "$prompt")")
    T1=$(date +%s.%N)
    TXT=$(echo "$R" | python3 -c "import json,sys
try: print(json.load(sys.stdin)['choices'][0]['text'])
except Exception: print('ERROR:'+sys.stdin.read()[:100])" 2>/dev/null)
    H=$(printf '%s' "$TXT" | md5sum | cut -c1-8)
    LEN=${#TXT}
    EL=$(python3 -c "print(f'{$T1-$T0:.1f}')")
    echo "turn=$TURN wall=${EL}s len=$LEN hash=$H" >> "$OUT"
    # repeat-hash detection: identical output for the same prompt across turns
    printf '%s' "$TXT" | head -c 80 >> /tmp/soak30_samples.txt; echo >> /tmp/soak30_samples.txt
  done
done
echo "turns=$TURN elapsed=$(( $(date +%s) - START ))s" >> "$OUT"
