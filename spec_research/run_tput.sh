#!/bin/bash
# Throughput sweep on one GPU. GRAPH=1: CUDA graphs on (captured up to the largest bs); GRAPH=0: all eager.
# Overlap scheduling is off for every config (comb needs it off).
# Configs: ar, chain S=7, comb S=7 K=4, eagle tree S=7 topk=7 (verify width 8 for all spec configs).
# Data: first 10 prompts of each specbench category; per bs the category is repeated so there are >= 3 waves.
# Usage: GPU=0 [GRAPH=1] [BS="1 8 32 64"] [CONFIGS="ar chain comb tree"] bash run_tput.sh
set -u
R=/home/lrz/AI_infra/spec_research
PY=/home/lrz/AI_infra/envs/sglang-spec/bin/python
M=/home/lrz/AI_infra/models
PORT=${PORT:-30151}
BS=${BS:-"1 8 32 64"}
MAXBS=$(echo $BS | tr ' ' '\n' | sort -n | tail -1)
GRAPH=${GRAPH:-1}
if [ $GRAPH = 1 ]; then GARGS="--cuda-graph-max-bs-decode $MAXBS"; SUF=graph; else GARGS="--disable-cuda-graph"; SUF=eager; fi
OUT=$R/tput; mkdir -p $OUT
DATA=$OUT/specbench_10.jsonl
[ -f $DATA ] || $PY -c "
import json, collections
n = collections.Counter()
with open('$DATA', 'w') as f:
    for l in open('/home/lrz/AI_infra/a100_pkg/specbench.jsonl'):
        r = json.loads(l)
        if n[r['category']] < 10:
            n[r['category']] += 1; f.write(l)
"
for cfg in ${CONFIGS:-ar chain comb tree}; do
  COMB_K=0; SPEC="--speculative-algorithm EAGLE3 --speculative-draft-model-path $M/Qwen3-8B_eagle3 --speculative-num-steps 7 --speculative-num-draft-tokens 8"
  case $cfg in
    ar)    SPEC="" ;;
    chain) SPEC="$SPEC --speculative-eagle-topk 1" ;;
    comb)  SPEC="$SPEC --speculative-eagle-topk 1"; COMB_K=4 ;;
    tree)  SPEC="$SPEC --speculative-eagle-topk 7" ;;
  esac
  LOG=$OUT/server_${cfg}_$SUF.log
  echo "== $cfg $(date)"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU SGLANG_COMB_K=$COMB_K \
  setsid $PY -m sglang.launch_server --model-path $M/Qwen3-8B --port $PORT \
    --mem-fraction-static ${MEM:-0.6} --max-running-requests $MAXBS --context-length 4096 \
    --disable-radix-cache --disable-overlap-schedule $GARGS $SPEC > $LOG 2>&1 &
  SPID=$!
  for i in $(seq 1800); do
    curl -s -m 2 localhost:$PORT/health >/dev/null && break
    kill -0 $SPID 2>/dev/null || { echo "server died: $LOG"; continue 2; }
    sleep 2
  done
  for bs in $BS; do
    copies=$(( (3 * bs + 9) / 10 )); [ $copies -lt 1 ] && copies=1
    $PY /home/lrz/AI_infra/a100_pkg/bench.py --port $PORT --model $M/Qwen3-8B --tag $cfg --bs $bs \
      --copies $copies --data $DATA --out $OUT/results_$SUF.jsonl --server-log $LOG >> $OUT/bench_${cfg}_$SUF.out 2>&1
  done
  kill -TERM -- -$SPID 2>/dev/null; sleep 10; kill -KILL -- -$SPID 2>/dev/null
done
echo TPUT_DONE $(date)
