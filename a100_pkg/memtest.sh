#!/bin/bash
# Memory check at MEM, TP=$TP on GPUs $GPUS, CUDA graph up to bs $MAXBS: start each spec server, then one
# full bs=$MAXBS wave of the summarization subset (longest prompts, copied to >= MAXBS requests). Out: memtest/tp<TP>_bs<MAXBS>_mem<MEM>/
set -u
A=/home/lrz/AI_infra/a100_pkg; PY=/home/lrz/AI_infra/envs/sglang-spec/bin/python; M=/home/lrz/models
O=$A/memtest/tp${TP}_bs${MAXBS}_mem${MEM}${TAG:-}; mkdir -p $O; PORT=30123
grep '"category": "summarization"' $A/specbench.jsonl > $O/summ.jsonl
SPEC="--speculative-algorithm EAGLE3 --speculative-draft-model-path $M/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex --speculative-num-steps 7 --speculative-num-draft-tokens 8"
for cfg in ${CONFIGS:-chain tree comb}; do
  K=0; case $cfg in chain) T=1;; tree) T=7;; comb) T=1; K=4;; esac
  log=$O/server_$cfg.log
  echo "== $cfg MEM=$MEM $(date +%T)"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS SGLANG_FLASHINFER_WORKSPACE_SIZE=1073741824 SGLANG_COMB_K=$K \
  setsid $PY -m sglang.launch_server --model-path $M/Qwen3-30B-A3B-Instruct-2507 --trust-remote-code --tp-size $TP --port $PORT \
    --mem-fraction-static $MEM --max-running-requests $MAXBS --cuda-graph-max-bs-decode $MAXBS --disable-radix-cache \
    --context-length 2048 ${XARGS:-} $SPEC --speculative-eagle-topk $T > $log 2>&1 &
  pid=$!; up=0
  for i in $(seq 450); do
    curl -s -m 2 localhost:$PORT/health >/dev/null && { up=1; break; }
    kill -0 $pid 2>/dev/null || break
    grep -q "memory allocation failed with OOM" $log && break
    sleep 2
  done
  grep TP0 $log | grep -E "KV Cache is allocated|Capture .* end" | sed 's/.*TP0\] //' | cut -c1-120
  if [ $up = 1 ]; then
    echo "startup OK"
    $PY $A/bench.py --port $PORT --model $M/Qwen3-30B-A3B-Instruct-2507 --tag mem_$cfg --bs $MAXBS --copies $(( (MAXBS + 79) / 80 )) \
      --data $O/summ.jsonl --out $O/$cfg.jsonl --server-log $log > $O/bench_$cfg.out 2>&1 && echo "bs$MAXBS wave OK" || echo "bs$MAXBS wave FAILED"
    tail -1 $O/$cfg.jsonl | cut -c1-260
  else
    echo "startup FAILED"; grep -E "OOM|out of memory|Error" $log | head -2 | cut -c1-200
  fi
  kill -TERM -- -$pid 2>/dev/null; for i in $(seq 40); do pgrep -g $pid >/dev/null || break; sleep 2; done; kill -KILL -- -$pid 2>/dev/null; sleep 5
done
echo MEMTEST_DONE
