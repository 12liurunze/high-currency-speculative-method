#!/bin/bash
# Collect big-tree dumps at bs=1 for offline tree-policy replay.
# Usage: GPU=2 NAME=qwen3_8b_eagle3 MODEL=... [DRAFT=...] ALGO=EAGLE3 TOPK=7 STEPS=7 [MEM=0.75] [PORT=30131]
#        [PER_CAT=20] [CHAT_KW='{"enable_thinking": false}'] [EXTRA=...] bash run_dump.sh
set -u
R=/home/lrz/AI_infra/spec_research
CHAT_KW=${CHAT_KW:-'{}'}
PY=${PY:-/home/lrz/AI_infra/envs/sglang-spec/bin/python}
POOL=$(( TOPK + (STEPS - 1) * TOPK * TOPK ))   # every candidate the draft produces gets verified
OUT=$R/dumps/$NAME; mkdir -p $OUT; rm -f $OUT/tag
DRAFT_ARG=""; [ -n "${DRAFT:-}" ] && DRAFT_ARG="--speculative-draft-model-path $DRAFT"
echo "== $NAME topk=$TOPK steps=$STEPS verify_tokens=$((POOL + 1)) $(date)"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$R/hook SPEC_DUMP_DIR=$OUT \
setsid $PY -m sglang.launch_server --model-path $MODEL --trust-remote-code --port ${PORT:-30131} \
  --mem-fraction-static ${MEM:-0.75} --max-running-requests 1 --context-length 4096 \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --speculative-algorithm $ALGO $DRAFT_ARG --speculative-num-steps $STEPS --speculative-eagle-topk $TOPK \
  --speculative-num-draft-tokens $((POOL + 1)) ${EXTRA:-} > $OUT/server.log 2>&1 &
SPID=$!
for i in $(seq 1800); do
  curl -s -m 2 localhost:${PORT:-30131}/health >/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "server died, see $OUT/server.log"; exit 1; }
  sleep 2
done
$PY $R/dump_client.py --port ${PORT:-30131} --model $MODEL --data /home/lrz/AI_infra/a100_pkg/specbench.jsonl \
  --dump-dir $OUT --per-cat ${PER_CAT:-20} --chat-kwargs "$CHAT_KW" > $OUT/client.out 2>&1
kill -TERM -- -$SPID 2>/dev/null; sleep 10; kill -KILL -- -$SPID 2>/dev/null
echo "DUMP_DONE $NAME $(date)"
