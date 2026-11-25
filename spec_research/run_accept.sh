#!/bin/bash
# Acceptance-length runs at bs=1 (greedy, eager) for chain / comb / eagle tree with the same verify width S+1.
# Usage: GPU=0 NAME=... MODEL=... [DRAFT=...] ALGO=EAGLE3 MODE=chain|comb|tree STEPS=7 [K=7] [MEM=0.6]
#        [PORT=30141] [PER_CAT=20] [CHAT_KW='{}'] bash run_accept.sh
set -u
R=/home/lrz/AI_infra/spec_research
CHAT_KW=${CHAT_KW:-'{}'}
PY=${PY:-/home/lrz/AI_infra/envs/sglang-spec/bin/python}
K=${K:-7}
COMB_K=0
case $MODE in
  chain) SPEC="--speculative-eagle-topk 1" ;;
  comb)  SPEC="--speculative-eagle-topk 1"; COMB_K=$K ;;
  tree)  SPEC="--speculative-eagle-topk $K" ;;
  *) echo "bad MODE=$MODE"; exit 1 ;;
esac
OUT=$R/accept/$NAME; mkdir -p $OUT; rm -f $OUT/tag $OUT/client.jsonl
DRAFT_ARG=""; [ -n "${DRAFT:-}" ] && DRAFT_ARG="--speculative-draft-model-path $DRAFT"
echo "== $NAME mode=$MODE steps=$STEPS K=$K verify_tokens=$((STEPS + 1)) $(date)"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU SGLANG_COMB_K=$COMB_K \
setsid $PY -m sglang.launch_server --model-path $MODEL --trust-remote-code --port ${PORT:-30141} \
  --mem-fraction-static ${MEM:-0.6} --max-running-requests 1 --context-length 4096 \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --speculative-algorithm $ALGO $DRAFT_ARG --speculative-num-steps $STEPS $SPEC \
  --speculative-num-draft-tokens $((STEPS + 1)) ${EXTRA:-} > $OUT/server.log 2>&1 &
SPID=$!
for i in $(seq 1800); do
  curl -s -m 2 localhost:${PORT:-30141}/health >/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "server died, see $OUT/server.log"; exit 1; }
  sleep 2
done
$PY $R/dump_client.py --port ${PORT:-30141} --model $MODEL --data /home/lrz/AI_infra/a100_pkg/specbench.jsonl \
  --dump-dir $OUT --per-cat ${PER_CAT:-20} --chat-kwargs "$CHAT_KW" > $OUT/client.out 2>&1
kill -TERM -- -$SPID 2>/dev/null; sleep 10; kill -KILL -- -$SPID 2>/dev/null
echo "ACCEPT_DONE $NAME $(date)"
