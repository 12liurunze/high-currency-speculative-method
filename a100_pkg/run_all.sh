#!/bin/bash
# One server config at a time: start server -> bench all bs in the list -> stop server; runs ar, chain_N<N>, tree_N<N>.
# Usage: [TP=8] [REP=1] [RUN_AR=1] [ONLY=tags] [TREE_K=ks] [COPIES=auto] [RESUME=0] [OUT=file] [EXTRA="--kv-cache-dtype fp8_e4m3"] bash run_all.sh <eagle|mtp> [budgets] [bs_list]
# COMB=1 also runs comb_N<N> (HedgeTree: topk=1 chain drafter + COMB_K (default 4) siblings per trunk position,
# same verify budget N; needs the comb-patched SGLang runtime, enabled per server via SGLANG_COMB_K)
# Optional hooks (shell commands): PRE_LAUNCH runs right before each server start, POST_STOP right after it is stopped
# (e.g. release / re-grab a GPU placeholder on a shared machine). Both are empty by default.
set -u
WORK=${WORK:-$(cd $(dirname $0) && pwd)}; MODELS=${MODELS:-$WORK/models}
[ -f $WORK/env.sh ] && source $WORK/env.sh
WHICH=$1; BUDGETS=${2:-8}; BS=${3:-"32,64,128,256"}
PY=${PY:-python}; MAXNEW=${MAXNEW:-256}; MEM=${MEM:-0.85}; TP=${TP:-1}; PORT=${PORT:-30000}; DATA=${DATA:-$WORK/specbench.jsonl}
REP=${REP:-1}; COPIES=${COPIES:-auto}; CHAT_KW='{}'
RES_FLAG=""; [ "${RESUME:-0}" = 1 ] && RES_FLAG=--resume
case $WHICH in
  # Qwen3-30B-A3B-Instruct-2507 + SpecForge EAGLE3 draft; draft max_position_embeddings=2048 -> context <= 2048
  eagle) MODEL=${EAGLE_MODEL:-$MODELS/Qwen3-30B-A3B-Instruct-2507}; BASE="--context-length 2048"
         SPEC="--speculative-algorithm EAGLE3 --speculative-draft-model-path ${EAGLE_DRAFT:-$MODELS/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex}" ;;
  # Qwen3.5-35B-A3B built-in MTP layer via NEXTN; 2048 context to match eagle; thinking off to match Instruct
  mtp)   MODEL=${MTP_MODEL:-$MODELS/Qwen3.5-35B-A3B}; BASE="--context-length 2048"
         SPEC="--speculative-algorithm NEXTN"; CHAT_KW='{"enable_thinking": false}' ;;
  *) echo "unknown model group $WHICH"; exit 1 ;;
esac
OUT=${OUT:-$WORK/results_${WHICH}.jsonl}
MAXBS=$(echo $BS | tr , "\n" | sort -n | tail -1)

stop_server() {  # $1=pid of the server's process group leader (started with setsid)
  kill -TERM -- -$1 2>/dev/null
  for i in $(seq 60); do pgrep -g $1 >/dev/null || break; sleep 2; done
  kill -KILL -- -$1 2>/dev/null
  # TP>1 spawns one scheduler per GPU; wait until all of them are gone so GPU memory and the port are free
  for i in $(seq 60); do pgrep -g $1 >/dev/null || break; sleep 2; done
  for i in $(seq 30); do curl -s -m 2 localhost:$PORT/health >/dev/null || break; sleep 2; done
  sleep 5
}

run_cfg() {  # $1=tag $2=extra server args
  # ONLY="tag1 tag2": run just these configs (empty = all)
  if [ -n "${ONLY:-}" ] && ! echo " $ONLY " | grep -q " $1 "; then return; fi
  if curl -s -m 2 localhost:$PORT/health >/dev/null; then echo "port $PORT busy (another server/run_all running?) - abort"; exit 1; fi
  # unique per (tag, rep, max bs, KV dtype) so later runs never overwrite earlier logs
  local kvt=bf16; case "${EXTRA:-}" in *fp8*) kvt=fp8;; esac
  local log=$WORK/logs/server_${WHICH}_$1_rep${REP}_bs${MAXBS}_${kvt}.log; mkdir -p $WORK/logs
  [ -n "${PRE_LAUNCH:-}" ] && eval "$PRE_LAUNCH"
  # own process group, so stop_server can take down every TP worker
  SGLANG_COMB_K=${COMB_K_CFG:-0} setsid $PY -m sglang.launch_server --model-path $MODEL --trust-remote-code --tp-size $TP --port $PORT \
    --mem-fraction-static $MEM --max-running-requests $MAXBS --cuda-graph-max-bs-decode $MAXBS \
    --disable-radix-cache $BASE ${EXTRA:-} $2 > $log 2>&1 &
  local pid=$!
  for i in $(seq $(( ${START_TIMEOUT:-1800} / 2 ))); do  # seconds to wait for the server (slow disks: raise it)
    curl -s -m 2 localhost:$PORT/health >/dev/null && break
    kill -0 $pid 2>/dev/null || { echo "{\"tag\":\"$1\",\"rep\":$REP,\"bs_list\":\"$BS\",\"error\":\"server died, see $log\"}" | tee -a $OUT; stop_server $pid; [ -n "${POST_STOP:-}" ] && eval "$POST_STOP"; return; }
    sleep 2
  done
  $PY $WORK/bench.py --port $PORT --model $MODEL --tag $1 --rep $REP --bs $BS --copies $COPIES $RES_FLAG --max-new $MAXNEW \
    --out $OUT --data $DATA --chat-kwargs "$CHAT_KW" --server-log $log
  stop_server $pid
  [ -n "${POST_STOP:-}" ] && eval "$POST_STOP"
}

[ ${RUN_AR:-1} = 1 ] && run_cfg ar ""
for N in $BUDGETS; do
  run_cfg chain_N$N "$SPEC --speculative-num-steps $((N-1)) --speculative-eagle-topk 1 --speculative-num-draft-tokens $N"
  # tree: depth = width = N-1, same verify budget N; N=8 -> (7,7,8)
  [ $N -ge 3 ] && run_cfg tree_N$N "$SPEC --speculative-num-steps $((N-1)) --speculative-eagle-topk $((N-1)) --speculative-num-draft-tokens $N"
  # TREE_K="2 ...": extra trees at the same depth N-1 / budget N with topk K, tagged tree_N<N>_k<K> (ignored by losing_bs.py)
  for K in ${TREE_K:-}; do [ $N -ge 3 ] && run_cfg tree_N${N}_k$K "$SPEC --speculative-num-steps $((N-1)) --speculative-eagle-topk $K --speculative-num-draft-tokens $N"; done
  [ "${COMB:-0}" = 1 ] && COMB_K_CFG=${COMB_K:-4} run_cfg comb_N$N "$SPEC --speculative-num-steps $((N-1)) --speculative-eagle-topk 1 --speculative-num-draft-tokens $N"
done
echo ALL_DONE
