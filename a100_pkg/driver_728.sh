#!/bin/bash
# Inserts tree 7-2-8 (tree_N8_k2, 3 reps, bs 32,64,128) between the N=8 stage and the budget shrink of the running
# eagle experiment: wait until the N=8 verdict line appears in exp_eagle_tp2_bs128.out, stop that run (and the N=7
# server it may have started), run tree_N8_k2, then resume the shrink with SKIP_N0=1 RESUME=1 (log exp_eagle_shrink.out).
set -u
A=/home/lrz/AI_infra/a100_pkg; cd $A
log() { echo "[$(date '+%F %T')] $*"; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 TP=2 PORT=30123 START_TIMEOUT=3600 MODELS=/home/lrz/models \
  PY=/home/lrz/AI_infra/envs/sglang-spec/bin/python WORK=$A COMB=1 RESUME=1 REPS=3 MEM=0.83 BS_ALL=32,64,128 \
  SGLANG_FLASHINFER_WORKSPACE_SIZE=1073741824
log "waiting for the N=8 verdict"
until grep -qE "^N=8 tree does not beat chain|EXPERIMENTS_DONE" exp_eagle_tp2_bs128.out; do sleep 20; done
LOSE=$(grep -m1 "^N=8 tree does not beat chain" exp_eagle_tp2_bs128.out | sed 's/.*bs: \[\([^]]*\)\].*/\1/')
log "N=8 verdict: tree does not beat chain at bs [$LOSE]; stopping the main run"
for p in $(pgrep -u lrz -f "^bash run_experiments.sh eagle") $(pgrep -u lrz -f "^/home/lrz/AI_infra/envs/sglang-spec/bin/python -m sglang.launch_server"); do
  kill -TERM -- -$(ps -o pgid= -p $p | tr -d " ") 2>/dev/null
done
for i in $(seq 90); do [ -z "$(nvidia-smi -i 0,1 --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
sleep 10
for REP in 1 2 3; do
  log "tree_N8_k2 rep $REP"
  REP=$REP RUN_AR=0 ONLY=tree_N8_k2 TREE_K=2 bash run_all.sh eagle 8 $BS_ALL
done
log "tree_N8_k2 done"
if [ -n "$LOSE" ]; then
  log "resuming the budget shrink (SKIP_N0=1) -> exp_eagle_shrink.out"
  SKIP_N0=1 bash run_experiments.sh eagle > exp_eagle_shrink.out 2>&1
  log "shrink finished: $(tail -1 exp_eagle_shrink.out)"
fi
log DRIVER_728_DONE
