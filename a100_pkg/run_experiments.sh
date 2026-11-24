#!/bin/bash
# Full flow for one group:
#  1) budget $N0 (8): {AR, chain, tree} at every bs in $BS_ALL; BF16 KV (bs listed in $FP8_BS, default none, use FP8 KV)
#  2) budget shrink: for each bs where tree does not beat chain (all categories, mean over reps),
#     retry chain/tree (no AR) at N0-1, ..., $MIN_N (tree = N-1,N-1,N) until tree wins at that bs
# Usage: TP=<n> [REPS=1] [COPIES=auto] [RESUME=0] [BS_ALL=...] [FP8_BS=] [N0=8] [MIN_N=3] bash run_experiments.sh <eagle|mtp>
#   RESUME=1 re-runs the same flow but skips (method, rep, bs, category) cells already in $OUT (after a crash)
set -u
WORK=${WORK:-$(cd $(dirname $0) && pwd)}; G=$1; OUT=${OUT:-$WORK/results_${G}.jsonl}; export OUT WORK
PY=${PY:-python}; REPS=${REPS:-1}; BS_ALL=${BS_ALL:-32,64,128,256,512}; FP8_BS=${FP8_BS:-}; N0=${N0:-8}; MIN_N=${MIN_N:-3}

in_fp8() { echo ",$FP8_BS," | grep -q ",$1,"; }
run_budget() {  # $1=N $2=bs list $3=RUN_AR ; reps outermost so drift spreads over methods
  local BF="" FP="" b
  for b in $(echo $2 | tr , " "); do if in_fp8 $b; then FP=$FP,$b; else BF=$BF,$b; fi; done
  BF=${BF#,}; FP=${FP#,}
  for REP in $(seq $REPS); do
    echo "== $G N=$1 bs=[$2] rep $REP $(date)"
    [ -n "$BF" ] && REP=$REP RUN_AR=$3 bash $WORK/run_all.sh $G $1 $BF
    [ -n "$FP" ] && REP=$REP RUN_AR=$3 EXTRA="--kv-cache-dtype fp8_e4m3" bash $WORK/run_all.sh $G $1 $FP
  done
}

run_budget $N0 $BS_ALL 1
LOSE=$($PY $WORK/losing_bs.py $OUT $N0 $BS_ALL)
for N in $(seq $((N0 - 1)) -1 $MIN_N); do
  echo "N=$((N + 1)) tree does not beat chain at bs: [$LOSE] $(date)"
  [ -z "$LOSE" ] && break
  run_budget $N $LOSE 0
  LOSE=$($PY $WORK/losing_bs.py $OUT $N $LOSE)
done
echo "final: tree still does not beat chain at bs [$LOSE] (empty = tree wins at every bs)"
$PY $WORK/summarize.py $OUT > $WORK/summary_${G}.txt; $PY $WORK/summarize.py $OUT --by-cat >> $WORK/summary_${G}.txt
echo "EXPERIMENTS_DONE $G $(date)"
