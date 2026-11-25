#!/bin/bash
# Sequential acceptance-length sweep on one GPU: Qwen3-8B + EAGLE3, verify width S+1 for S in 7 3.
cd /home/lrz/AI_infra/spec_research
M=/home/lrz/AI_infra/models
for S in 7 3; do
  for cfg in "chain 1" "comb 7" "comb 4" "tree 7"; do
    set -- $cfg
    GPU=${GPU:-0} NAME=q8e3_${1}_s${S}_k$2 MODEL=$M/Qwen3-8B DRAFT=$M/Qwen3-8B_eagle3 ALGO=EAGLE3 \
      MODE=$1 STEPS=$S K=$2 PER_CAT=20 bash run_accept.sh
  done
done
echo ALL_DONE $(date)
