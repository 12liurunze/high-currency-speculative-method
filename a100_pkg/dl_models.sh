#!/bin/bash
# Download target models + EAGLE3 draft from ModelScope into $MODELS
set -u
WORK=${WORK:-$(cd $(dirname $0) && pwd)}; MODELS=${MODELS:-$WORK/models}
[ -f $WORK/env.sh ] && source $WORK/env.sh
for m in Qwen/Qwen3-30B-A3B-Instruct-2507 lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex Qwen/Qwen3.5-35B-A3B; do
  echo "== $m $(date)"
  for i in 1 2 3; do modelscope download --model $m --local_dir $MODELS/${m#*/} && break; echo retry $i; done
done
echo MS_DONE $(date)
