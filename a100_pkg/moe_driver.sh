#!/bin/bash
# Unattended driver for the MoE (eagle group, AGENT.md) run on this 4x A6000 box:
#  1) wait for the Qwen3-8B CUDA-graph throughput sweep on GPU0 to finish (spec_research/tput_graph.out);
#  2) smoke-test comb with SGLang's default (overlap) scheduling on GPU0: Qwen3-8B + EAGLE3, CUDA graph, bs=8,
#     70 prompts x 3; COMB=1 iff the server runs overlapped, all 7 categories succeed and accept_len > 2.1;
#  3) wait until GPUs 0,1 or 2,3 are idle (< 1000 MiB used, no compute process) for 5 consecutive minutes;
#  4) on that pair, TP=2: check_env.sh eagle -> smoke (AGENT 5.3) -> formal run (AGENT 5.4) with BS_ALL=32,64,128,256,
#     comb_N<N> added when COMB=1. If the smoke fails only on comb rows, comb is dropped and the smoke is redone.
# Log: moe_driver.out (this dir). Stops at the first failed check and says why.
set -u
A=/home/lrz/AI_infra/a100_pkg; S=/home/lrz/AI_infra/spec_research
PY=/home/lrz/AI_infra/envs/sglang-spec/bin/python; M8=/home/lrz/AI_infra/models
cd $A
log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for the throughput sweep on GPU0"
until grep -q TPUT_DONE $S/tput_graph.out 2>/dev/null; do sleep 60; done

# ---- 2) comb + overlap smoke ----
SM=$S/tput/overlap_smoke; mkdir -p $SM; rm -f $SM/results.jsonl
log "comb overlap smoke on GPU0 (log $SM/server.log)"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 SGLANG_COMB_K=4 setsid $PY -m sglang.launch_server \
  --model-path $M8/Qwen3-8B --port 30161 --mem-fraction-static 0.6 --max-running-requests 8 \
  --cuda-graph-max-bs-decode 8 --context-length 4096 --disable-radix-cache \
  --speculative-algorithm EAGLE3 --speculative-draft-model-path $M8/Qwen3-8B_eagle3 \
  --speculative-num-steps 7 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8 > $SM/server.log 2>&1 &
SP=$!; up=0
for i in $(seq 1800); do
  curl -s -m 2 localhost:30161/health >/dev/null && { up=1; break; }
  kill -0 $SP 2>/dev/null || break; sleep 2
done
[ $up = 1 ] && $PY $A/bench.py --port 30161 --model $M8/Qwen3-8B --tag comb_overlap --bs 8 --copies 3 \
  --data $S/tput/specbench_10.jsonl --out $SM/results.jsonl > $SM/bench.out 2>&1
kill -TERM -- -$SP 2>/dev/null; sleep 15; kill -KILL -- -$SP 2>/dev/null; sleep 5
COMB=$($PY - $SM/results.jsonl $SM/server.log <<'EOF'
import json, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1])]
except FileNotFoundError:
    rows = []
good = [r for r in rows if "error" not in r and r.get("n_fail") == 0 and r.get("accept_len")]
acc = sum(r["out_tokens"] for r in good) / sum(r["out_tokens"] / r["accept_len"] for r in good) if good else 0.0
sync = "Non-overlap (synchronous)" in open(sys.argv[2]).read()
ok = len(good) == 7 and acc > 2.1 and not sync
print(f"comb overlap smoke: categories ok {len(good)}/7, accept_len {acc:.3f}, overlap {'OFF' if sync else 'on'}"
      f" -> {'PASS' if ok else 'FAIL'}", file=sys.stderr)
print(1 if ok else 0)
EOF
)
log "COMB=$COMB"

# ---- 3) wait for an idle GPU pair ----
idle() {  # all GPUs in $1 (comma list) below 1000 MiB and running no compute process
  local g u uuid
  for g in $(echo $1 | tr , ' '); do
    u=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
    [ "$u" -lt 1000 ] || return 1
    uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid" && return 1
  done
  return 0
}
log "waiting for GPUs 0,1 or 2,3 to be idle for 5 consecutive minutes"
c01=0; c23=0; PAIR=""
while [ -z "$PAIR" ]; do
  idle 0,1 && c01=$((c01 + 1)) || c01=0
  idle 2,3 && c23=$((c23 + 1)) || c23=0
  if [ $c01 -ge 5 ]; then PAIR=0,1; elif [ $c23 -ge 5 ]; then PAIR=2,3; else sleep 60; fi
done
log "using GPUs $PAIR"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# ---- 4) AGENT.md eagle group, TP=2 ----
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$PAIR TP=2 PORT=30123 START_TIMEOUT=3600
export MODELS=/home/lrz/models PY COMB WORK=$A
if ! bash check_env.sh eagle > precheck_eagle.out 2>&1; then log "check_env.sh failed, see precheck_eagle.out"; exit 1; fi
log "precheck passed"

smoke_ok() {  # AGENT 5.3 pass criteria (+ comb rows when COMB=1); prints failures to stderr
  $PY - $A/smoke_eagle.jsonl $A/smoke_eagle.out $COMB <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
out = open(sys.argv[2]).read(); comb = sys.argv[3] == "1"
bad = []
if "EXPERIMENTS_DONE" not in out: bad.append("no EXPERIMENTS_DONE")
for t in sorted({r.get("tag", "?") for r in rows if "error" in r}):
    bad.append(f"error rows (server died) for {t}")
for t in ["ar", "chain_N8", "tree_N8"] + (["comb_N8"] if comb else []):  # 2 bs x 7 categories each
    k = sum(1 for r in rows if "error" not in r and r["tag"] == t)
    if k != 14: bad.append(f"{t} rows {k}/14")
for r in rows:
    if "error" in r: continue
    if r["n_fail"] or r["cached_tokens"] or r["tp_size"] != 2 or r["srv_max_running"] != 16:
        bad.append(f"bad row {r['tag']} bs{r['bs']} {r['category']}")
    if r["tag"].startswith("tree_N8") and (r["speculative_num_steps"], r["speculative_eagle_topk"], r["speculative_num_draft_tokens"]) != (7, 7, 8):
        bad.append("tree_N8 spec config mismatch")
for b in bad: print("SMOKE FAIL:", b, file=sys.stderr)
only_comb = bool(bad) and all("comb" in b for b in bad)
print("ok" if not bad else ("comb" if only_comb else "fail"))
EOF
}
run_smoke() {
  rm -f smoke_eagle.jsonl
  COPIES=1 BS_ALL=8,16 MIN_N=7 OUT=$A/smoke_eagle.jsonl bash run_experiments.sh eagle > smoke_eagle.out 2>&1
}
log "smoke (COMB=$COMB)"; run_smoke; R=$(smoke_ok)
if [ "$R" = comb ]; then
  mkdir -p invalid; mv smoke_eagle.jsonl invalid/smoke_eagle_comb_failed.jsonl; mv smoke_eagle.out invalid/smoke_eagle_comb_failed.out
  log "smoke failed only on comb rows -> COMB=0, redoing smoke"; COMB=0; export COMB
  run_smoke; R=$(smoke_ok)
fi
[ "$R" = ok ] || { log "smoke failed, see smoke_eagle.out / smoke_eagle.jsonl"; exit 1; }
log "smoke passed (COMB=$COMB); formal run"

BS_ALL=32,64,128,256 bash run_experiments.sh eagle > exp_eagle.out 2>&1
log "formal run finished: $(tail -1 exp_eagle.out)"
