#!/bin/bash
# Pre-flight check before any experiment. Prints OK/FAIL per item; exits 1 if any FAIL.
# Usage: TP=<n> [MODELS=dir] [PORT=30000] bash check_env.sh [eagle|mtp|all]
set -u
WORK=${WORK:-$(cd $(dirname $0) && pwd)}; MODELS=${MODELS:-$WORK/models}
[ -f $WORK/env.sh ] && source $WORK/env.sh
PY=${PY:-python}; TP=${TP:-1}; PORT=${PORT:-30000}; G=${1:-all}; FAIL=0
ok() { echo "OK   $*"; }; bad() { echo "FAIL $*"; FAIL=1; }; warn() { echo "WARN $*"; }

echo "== GPU"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || bad "nvidia-smi"
VIS=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}
NVIS=$(echo $VIS | tr , "\n" | grep -c .)
[ $NVIS -ge $TP ] && ok "visible GPUs ($VIS) >= TP=$TP" || bad "visible GPUs $NVIS < TP=$TP"
for g in $(echo $VIS | tr , " " | cut -d" " -f1-$TP); do
  used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  [ $used -lt 1000 ] && ok "GPU $g free (${used} MiB used)" || bad "GPU $g busy (${used} MiB used) - pick other GPUs via CUDA_VISIBLE_DEVICES"
done
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/     app: /'
echo "     topology (record in report; no NVLink -> TP all-reduce over PCIe is slower):"
nvidia-smi topo -m 2>/dev/null | head -12 | sed 's/^/     /'

echo "== Python / SGLang"
$PY - <<'EOF' || FAIL=1
import os, sys
try:
    import sglang, torch, transformers
except Exception as e:
    print("FAIL import:", e); sys.exit(1)
print(f"OK   python {sys.version.split()[0]} sglang {sglang.__version__} torch {torch.__version__} transformers {transformers.__version__} cuda {torch.version.cuda}")
if sglang.__version__ != "0.5.23": print("WARN sglang != 0.5.23 (reference version); record in report")
d = os.path.join(os.path.dirname(sglang.__file__), "srt", "models"); fail = 0
for f in ("qwen3_moe.py", "qwen3_5.py", "qwen3_5_mtp.py"):
    hit = os.path.exists(os.path.join(d, f)); fail |= not hit
    print(("OK  " if hit else "FAIL") + f" sglang model file {f}")
if torch.cuda.is_available(): print(f"OK   torch sees {torch.cuda.device_count()} GPU(s)")
else: print("FAIL torch.cuda unavailable"); fail = 1
sys.exit(fail)
EOF
$PY -m sglang.launch_server --help 2>/dev/null | grep -q -- "--cuda-graph-max-bs-decode" && ok "--cuda-graph-max-bs-decode supported" || bad "--cuda-graph-max-bs-decode missing"
command -v modelscope >/dev/null && ok "modelscope cli" || warn "modelscope cli missing (only needed for download)"

echo "== Files"
[ "$(md5sum $WORK/specbench.jsonl | cut -d' ' -f1)" = 63af56744be512743674731517f6e54b ] && ok "specbench.jsonl md5" || bad "specbench.jsonl md5 mismatch"
chk() { [ -f $MODELS/$1/config.json ] && [ -n "$(ls $MODELS/$1/*.safetensors 2>/dev/null)" ] && ok "model $1" || bad "model $1 missing under $MODELS"; }
case $G in eagle|all) chk Qwen3-30B-A3B-Instruct-2507; chk SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex;; esac
case $G in mtp|all) chk Qwen3.5-35B-A3B
  [ $TP -ge 2 ] || bad "mtp group needs TP>=2 on A100 80GB (weights ~72GB)"
  [ $TP -ge 8 ] && ok "mtp group TP=$TP" || warn "mtp group TP=$TP < 8: speculative SSM state at bs=512 (~67GB/GPU at TP=4) likely does not fit (OOM or capped concurrency)";; esac
case $G in eagle|all) [ $TP -ge 4 ] && ok "eagle group TP=$TP" || warn "eagle group TP=$TP < 4: BF16 KV at bs=512 is tight (TP=2 ~0.73M tokens vs need up to ~0.77M)";; esac
df -h $WORK $MODELS 2>/dev/null | sed 's/^/     /'

echo "== Port"
curl -s -m 2 localhost:$PORT/health >/dev/null && bad "port $PORT already serving - use PORT=<other>" || ok "port $PORT free"

[ $FAIL = 0 ] && echo "ALL CHECKS PASSED" || { echo "SOME CHECKS FAILED"; exit 1; }
