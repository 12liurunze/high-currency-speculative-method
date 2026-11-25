# eagle group on this A6000 box: GPUs 2,3 (same NUMA node), TP=2, bs 32-256; the guard holds GPUs 2,3 between servers
cd /home/lrz/AI_infra/a100_pkg
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 TP=2 PORT=30123 START_TIMEOUT=3600
export MODELS=/home/lrz/models PY=/home/lrz/AI_infra/envs/sglang-spec/bin/python BS_ALL=32,64,128,256
export PRE_LAUNCH="bash /home/lrz/gpu_hold/guard_ctl.sh release" POST_STOP="bash /home/lrz/gpu_hold/guard_ctl.sh grab"
bash run_experiments.sh eagle
