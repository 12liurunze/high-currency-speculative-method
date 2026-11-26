# AGENT.md — 8×A100 投机解码吞吐实验

本文件给在目标机器（8×A100）上执行实验的 agent（或人）使用。按顺序执行；标明“停止并报告”的情况不要自行绕过。

## 0. 目标

在 RedHatAI/speculator_benchmarks 的 7 个子集上，测量两组模型在不同并发 bs 下 AR / chain / tree / comb 的输出吞吐与接受长度；若某个 bs 下 tree 打不过 chain，则对该 bs 逐步缩小 budget，直到 tree 胜出（comb 在每个 budget 下与 chain、tree 一起测）。

| 组 | 目标模型（ModelScope） | 草稿 | SGLang 算法 |
|---|---|---|---|
| `eagle` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | `lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex`（外挂 EAGLE3） | `EAGLE3` |
| `mtp` | `Qwen/Qwen3.5-35B-A3B` | 模型自带 1 层 MTP（多步训练） | `NEXTN` |

## 1. 实验设计

| 项 | 取值 |
|---|---|
| 方法 | `ar`（不开投机）；`chain_N<N>`：steps=N-1, topk=1, draft=N；`tree_N<N>`：steps=N-1, topk=N-1, draft=N；`comb_N<N>`：steps=N-1, topk=1, draft=N，加环境变量 `SGLANG_COMB_K=4`。初始 N=8（chain 718，tree 778，comb 718+K4） |
| comb（HedgeTree） | drafter 与 chain 完全相同（沿 topk=1 主干跑 N-1 步，草稿计算量相同）；每步额外保留草稿分布的 top-K（K=4）个 token，top-1 作主干，其余作只当叶子、不再展开的兄弟节点；在 (N-1)×K 个候选中按累计草稿分数（路径概率乘积）取前 N-1 个，加根节点组成验证树，验证 token 数与 chain、tree 相同（都是 N）。只接受目标模型 greedy 认可的路径，输出与 chain 逐字一致。需要先给 SGLang 打补丁（第 4 节），由 `COMB=1`（`run_all.sh` / `run_experiments.sh`）开启，`COMB_K` 改 K（默认 4，不要改） |
| bs（并发上限） | 32, 64, 128, 256, 512，全部 BF16 KV |
| 每个 (子集, bs) 的请求数 | 子集**整份**复制 `3 × ceil(bs / n)` 份（n = 子集条数），不截断：bs ≤ n 时就是子集 × 3；bs > n 时按需加倍，保证至少 3 轮 bs 满并发。客户端 bs 个线程并发，完成一条补一条 |
| 请求数一览 | 80 条的子集：bs=32/64 → 240，128 → 480，256 → 960，512 → 1680；humaneval（164）：bs=32/64/128 → 492，256 → 984，512 → 1968 |
| 数据 | `specbench.jsonl`：humaneval 164、math_reasoning / qa / mt_bench / rag / summarization / translation 各 80，共 644 条 |
| 采样 | greedy（temperature=0），`max_new_tokens=256` |
| 上下文 | 两组都是 `--context-length 2048`（EAGLE3 草稿位置上限 2048；mtp 组对齐）。全部 prompt + 256 ≤ 2048，已用两种分词器核对 |
| 服务端通用参数 | `--tp-size $TP --mem-fraction-static 0.85 --max-running-requests <该轮最大bs> --cuda-graph-max-bs-decode <该轮最大bs> --disable-radix-cache`，其余均为 SGLang 默认值 |
| mtp 组对话模板 | `enable_thinking=False`（Qwen3.5 默认开思考，关掉以对齐 Instruct 非思考输出） |
| 重复 | 默认每个配置跑 1 遍（`REPS=1`，每个子集本身已复制 ≥3 份）；`REPS=3` 可改为独立重启服务跑 3 遍 |

### 1.1 缩小 budget 规则

- 判定口径：同一 (bs, N) 下，取 chain 和 tree 都有数据（无 `error`）的子集，合并算总吞吐（总输出 token / 总墙钟时间），多 rep 时取均值；`tree > chain` 才算 tree 胜出。comb 不参与判定，只在每个 (N, bs) 下与 chain、tree 一起测。
- 流程（`run_experiments.sh` 自动完成，判定由 `losing_bs.py` 执行，**不要手动挑 bs 或 N**）：
  1. N=8，全部 bs，跑 ar + chain_N8 + tree_N8 + comb_N8。
  2. 对 tree 未胜出的 bs，在 N=7 重跑 chain_N7 + tree_N7 + comb_N7（不再跑 ar）；仍未胜出的继续 N=6、5、4、3。某个 bs 一旦 tree 胜出就停止。
  3. 某个 bs 若没有任何可比子集（服务崩溃、tree 不支持），`losing_bs.py` 会在日志里打印 `cannot judge` 并**不再对它缩小 budget**（缩小 budget 解决不了这类问题），在报告中单列。
- 各 N 的配置（chain / tree / comb）：N=7 → 617 / 667 / 617+K4；N=6 → 516 / 556 / 516+K4；N=5 → 415 / 445 / 415+K4；N=4 → 314 / 334 / 314+K4；N=3 → 213 / 223 / 213+K4。

## 2. 硬件与 TP

**两组都用 `TP=8`、串行执行（先 eagle 后 mtp）。只需设置 `TP`，其余参数脚本自动处理。**

| 组 | 权重 | bs=512 时每卡显存（TP=8，MEM=0.85，粗估） | 为什么要 TP=8 |
|---|---|---|---|
| eagle | 约 61GB | 权重约 7.6GB；BF16 KV 约 59GB ≈ 240 万 token，需求最多约 77 万 | TP≥4 即可；TP=2 只有约 73 万 token，临界 |
| mtp | 约 72GB | 权重约 9GB；投机验证的线性注意力中间状态约 35GB（30 层 × 513 × draft 8 × 4 头 × 128 × 128 × fp32） | TP=4 时这块约 67GB/卡，放不下，bs=512 会 OOM 或被压低并发 |

- **同一组内所有方法、bs、budget 必须用同一个 TP 和同一个 MEM**；报告中写明。
- 不要两组各 4 卡并行：mtp 组在 TP=4 下 bs=512 放不下。
- bs=512 也用 BF16 KV（单卡时才需要 FP8）。如确需 FP8，设 `FP8_BS=512`（报告中注明）。
- 多卡结果与此前单卡 TP=1 的结果**不可直接比较**（TP 通信开销不同）。
- 机器若无 NVLink（`nvidia-smi topo -m` 显示 PIX/PHB/SYS），TP 的 all-reduce 走 PCIe，吞吐会偏低；不影响实验进行，记录在报告里。
- 上表是估算；服务端实际生效的并发上限见结果里的 `srv_max_running`（应等于该轮最大 bs）。

## 3. 文件清单

| 文件 | 作用 |
|---|---|
| `AGENT.md` | 本说明 |
| `check_env.sh` | 预检：GPU 空闲、TP、SGLang 版本与模型支持、模型文件、数据 md5、端口 |
| `dl_models.sh` | 从 ModelScope 下载 3 个模型到 `$MODELS` |
| `specbench.jsonl` | 数据（md5 `63af56744be512743674731517f6e54b`），**直接用，不要重新生成** |
| `prep_data.py` | 数据生成脚本（仅当数据文件丢失时用；需能访问 huggingface.co） |
| `bench.py` | 压测客户端（子集 × 3·ceil(bs/n) 份请求、bs 并发、记录时间窗与服务端配置、支持续跑） |
| `run_all.sh` | 单轮：起服务 → 压测 → 关服务（整个进程组，含所有 TP worker），依次跑 ar / chain / tree / comb（`COMB=1` 时） |
| `run_experiments.sh` | 完整流程：N=8 全部 bs → 自动缩小 budget → 汇总 |
| `losing_bs.py` | 判定某个 N 下 tree 未胜出的 bs |
| `summarize.py` | 汇总：吞吐（均值 ± 标准差）、接受长度、相对 AR / 相对 chain 的倍数 |
| `overrides_cu129.txt` | 参考环境的依赖约束（仅供参考） |
| `../spec_research/comb_patch.py` | 给 SGLang 打 comb 补丁（第 4 节）；`SGLANG_COMB_K` 未设置时 SGLang 行为不变 |

## 4. 环境

参考环境（已在 1×A100 80GB 上跑通）：Python 3.11.14，`sglang==0.5.23`，torch 2.14.1+cu130，transformers 5.19.0，驱动 570.195.03（CUDA 13.0）。

- 优先使用机器上已有的、满足要求的环境；否则在**独立 venv** 中安装 `sglang==0.5.23`（不要改系统 Python）。驱动的 CUDA 版本低于 torch 编译版本时，按 sglang 官方安装说明选对应 CUDA 版本的 wheel。
- 必须包含 `qwen3_5.py` / `qwen3_5_mtp.py`（`check_env.sh` 会检查）。sglang 版本不是 0.5.23 时可以继续，但必须写进报告。
- **comb 补丁**：对实验用的 SGLang 运行一次（只需一次，原文件备份为 `*.pre-comb`；已打过的修改会跳过，所以用旧版补丁打过的环境重新运行一次即可补上新增的修改，原始备份保留，然后重启服务；旧版补丁下测得的 mtp 组 `comb_N*` 结果无效，先移到 `invalid/` 再测，不要用 `RESUME=1` 续在上面）：
  ```bash
  SRT=$(${PY:-python} -c 'import sglang, os; print(os.path.dirname(sglang.__file__))')/srt
  ${PY:-python} ../spec_research/comb_patch.py $SRT
  grep -c SGLANG_COMB_K $SRT/speculative/eagle_worker_v2.py $SRT/layers/attention/hybrid_linear_attn_backend.py $SRT/mem_cache/memory_pool.py   # 每个都应 ≥ 1
  grep -c comb_k $SRT/speculative/eagle_draft_cuda_graph_runner.py   # 应 ≥ 1
  ```
  补丁修改 `speculative/eagle_worker_v2.py`、`speculative/eagle_draft_cuda_graph_runner.py`、`layers/attention/hybrid_linear_attn_backend.py` 和 `mem_cache/memory_pool.py`（后两个让 Qwen3.5 的线性注意力层按树验证，缺少时 mtp 组 comb 的输出是错的）；每处修改要求原文件中恰好匹配一次，报 `expected 1 match` 说明 SGLang 版本不同，停止并报告，不要手动改代码。CUDA graph 与 overlap 调度都可正常开启（`comb_patch.py` 开头注释里要求关闭二者是早期版本的限制，已过时）。
- 代理、镜像、`LD_LIBRARY_PATH` 等机器相关的环境变量写进本目录的 `env.sh`（各脚本会自动 `source`）；用的不是默认 `python` 时设置 `PY=/path/to/venv/bin/python`。

## 5. 执行步骤

所有长任务用 `nohup ... &` 后台运行并写日志，SSH 断开不影响。

```bash
cd <本目录>; export WORK=$(pwd); export MODELS=${MODELS:-$WORK/models}
```

### 5.1 下载模型（约 135GB）

```bash
nohup bash dl_models.sh > dl_models.log 2>&1 &
```

完成标志：`dl_models.log` 出现 `MS_DONE`。

### 5.2 预检（不通过不得继续）

```bash
TP=8 bash check_env.sh all
```

必须输出 `ALL CHECKS PASSED`。GPU 被他人占用 → 用 `CUDA_VISIBLE_DEVICES` 换卡或等待，**不得 kill 他人进程**。

### 5.3 冒烟测试（每组各一次，约 15–25 分钟/组）

用很小的 bs、每个子集只发 1 份，走完整流程（含一轮缩小 budget）：

```bash
TP=8 COMB=1 COPIES=1 BS_ALL=8,16 MIN_N=7 OUT=$WORK/smoke_eagle.jsonl bash run_experiments.sh eagle > smoke_eagle.out 2>&1
TP=8 COMB=1 COPIES=1 BS_ALL=8,16 MIN_N=7 OUT=$WORK/smoke_mtp.jsonl bash run_experiments.sh mtp > smoke_mtp.out 2>&1
```

通过标准：
- `smoke_<组>.out` 末尾有 `EXPERIMENTS_DONE`，没有 `server died`；
- `smoke_<组>.jsonl` 中 N=8 有 4 方法 × 2 bs × 7 子集 = 56 行，无 `error`，`n_fail` 全为 0；
- `speculative_*` 字段与配置一致（tree_N8 为 7/7/8，chain_N8 与 comb_N8 都是 7/1/8），comb_N8 的 `accept_len` 高于同 bs、同子集的 chain_N8，`tp_size` 为 8，`srv_max_running` 为 16，`cached_tokens` 全为 0。

不通过 → 看第 8 节，仍无法解决则停止并报告。冒烟结果单独交付，不并入正式结果。

### 5.4 正式实验

串行：

```bash
TP=8 COMB=1 nohup bash run_experiments.sh eagle > exp_eagle.out 2>&1 &
```

等 `exp_eagle.out` 出现 `EXPERIMENTS_DONE eagle` 后：

```bash
TP=8 COMB=1 nohup bash run_experiments.sh mtp > exp_mtp.out 2>&1 &
```

进度：
- `grep -c '"tag"' results_<组>.jsonl`：N=8 阶段每组 4 方法 × 5 bs × 7 子集 = **140 行**（×REPS）；缩小 budget 每多测一个 (N, bs) 加 3 × 7 = 21 行（chain、tree、comb）。
- `grep -E "^==|does not beat|cannot judge|final" exp_<组>.out`：当前阶段与缩小 budget 的判定。

耗时（粗估）：每次起服务 3–5 分钟；N=8 阶段每组 4 次起服务，每个方法约 2.6 万条请求、约 470 万输出 token。单组 1–4 小时，加上缩小 budget 每轮 0.5–2 小时、最多 5 轮。

### 5.5 汇总

`run_experiments.sh` 结束时已生成 `summary_<组>.txt`（总表 + 分子集表）；comb_N<N> 行同时给出相对 AR 和相对同 N chain 的倍数。手动查看：

```bash
python summarize.py results_eagle.jsonl
python summarize.py results_eagle.jsonl --by-cat
```

## 6. 结果记录格式

`results_<组>.jsonl` 每行一个 (方法, rep, bs, 子集)：

| 字段 | 含义 |
|---|---|
| `tag` / `rep` / `bs` / `category` | 方法（`ar` / `chain_N<N>` / `tree_N<N>` / `comb_N<N>`）/ 第几遍 / 并发上限 / 子集 |
| `n_req` / `copies` / `n_fail` | 实际请求数 / 子集复制份数（= 3·ceil(bs/n)）/ 被拒请求数（应为 0） |
| `e2e_s` / `out_tokens` / `throughput_tok_s` | 墙钟时间 / 输出 token 总数 / 吞吐 |
| `accept_len` | 平均接受长度（AR 为 null） |
| `cached_tokens` | 前缀缓存命中的 token 数，应全为 0（radix cache 已关闭，重复的 prompt 也必须重新 prefill）；不为 0 则停止并报告 |
| `kv_need` / `kv_cap` / `kv_ok` | 最坏情况 KV 需求 / 服务端 KV 容量 / 是否放得下（参考；TP=8 下应全为 true） |
| `srv_max_running` | 服务端实际生效的最大并发上限（小于 bs 说明服务端压低了并发，需在报告中注明） |
| `t0` / `t1` / `server_log` | 测量时间窗与对应服务端日志 |
| `speculative_*` / `kv_cache_dtype` / `tp_size` | 服务端实际生效的配置（用来核对；comb 与 chain 的 `speculative_*` 相同，靠 `tag` 区分） |

服务端日志：`logs/server_<组>_<tag>_rep<r>_bs<该轮最大bs>_<bf16|fp8>.log`，互不覆盖。

## 7. 正式实验中的判读

- `exp_<组>.out` 出现 `cannot judge bs [...]`：这些 bs 没有可比数据（服务崩溃或 tree 不支持）。查结果里的 `error` 行和对应 server 日志，在报告中说明原因。
- comb_N<N> 的 `accept_len` 与同 N 的 chain_N<N> 几乎相同：说明 `SGLANG_COMB_K` 没有生效（补丁未打到实际 import 的 SGLang 上），停止并报告；正常情况下 comb 的接受长度高于 chain、略低于 tree。
- `kv_ok=false` 或 `srv_max_running` 小于 bs：服务端可能无法同时跑满 bs 条，结果照常保留，在报告中列出这些组合。

## 8. 已知问题与处理

| 现象 | 原因 | 处理 |
|---|---|---|
| 服务端日志里有 `Init Unified Radix Cache` / `Tree cache initialized` | SGLang 0.5.23 总会创建该缓存对象；`--disable-radix-cache` 生效时前缀匹配直接返回空 | 无需处理，以结果里 `cached_tokens` 全为 0 为准 |
| 一轮压测结束后出现 `run_all.sh: line NN: <pid> Killed  setsid ... sglang.launch_server ...` | 正常关闭：进程组收到 SIGTERM 后，SGLang 主进程清理子进程并以 SIGKILL 结束自己 | 无需处理。真正的失败会在结果文件里写 `{"error":"server died ..."}` |
| 起服务时 `CUDA out of memory`（多在 Capture cuda graph 阶段） | MEM 过高，CUDA graph / 树形草稿 / mtp 投机中间状态没空间（单卡 bs=512 时出现过） | 该组改 `MEM=0.80` 从头重跑**该组全部实验**（同组必须同一 MEM），报告中注明 |
| `{"error":"server died ..."}` | 服务崩溃 | 看对应 server 日志末尾 Traceback；OOM 按上一行处理；其他错误记录原文后停止并报告 |
| `port 30000 busy` | 上一个服务没退出，或他人占用 | 只处理自己启动的进程；他人占用则换端口：`PORT=30010` |
| mtp 组 tree 启动或运行时报错，提及 topk / GDN / mamba / `retrieve_parent_token` | 混合线性注意力的树形验证路径不支持当前配置（0.5.23 已在 1×A100 上实测 Qwen3.5-35B-A3B 的 tree_N8 / tree_N3 可用，bs=1；更大 bs、TP=8 未测） | **不要改代码绕过**；记录报错原文。`losing_bs.py` 会对这些 bs 报 `cannot judge` 而不再缩小 budget；mtp 组报告 ar 与 chain |
| mtp 组 `srv_max_running` 小于 bs | 混合模型的 mamba 状态缓存限制了并发 | 记录日志开头的 `max_running_requests` 相关行，报告；不要改 mamba 相关参数 |
| comb 服务启动时 `RuntimeError: comb (SGLANG_COMB_K=...) verifies a tree ... target attention backend [...]` | comb 以 topk=1 启动却验证一棵树，只有 FlashInfer 后端在 topk=1 时仍按树掩码验证；fa3（H100 等 Hopper 卡上 topk=1 时的默认后端）不读树掩码，验证结果是错的 | 服务端命令加 `--attention-backend flashinfer`（`run_all.sh` 可通过 `EXTRA` 传入），报告中注明；不要改补丁绕过检查 |
| comb 服务启动时 `AssertionError: comb drafts a chain` 或 `comb is greedy-only` | comb 只支持 `--speculative-eagle-topk 1` 和 greedy | 由脚本自动设置，正常不会出现；出现说明命令被改动，恢复原命令 |
| mtp 组 comb 启动或运行时报错 | comb 走树形验证路径，与上面 mtp 组 tree 的限制相同（四个文件都打上补丁后 mtp 冒烟已通过；若 comb 的 `accept_len` 低于 chain，或输出 token 数明显多于 chain，先按第 4 节检查补丁是否完整） | 同 mtp 组 tree 的处理：**不要改代码绕过**，记录报错原文，报告中注明；不影响缩小 budget 的判定 |
| ModelScope 下载中断 | 网络 | 脚本自带 3 次重试；仍失败则重跑 `dl_models.sh`（续传） |
| 服务停不掉 / GPU 显存未释放 | TP worker 残留 | `run_all.sh` 按进程组停止并等待退出；仍有残留时只 kill 自己启动的进程（`ps -eo pid,pgid,args | grep sglang` 核对） |

## 9. 规则

- 只在本目录及 `$MODELS` 下读写；不改系统配置、驱动、全局 Python 环境。
- 不 kill 非自己启动的进程；不占用他人正在使用的 GPU。
- 不修改实验参数（方法配置、bs、请求数、MAXNEW、数据、上下文、TP、MEM）；唯一允许的调整是第 8 节列出的 MEM / PORT，以及用 `CUDA_VISIBLE_DEVICES` 选卡，且必须在报告中注明。
- 中途崩溃（服务挂掉、机器重启、SSH 断开导致进程被杀）：先查明原因并处理，然后用**完全相同的命令**加 `RESUME=1` 重跑，已完成的 (方法, rep, bs, 子集) 会被跳过，其余照常测（每个单元格在一次服务会话内独立完成，续跑不影响可比性）。`error` 行不会被跳过，会重测。
- 若要改变任何实验参数（例如按第 8 节改了 MEM），不能续跑：把该组旧结果移到 `invalid/` 并说明原因，从头重跑该组。
- 两组串行执行，不要同时跑。

## 10. 交付

1. `results_eagle.jsonl`、`results_mtp.jsonl`、`exp_eagle.out`、`exp_mtp.out`、`summary_eagle.txt`、`summary_mtp.txt`，以及冒烟测试的 `smoke_*.jsonl` / `smoke_*.out`。
2. `logs/` 目录打包。
3. 每组每个 bs 的结论：tree 首次胜出的 N 及该 N 下 tree/chain 吞吐比；或“N=8–3 均未胜出”；或“无法判定”及原因。另给出每个 (N, bs) 下 comb 相对 chain、tree、AR 的吞吐比和接受长度。
4. 所有 `error`、`n_fail>0`、`kv_ok=false`、`srv_max_running` 小于 bs 的组合列表。
5. 简短报告：硬件（GPU 型号、数量、显存、`nvidia-smi topo -m`）、软件版本、使用的 TP / MEM / PORT / CUDA_VISIBLE_DEVICES、所有偏离本文件的操作及原因。
