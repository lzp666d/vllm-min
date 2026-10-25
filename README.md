# vLLM-min 增量：chunked prefill、并行正确性验证与性能剖析

我在一个 vLLM-min 教学实现上做的扩展与验证。仓库里是我**改动过的文件的完整快照**（`code/`）
加上对应的提交补丁（`patches/`），不含上游未改动部分，所以不能独立运行——需要在工作区里对照。

## 1. 四个增量

| # | 提交 | 内容 |
|---|---|---|
| 1 | `feat(v3)` | 可配置的 chunked prefill：新增 `max_prefill_chunk` 配置与 `--chunked-prefill-size` 入口；`_pull_prefill_chunk` 支持单步只推进一个 partial chunk，prompt 超长时归为「长请求」并单独限流；配套 139 行调度测试 |
| 2 | `bench(v5)` | 单卡参数 sweep（`single_card_sweep.py`）+ 4×3090 矩阵 harness（`run_4090_matrix.sh`），统一 jsonl 结果格式 |
| 3 | `test(v5)` | tp=2 / pp=2 正确性校验：以单卡 fp32 输出为 golden，逐 token 比对，全 PASS |
| 4 | `docs(v5)` | 验证矩阵：把「简历里的每句话」映射到 代码位置 / 命令 / 产出指标 |

## 2. 代码导读

`code/vllm-v3/src/control/scheduler.py` 是这次改动的核心：

- `Scheduler.__init__(max_prefill_chunk=0)`：`0` 表示不主动切分，只受 `max_num_batched_tokens` 约束（原有行为）
- `_pull_prefill_chunk(...)`：从 waiting 队列拉 prefill，单步 chunk 大小逐层取 min —
  `need`（剩余 token）、`allowed`（本步预算）、`max_prefill_chunk`（每请求上限），
  下限取 `block_size` 避免切出比一页还碎的 chunk
- `is_partial = chunk < need`：标记「半截 chunk」，长请求（prompt 本身超过阈值）走单独的上限
- `n_partial`：本步产生的 partial chunk 数；`_schedule_mixed` 先 decode 后 prefill，
  保证 decode 的 TPOT 不被 prefill 抢占

`code/vllm-v3/src/control/engine.py`、`config.py`、`run.py` 是把 `max_prefill_chunk` 从 CLI 一路传到调度器的链路。
`code/vllm-v5/bench/vs_vllm.py` 是与官方 vLLM 的对照压测，`moe_e2e.py` 是 MoE 端到端。

## 3. 关键结果

环境：RTX 4090 D 24 GB（单卡）、4× RTX 3090 24 GB（无 NVLink，P2P 不可用）；
Qwen3-0.6B / Qwen3-8B；PyTorch 2.13、FlashAttention-2 2.8.3、Triton 3.7.1。
统一口径：prompt 256、greedy（temperature=0）、max_tokens 128。

**单卡 0.6B**

| 指标 | Eager | CUDA Graph | 提升 |
|---|---|---|---|
| 单流 Decode (tok/s) | 65.3 | 433.9 | 6.6× |
| batch-8 聚合 (tok/s) | 487.3 | 2734.4 | 5.6× |
| TTFT (ms) | 19.43 | 4.78（+prefill torch.compile） | 4.1× |

**4 卡 8B（Qwen3-8B，TP=4 + Decode CUDA Graph）**

| 指标 | 单卡 CUDA Graph | TP=4 CUDA Graph |
|---|---|---|
| 单流 Decode (tok/s) | 50.2 | 105.4（2.10×） |
| TTFT (ms) | — | 86.6 |

eager 下 TP=4 是负收益（0.74×，每层一次 all-reduce 的 launch 开销吃掉收益），
开 Decode CUDA Graph 把 all-reduce 收进图之后才转正 —— 这是这次验证的主要结论。

**正确性**：tp=2 / pp=2 切分后与单卡 fp32 逐 token 一致。

## 4. 目录

```
code/vllm-v3/   调度器、Engine、配置、CLI 入口、chunked prefill 测试
code/vllm-v5/   bench（单卡 sweep / 4090 矩阵 / vs 官方 vLLM / MoE e2e）、分布式正确性测试、验证文档
patches/        4 个提交的 patch（参考用；里面的本地 venv 路径已改写）
```

## 5. 运行

需要完整的 vLLM-min 工作区。单卡 sweep：

```bash
cd vllm-v5 && export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.
python bench/single_card_sweep.py --model Qwen3-0.6B --prompt-len 256 --max-tokens 128
```

4 卡矩阵：

```bash
bash bench/run_4090_matrix.sh
```
