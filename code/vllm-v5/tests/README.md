# v5 测试与性能对比说明

目标能力（当前树）：

- Dense：Qwen3-0.6B（回归引擎 / FlashAttention / 调度 / 并行切分）
- MoE：Qwen1.5-MoE-A2.7B（TP / EP）、Qwen3-30B-A3B（大模型冒烟）
- 不含 VL / OpenAI / DeepEP

---

## 一、应该测什么

### A. 正确性（`tests/`）

1. `unit` — 无 GPU：routing / backend / EP map / EngineConfig MoE 开关
2. `gpu` — 单卡：dense greedy；fused vs naive；A2.7B greedy 冒烟
3. `distributed` — ≥2 同型号卡：
   - **fp32 golden**：`test_dense_parallel.py` 用单卡 fp32 输出当参考，
     `tp=2` / `pp=2` 逐 token 一致（TP = NCCL all-reduce 切分，PP = NCCL P2P 切层）
   - **MoE 并行**：`test_moe_parallel.py`，`tp=1` / `tp=2` / `tp=2+ep` 贪心 token 一致

### B. MoE 专项（正确性 + 性能）

| 项 | 目的 | 入口 |
|---|---|---|
| fused vs naive | Triton/cutlass 数值 | `tests/gpu/test_moe_kernel.py` |
| EP map 分片求和 | expert_map 切分语义 | 同上 |
| tp / ep 对齐 | 并行不改 greedy 结果 | `tests/distributed/test_moe_parallel.py` |
| backend 选择 | auto→cutlass\\|triton 规则 | `tests/unit` |
| TTFT / TPOT / tok/s | 端到端延迟吞吐 | `bench/moe_e2e.py` |
| vs 官方 vLLM | 同负载对比延迟/吞吐/token | `bench/vs_vllm.py` |

### C. 性能对比口径（与官方 vLLM 对齐）

- greedy + `ignore_eos`，固定 prompt / `max_tokens` / 模型 / dtype
- 指标：**TTFT (ms)、TPOT (ms)、decode tok/s、throughput tok/s**
- 单请求（`--concurrency 1`）与小 batch（`--concurrency 8`）分开报
- 同 GPU、同 dtype、同 `max_model_len`；对比 CG 时两边都开 `--cg` / 关 eager

---

## 二、如何跑

```bash
cd vllm-v5
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.
# 建议避开默认 cuda:0，用物理卡号
export CUDA_VISIBLE_DEVICES=1   # 例：3090

# 正确性
pytest tests/unit -q
pytest tests/gpu -q --tb=short -m 'not slow'
# 多卡单独跑（避免与其它 GPU 作业争用导致 NCCL 挂起）
pytest tests/distributed -q --tb=short

# 只跑 fp32 golden（tp=2 / pp=2）
pytest tests/distributed/test_dense_parallel.py -q --tb=short

# 端到端性能（本引擎）
PYTHONPATH=. python bench/moe_e2e.py --model dense --devices 0 --cg

# 单卡档位 sweep（eager / cg / cg+compile，直接出「vs eager ×」）
PYTHONPATH=. python bench/single_card_sweep.py --device 0

# 与官方 vLLM 对比（需另装官方包，见 bench/README.md）
PYTHONPATH=. python bench/vs_vllm.py --model dense --devices 0 --vllm-python <venv>/bin/python
PYTHONPATH=. python bench/vs_vllm.py --model dense --devices 0,1,2,3 --tp 4 \
    --concurrency 8 --vllm-python <venv>/bin/python
PYTHONPATH=. python bench/vs_vllm.py --model moe --devices 0,1 --tp 2 --ep \
    --vllm-python <venv>/bin/python
```

模型默认路径：`~/huggingface/{Qwen3-0.6B, Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B}`。

一键跑完整 4090 矩阵：`bash bench/run_4090_matrix.sh`（见 `bench/README.md`）。
