# bench — 本引擎延迟/吞吐 与 官方 vLLM 对比

三个脚本 + 一个总编排：

| 脚本 | 作用 |
|---|---|
| `moe_e2e.py` | 本引擎单点：TTFT / TPOT / decode tok/s / e2e tok/s，`--json` 便于汇总 |
| `single_card_sweep.py` | 单卡扫 eager / cg / cg+compile，出「**vs eager ×**」加速比表 |
| `vs_vllm.py` | 与官方 vLLM 同负载对比（含并发吞吐 `throughput_tok_s` 与 TPOT） |
| `run_4090_matrix.sh` | 4×4090 一键：正确性 + 单卡 sweep + vs vLLM |

---

## 装官方 vLLM（**独立** venv，别混进本项目的 Torch/FlashInfer 栈）

```bash
uv venv ~/venvs/vllm-official
uv pip install vllm --python ~/venvs/vllm-official/bin/python
```

AutoDL 上本仓库现成的等价环境是 `/root/autodl-tmp/venvs/vllm`（vLLM 0.26.0），
可直接把它当 `--vllm-python`。

---

## 跑法

```bash
cd vllm-v5
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.

# 1) 本引擎单点
python bench/moe_e2e.py --model dense --devices 0 --cg
python bench/moe_e2e.py --model moe --devices 0,1 --tp 2 --ep --eager

# 2) 单卡档位 sweep（Qwen3-0.6B）：tok/s 与 vs eager 加速比
python bench/single_card_sweep.py --device 0

# 3) 与官方 vLLM 对比
#    单请求
python bench/vs_vllm.py --model dense --devices 0 \
    --vllm-python /root/autodl-tmp/venvs/vllm/bin/python
#    4 卡 TP=4 + 8 并发（吞吐）
python bench/vs_vllm.py --model dense --devices 0,1,2,3 --tp 4 --concurrency 8 \
    --vllm-python /root/autodl-tmp/venvs/vllm/bin/python
#    MoE TP=2 + EP
python bench/vs_vllm.py --model moe --devices 0,1 --tp 2 --ep \
    --vllm-python /root/autodl-tmp/venvs/vllm/bin/python

# 4) 一把梭
PY=python \
VLLM_PY=/root/autodl-tmp/venvs/vllm/bin/python \
bash bench/run_4090_matrix.sh
```

---

## 指标口径

两个引擎都用同一组 prompt / `max_tokens` / greedy / `ignore_eos`：

```
ttft_ms            首 token 延迟（平均）
tpot_ms            每输出 token 时间 = (e2e - ttft) / (n - 1)
decode_tok_s       单请求 decode 速度 = (n - 1) / (e2e - ttft)
throughput_tok_s   并发批总吞吐 = 批内总输出 token / 墙钟
tokens             逐 token 输出（正确性 spot-check）
```

> 注：官方 `LLM.generate` 不暴露流式首 token，TTFT 用同批 `max_tokens=1` 的墙钟近似；
> 本引擎是真实首 token 时间。对比时留意这个口径差异（vLLM 侧 TTFT 偏保守）。

MoE 矩阵（TP / EP / CUDA Graph）另有 `run_moe_matrix.sh`。
