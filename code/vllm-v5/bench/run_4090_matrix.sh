#!/usr/bin/env bash
# 4x RTX 4090 一键复现：并行正确性 + 单卡档位 sweep + 与官方 vLLM 对比。
#
# 用法:
#   cd vllm-v5
#   bash bench/run_4090_matrix.sh
#
# 可覆盖的环境变量:
#   PY       本项目解释器（默认 python，建议 python）
#   VLLM_PY  装了官方 vLLM 的解释器（默认空 → 跳过对比，建议 venvs/vllm/bin/python）
#   OUT      日志路径（默认 /tmp/vllmmin_4090_<时间戳>.log）
#   N        每次请求输出 token 数（默认 64）
#   CONC     并发数（默认 8）
#
# 例:
#   PY=python \
#   VLLM_PY=/root/autodl-tmp/venvs/vllm/bin/python \
#   bash bench/run_4090_matrix.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
VLLM_PY="${VLLM_PY:-}"
OUT="${OUT:-/tmp/vllmmin_4090_$(date +%Y%m%d_%H%M%S).log}"
N="${N:-64}"
CONC="${CONC:-8}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=.

run() { echo; echo "===== $* ====="; "$@" 2>&1 | tee -a "$OUT"; }

: >"$OUT"
echo "interpreter : $PY" | tee -a "$OUT"
echo "vllm python : ${VLLM_PY:-<unset>}" | tee -a "$OUT"
echo "log         : $OUT" | tee -a "$OUT"

# --- 0. 环境自检 ---
run "$PY" -c "import torch, triton, flash_attn, flashinfer; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'gpus')"

# --- 1. 正确性 ---
# unit 无 GPU 也能过；gpu / distributed 需要卡。
run "$PY" -m pytest tests/unit -q
run "$PY" -m pytest tests/gpu -q --tb=short -m 'not slow'
# fp32 golden：tp=2 / pp=2 逐 token 一致（需 >=2 张同型号卡）
run "$PY" -m pytest tests/distributed/test_dense_parallel.py -q --tb=short
# MoE tp=2 / tp2+ep 与 tp=1 一致（需 Qwen1.5-MoE-A2.7B）
run "$PY" -m pytest tests/distributed/test_moe_parallel.py -q --tb=short

# --- 2. 单卡 4090: Qwen3-0.6B  eager / cg / cg+compile ---
# 逐张卡各扫一遍，方便挑空闲卡，也便于看卡间波动。
for d in 0 1 2 3; do
  run "$PY" bench/single_card_sweep.py --device "$d" --model dense --n "$N"
done

# --- 3. 与官方 vLLM 对比: TP=1 / TP=4，单并发 + 并发批 ---
if [[ -n "$VLLM_PY" ]]; then
  for c in 1 "$CONC"; do
    run "$PY" bench/vs_vllm.py --model dense --devices 0 --tp 1 \
        --concurrency "$c" --n "$N" --vllm-python "$VLLM_PY"
    run "$PY" bench/vs_vllm.py --model dense --devices 0,1,2,3 --tp 4 \
        --concurrency "$c" --n "$N" --vllm-python "$VLLM_PY"
  done
  # MoE（Qwen1.5-MoE-A2.7B）TP=2 + EP
  run "$PY" bench/vs_vllm.py --model moe --devices 0,1 --tp 2 --ep \
      --n "$N" --vllm-python "$VLLM_PY"
else
  echo "[skip] 官方 vLLM 对比：设 VLLM_PY=<venv>/bin/python 后重跑" | tee -a "$OUT"
fi

echo; echo "done -> $OUT" | tee -a "$OUT"
