"""Dense TP/PP golden: fp32 single-GPU reference must match tp=2 / pp=2 token-for-token.

需要 >=2 张同型号卡。参考实现 = **单卡 fp32**，被测 = tp=2 / pp=2；两边都跑
greedy + ignore_eos，**逐 token 完全一致**才算切分正确。

跑法（在 vllm-v5/ 下）::

    export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.
    pytest tests/distributed/test_dense_parallel.py -q --tb=short
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import needs_2gpu, needs_cuda
from tests.helpers import DENSE_MODEL, pick_gpus, require_model
from tests.helpers.devices import run_isolated

pytestmark = [
    pytest.mark.distributed,
    needs_cuda,
    needs_2gpu,
    require_model(DENSE_MODEL),
]

PROMPT = "The capital of France is"
N = 16

_WORKER = r"""
import json, sys
from tests.helpers.engine import cfg, greedy_tokens
from tests.helpers.paths import DENSE_MODEL

prompt, n_s, tp_s, pp_s, dtype, port_s, gmu_s = sys.argv[1:8]
n, tp, pp, port = int(n_s), int(tp_s), int(pp_s), int(port_s)
world = tp * pp
tokens = greedy_tokens(
    cfg(
        DENSE_MODEL,
        device_ids=list(range(world)),
        tp_size=tp,
        pp_size=pp,
        dtype=dtype,
        max_num_seqs=1,
        context_len=512,
        num_kvcache_blocks=16,
        gpu_memory_utilization=float(gmu_s),
        nccl_port=port,
        prefix_backend="none",
    ),
    prompt,
    n=n,
)
print(json.dumps({"tp": tp, "pp": pp, "dtype": dtype, "tokens": tokens}))
"""


def _tokens(phys: list[int], *, tp: int, pp: int, port: int, dtype: str = "float32") -> list[int]:
    out = run_isolated(
        phys,
        _WORKER,
        PROMPT,
        str(N),
        str(tp),
        str(pp),
        dtype,
        str(port),
        "0.70",
        timeout=1800,
    )
    lines = [ln for ln in out.splitlines() if ln.strip().startswith("{")]
    assert lines, out
    return json.loads(lines[-1])["tokens"]


@pytest.fixture(scope="module")
def two_gpus() -> list[int]:
    # 4090 优先；没有就退到任意两张同型号空闲卡。
    try:
        return pick_gpus(2, prefer="4090", min_free_mib=6000)
    except RuntimeError:
        return pick_gpus(2, prefer="3090", min_free_mib=6000)


@pytest.fixture(scope="module")
def fp32_reference(two_gpus: list[int]) -> list[int]:
    """单卡 fp32 golden（只跑一次，供 tp2 / pp2 共用）。"""
    return _tokens(two_gpus[:1], tp=1, pp=1, port=29710)


def test_tp2_matches_fp32_reference(two_gpus: list[int], fp32_reference: list[int]) -> None:
    """tp=2（NCCL all-reduce 切分）与单卡 fp32 参考逐 token 一致。"""
    tp2 = _tokens(two_gpus[:2], tp=2, pp=1, port=29712)
    assert tp2 == fp32_reference, (
        f"tp=2 与 fp32 参考不一致\n ref={fp32_reference}\n tp2={tp2}"
    )


def test_pp2_matches_fp32_reference(two_gpus: list[int], fp32_reference: list[int]) -> None:
    """pp=2（NCCL P2P 切层）与单卡 fp32 参考逐 token 一致。"""
    pp2 = _tokens(two_gpus[:2], tp=1, pp=2, port=29714)
    assert pp2 == fp32_reference, (
        f"pp=2 与 fp32 参考不一致\n ref={fp32_reference}\n pp2={pp2}"
    )
