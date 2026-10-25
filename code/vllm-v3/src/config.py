from __future__ import annotations

import os
from dataclasses import dataclass, field

from transformers import AutoConfig, PretrainedConfig


@dataclass
class EngineConfig:
    """引擎配置：调度器 / 内存 / 算子 / 采样参数。"""

    #  模型 
    model: str  # 本地模型路径

    #  调度器 
    context_len: int = 4096
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    mix_prefill_decode: bool = True  # True=decode优先混合调度，False=全prefill优先
    # chunked prefill（参数语义对齐 vLLM SchedulerConfig）
    # vLLM 对应: long_prefill_token_threshold / max_num_partial_prefills / max_long_partial_prefills
    max_prefill_chunk: int = 0  # 单个 prefill chunk 的 token 上限；0=不切分（仅受 batch 预算约束）
    max_num_partial_prefills: int = 1  # 每步最多允许几个未完成 prefill
    max_long_partial_prefills: int = 1  # 其中长请求(prompt>max_prefill_chunk)的未完成 prefill 上限
    # Sarathi 式"每步 prefill 总配额"（>0 启用）：本步所有 prefill chunk 的 token 之和上限。
    # 与 max_prefill_chunk 的区别：后者限制"单个请求"，前者限制"整步的总工作量"。
    # >0 时多个长请求可在同一步各推进一片，且不再受半截数量上限约束。
    max_prefill_tokens_per_step: int = 0

    #  内存 
    kvcache_block_size: int = 256  # 必须是256的倍数（FA2 paged 约束）
    gpu_memory_utilization: float = 0.9
    num_kvcache_blocks: int = 0  # 0=自动按显存计算，>0=固定块数(基准测试用)
    prefix_backend: str = "hash"  # 前缀缓存后端: none / hash / radix

    #  算子 
    enforce_eager: bool = True  # False=启用 Decode CUDA Graph
    torch_compile: bool = False  # True=prefill 首 chunk 走 torch.compile（全图编译，无 piecewise/静态 buffer copy）
    compile_mode: str = "reduce-overhead"  # torch.compile mode: default / reduce-overhead / max-autotune
    compile_dynamic: bool = True  # True=符号形状（一次编译覆盖任意长度，避免按长度重编译）；False=固定形状重放（仅离线固定长度，新长度会重编译 ~2-12s）
    dtype: str = "auto"

    #  运行时 
    hf_config: PretrainedConfig = field(init=False)

    def __post_init__(self) -> None:
        self.model = os.path.expanduser(self.model)
        if not os.path.isdir(self.model):
            raise FileNotFoundError(
                f"Model path not found: {self.model}. "
                "Use a local directory (e.g. ~/huggingface/Qwen3-0.6B)."
            )
        if self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be a multiple of 256 (FA2 paged)")
        if self.prefix_backend not in ("none", "hash", "radix"):
            raise ValueError(
                f"prefix_backend must be none|hash|radix, got {self.prefix_backend!r}"
            )
        if self.max_prefill_chunk < 0:
            raise ValueError("max_prefill_chunk must be >= 0")
        if self.max_num_partial_prefills <= 0:
            raise ValueError("max_num_partial_prefills must be > 0")
        if self.max_long_partial_prefills <= 0:
            raise ValueError("max_long_partial_prefills must be > 0")
        if self.max_prefill_tokens_per_step < 0:
            raise ValueError("max_prefill_tokens_per_step must be >= 0")
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.context_len = min(
            self.context_len,
            getattr(self.hf_config, "max_position_embeddings", self.context_len),
        )

    @property
    def block_size(self) -> int:
        return self.kvcache_block_size
