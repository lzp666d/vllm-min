"""chunked prefill 行为验证（对齐 vLLM SchedulerConfig 语义）。

验证 max_prefill_chunk（vLLM: long_prefill_token_threshold）：
  1. =0 时不主动切分：prompt=3000 一步 prefill 完成，无"半截"步；
  2. =1024 时主动切片：3000 跨 3 个 chunk（2 次半截 + 1 次收尾）；
  3. 切分期间 decode 仍每步推进（混合批处理未被破坏）；
  4. max_num_partial_prefills=1：同一步最多一个未完成 prefill。

用法:
    python test/test_chunked_prefill.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import destroy, dummy_ids, make_engine, warmup  # noqa: E402
from src import SamplingParams  # noqa: E402

PROMPT_LEN = 3000
CHUNK = 1024


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _run(chunk: int, n_long: int = 1, batch_cap: int = 0):
    """返回 (半截 prefill 步数, 单步并发半截峰值, prefill 窗口内短请求产出 token 数, ...)。"""
    engine = make_engine(
        cg=False, compile=False, prefix="none", mix=True,
        max_num_seqs=8, max_batched=8192, gpu_mem=0.85,
        max_prefill_chunk=chunk,
        max_prefill_tokens_per_step=batch_cap,
    )
    try:
        warmup(engine, 8)
        sp_short = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
        sp_long = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
        shorts = [engine.add_request(dummy_ids(128, salt=i), sp_short) for i in range(4)]
        longs = [
            engine.add_request(dummy_ids(PROMPT_LEN, salt=9 + j), sp_long)
            for j in range(n_long)
        ]

        partial_steps = 0
        max_concurrent_partial = 0
        decode_in_window = 0
        prev_short = [s.num_completion_tokens for s in shorts]
        prev_cached = [s.num_cached_tokens for s in longs]

        while not engine.scheduler.is_finished():
            engine.step()
            # 本步"推进了但没跑完"的长请求 = 半截 prefill
            n_now = sum(
                1
                for s, pc in zip(longs, prev_cached)
                if s.num_cached_tokens > pc and s.num_cached_tokens < s.num_prompt_tokens
            )
            if n_now:
                partial_steps += 1
                max_concurrent_partial = max(max_concurrent_partial, n_now)
            # 长请求尚未 prefill 完的窗口里，短请求产出了多少 token
            if any(s.num_cached_tokens < s.num_prompt_tokens for s in longs):
                decode_in_window += sum(
                    s.num_completion_tokens - p for s, p in zip(shorts, prev_short)
                )
            prev_short = [s.num_completion_tokens for s in shorts]
            prev_cached = [s.num_cached_tokens for s in longs]

        return partial_steps, max_concurrent_partial, decode_in_window, shorts, longs
    finally:
        destroy(engine)


def check_chunk_off() -> None:
    steps, _, _, shorts, longs = _run(0)
    _check(steps == 0, f"chunk=0 不应出现半截 prefill，实际 {steps} 步")
    _check(all(s.num_completion_tokens == 8 for s in shorts), "短请求未全部完成")
    _check(all(s.num_completion_tokens == 4 for s in longs), "长请求未完成")


def check_chunk_on() -> None:
    steps, _, decode_in_window, shorts, longs = _run(CHUNK)
    expect = math.ceil(PROMPT_LEN / CHUNK) - 1  # 3 个 chunk -> 2 次半截
    _check(steps == expect, f"chunk={CHUNK} 期望 {expect} 次半截 prefill，实际 {steps}")
    _check(decode_in_window > 0, "切分期间 decode 未推进（混合调度失效）")
    _check(all(s.num_completion_tokens == 8 for s in shorts), "短请求未全部完成")
    _check(all(s.num_completion_tokens == 4 for s in longs), "长请求未完成")


def check_partial_limit() -> None:
    _, conc, _, shorts, longs = _run(CHUNK, n_long=2)
    _check(conc <= 1, f"max_num_partial_prefills=1，但出现 {conc} 个并发半截 prefill")
    _check(all(s.num_completion_tokens == 8 for s in shorts), "短请求未全部完成")
    _check(all(s.num_completion_tokens == 4 for s in longs), "长请求未完成")


def check_sarathi_fair_share() -> None:
    """Sarathi 模式（每步 prefill 总配额 + 公平份额）：一步内多个长请求各推进一片。"""
    _, conc, decode_in_window, shorts, longs = _run(CHUNK, n_long=4, batch_cap=2048)
    _check(conc >= 2, f"每步配额模式下一步应推进多个长请求，实际并发峰值 {conc}")
    _check(decode_in_window > 0, "切分期间 decode 未推进")
    _check(all(s.num_completion_tokens == 8 for s in shorts), "短请求未全部完成")
    _check(all(s.num_completion_tokens == 4 for s in longs), "长请求未完成")


def main() -> None:
    torch.cuda.set_device("cuda:0")
    cases = [
        ("chunk_off_no_partial", check_chunk_off),
        ("chunk_on_1024_three_chunks", check_chunk_on),
        ("max_num_partial_prefills_1", check_partial_limit),
        ("sarathi_fair_share_multi", check_sarathi_fair_share),
    ]
    failed = 0
    for name, fn in cases:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {e}")

    if failed:
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
