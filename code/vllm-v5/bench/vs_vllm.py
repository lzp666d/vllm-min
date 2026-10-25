"""Compare this engine vs official vLLM on the same greedy workload.

Official vLLM is loaded via a **separate** interpreter (``--vllm-python``) so it
does not fight this repo's Torch / FlashInfer stack.

Metrics (both engines use the same prompts / max_tokens / greedy + ignore_eos):

    ttft_ms           首 token 延迟（平均）
    tpot_ms           每输出 token 时间 = (e2e - ttft) / (n - 1)
    decode_tok_s      单请求 decode 速度 = 1000 / tpot_ms
    throughput_tok_s  并发批的总吞吐 = 批内总输出 token / 墙钟
    tokens            逐 token 输出，用于正确性 spot-check

``--concurrency C`` 会一次并发投递 C 条不同请求（C==1 即单请求基线）。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import Engine, EngineConfig, SamplingParams  # noqa: E402
from tests.helpers.paths import DENSE_MODEL, MOE_MODEL  # noqa: E402

PROMPT_POOL = (
    "The capital of France is",
    "Hello, my name is",
    "The quick brown fox jumps over",
    "In machine learning, attention is",
    "Write a short poem about the sea:",
    "The best way to learn Python is",
    "Once upon a time, there was a",
    "Explain why the sky is blue:",
)

_VLLM_WORKER = r"""
import json, statistics, sys, time
model, prompts_json, n, warmup, reps, tp, max_model_len = sys.argv[1:8]
n, warmup, reps, tp, max_model_len = map(int, (n, warmup, reps, tp, max_model_len))
prompts = json.loads(prompts_json)
from vllm import LLM, SamplingParams
llm = LLM(
    model=model,
    tensor_parallel_size=tp,
    max_model_len=max_model_len,
    enforce_eager=True,
    disable_log_stats=True,
    gpu_memory_utilization=0.85,
)
for _ in range(warmup):
    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
e2es, last, total = [], [], 0
for _ in range(reps):
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    e2es.append(time.perf_counter() - t0)
    last = [list(o.outputs[0].token_ids) for o in outs]
    total = sum(len(t) for t in last)
e2e = statistics.mean(e2es)
# TTFT：官方 LLM.generate 不暴露流式首 token，用同一批 max_tokens=1 的墙钟近似。
sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
tt = []
for _ in range(reps):
    t0 = time.perf_counter()
    llm.generate(prompts, sp1)
    tt.append((time.perf_counter() - t0) * 1e3)
ttft = statistics.mean(tt)
decode_s = max(e2e - ttft / 1e3, 1e-6)
print(json.dumps({
    "engine": "vllm",
    "concurrency": len(prompts),
    "ttft_ms": ttft,
    "tpot_ms": (e2e - ttft / 1e3) / max(n - 1, 1) * 1e3,
    "e2e_s": e2e,
    "decode_tok_s": (n - 1) / decode_s,
    "throughput_tok_s": total / e2e,
    "tokens": last,
}))
"""


def _prompts(concurrency: int, num_prompts: int) -> list[str]:
    out: list[str] = []
    for i in range(num_prompts):
        p = PROMPT_POOL[i % len(PROMPT_POOL)]
        if i >= len(PROMPT_POOL):
            p = f"{p} ({i})"
        out.append(p)
    return out


def _ours(
    model: str,
    prompts: list[str],
    *,
    n: int,
    warmup: int,
    reps: int,
    devices: list[int],
    tp: int,
    enable_ep: bool,
    moe_backend: str,
) -> dict:
    eng = Engine(
        EngineConfig(
            model=model,
            device_ids=devices[:tp],
            tp_size=tp,
            enable_ep=enable_ep,
            enforce_eager=True,
            max_num_seqs=max(len(prompts), 1),
            context_len=2048,
            num_kvcache_blocks=64,
            gpu_memory_utilization=0.85,
            prefix_backend="none",
            moe_backend=moe_backend,  # type: ignore[arg-type]
        )
    )
    try:
        sp_w = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
        for _ in range(warmup):
            for prompt in prompts:
                seq = eng.add_request(prompt, sp_w)
                while not seq.is_finished:
                    eng.step()

        sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        walls: list[float] = []
        ttfts: list[float] = []
        tpots: list[float] = []
        last: list[list[int]] = []
        for _ in range(reps):
            seqs = [eng.add_request(prompt, sp) for prompt in prompts]
            t0 = time.perf_counter()
            first: list[float | None] = [None] * len(seqs)
            while any(not s.is_finished for s in seqs):
                eng.step()
                now = time.perf_counter()
                for i, seq in enumerate(seqs):
                    if first[i] is None and seq.num_completion_tokens >= 1:
                        first[i] = now
            t1 = time.perf_counter()
            wall = t1 - t0
            walls.append(wall)
            for i, seq in enumerate(seqs):
                f = first[i] if first[i] is not None else t1
                ttfts.append((f - t0) * 1e3)
                tpots.append((wall - (f - t0)) / max(n - 1, 1) * 1e3)
            last = [list(seq.token_ids[seq.num_prompt_tokens :]) for seq in seqs]

        e2e = statistics.mean(walls)
        ttft = statistics.mean(ttfts)
        tpot = statistics.mean(tpots)
        decode_s = max(e2e - ttft / 1e3, 1e-6)
        total_tokens = sum(len(t) for t in last)
        return {
            "engine": "v5",
            "concurrency": len(prompts),
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "e2e_s": e2e,
            "decode_tok_s": (n - 1) / decode_s,
            "throughput_tok_s": total_tokens / e2e,
            "tokens": last,
        }
    finally:
        eng.destroy()


def _vllm(
    python: str,
    model: str,
    prompts: list[str],
    *,
    n: int,
    warmup: int,
    reps: int,
    tp: int,
    max_model_len: int,
    env: dict,
) -> dict:
    cmd = [
        python,
        "-c",
        _VLLM_WORKER,
        model,
        json.dumps(prompts),
        str(n),
        str(warmup),
        str(reps),
        str(tp),
        str(max_model_len),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            "official vLLM worker failed.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}\n"
            "Install vLLM in a separate env and pass --vllm-python."
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"no JSON from vLLM worker\n{proc.stdout}\n{proc.stderr}")
    return json.loads(lines[-1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("dense", "moe"), default="dense")
    p.add_argument("--devices", default="1")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--ep", action="store_true", help="v5 only (official uses TP)")
    p.add_argument("--concurrency", type=int, default=1, help="并发请求数（批量吞吐）")
    p.add_argument(
        "--num-prompts",
        type=int,
        default=0,
        help="总请求数，默认等于 --concurrency",
    )
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--prompt", default="")
    p.add_argument("--moe-backend", default="triton")
    p.add_argument(
        "--vllm-python",
        default="",
        help="Python with official `vllm` installed (separate venv recommended)",
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--skip-vllm", action="store_true")
    args = p.parse_args()

    devices = [int(x) for x in args.devices.split(",") if x.strip() != ""]
    model = str(DENSE_MODEL if args.model == "dense" else MOE_MODEL)
    if not Path(model).is_dir():
        raise SystemExit(f"missing model: {model}")

    num_prompts = args.num_prompts or max(args.concurrency, 1)
    if args.prompt:
        prompts = [args.prompt] * num_prompts
    else:
        prompts = _prompts(args.concurrency, num_prompts)

    ours = _ours(
        model,
        prompts,
        n=args.n,
        warmup=args.warmup,
        reps=args.reps,
        devices=devices,
        tp=args.tp,
        enable_ep=bool(args.ep),
        moe_backend=args.moe_backend,
    )
    if args.json:
        print(json.dumps({"ours": ours}, ensure_ascii=False))
    else:
        print(f"== v5 ({args.model}, concurrency={len(prompts)}) ==")
        print(json.dumps(ours, ensure_ascii=False))

    if args.skip_vllm:
        return
    vpy = args.vllm_python.strip()
    if not vpy:
        print(
            "\n[skip official vLLM] pass --vllm-python /path/to/venv/bin/python "
            "(see bench/README.md). Use --skip-vllm to silence this."
        )
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices[: args.tp])
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    theirs = _vllm(
        vpy,
        model,
        prompts,
        n=args.n,
        warmup=args.warmup,
        reps=args.reps,
        tp=args.tp,
        max_model_len=2048,
        env=env,
    )
    if args.json:
        print(json.dumps({"theirs": theirs}, ensure_ascii=False))
    else:
        print(f"\n== official vLLM ({args.model}) via {vpy} ==")
        print(json.dumps(theirs, ensure_ascii=False))

    lower_is_better = ("ttft_ms", "tpot_ms")
    print("\n== ratio (v5 / vllm) ==")
    for k in ("ttft_ms", "tpot_ms", "decode_tok_s", "throughput_tok_s"):
        a, b = ours[k], theirs[k]
        if b:
            print(f"  {k}: v5={a:.3f} vllm={b:.3f}  v5/vllm={a / b:.3f}")
    tok_same = ours["tokens"] == theirs["tokens"]
    print(f"  token_match={tok_same}")
    if not tok_same and len(ours["tokens"]) == 1:
        print(f"    v5[0]={ours['tokens'][0][:16]}\n    vllm[0]={theirs['tokens'][0][:16]}")


if __name__ == "__main__":
    main()
