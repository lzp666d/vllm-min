"""Single-GPU sweep: eager vs Decode CUDA Graph vs CG+prefill compile.

每档在**独立子进程**里跑 ``bench/moe_e2e.py --json``（避免 CUDA Graph / compile
状态互相污染），最后汇总成 Markdown 表。单卡 Qwen3-0.6B 的
``e2e tok/s`` 与 ``vs eager`` 一列，就是简历里「×× tok/s、较 Eager 提升 ×」的口径。

用法（在 vllm-v5/ 下）::

    export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.
    python bench/single_card_sweep.py --device 0                 # Qwen3-0.6B
    python bench/single_card_sweep.py --device 1 --model moe     # MoE 档位
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS: tuple[tuple[str, list[str]], ...] = (
    ("eager", ["--eager"]),
    ("cg", ["--cg"]),
    ("cg+compile", ["--cg", "--compile"]),
)


def _run_cell(
    *,
    device: int,
    cell_args: list[str],
    model: str,
    n: int,
    warmup: int,
    reps: int,
    backend: str,
) -> dict:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    root = str(ROOT)
    env["PYTHONPATH"] = root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    cmd = [
        sys.executable,
        "bench/moe_e2e.py",
        "--model",
        model,
        "--devices",
        "0",
        "--n",
        str(n),
        "--warmup",
        str(warmup),
        "--reps",
        str(reps),
        "--moe-backend",
        backend,
        "--json",
        *cell_args,
    ]
    proc = subprocess.run(
        cmd, cwd=root, env=env, capture_output=True, text=True, timeout=7200
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"cell {cell_args} failed (device={device})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"no JSON from cell {cell_args}\n{proc.stdout}")
    return json.loads(lines[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0, help="物理 GPU 号（可用 nvidia-smi 查）")
    ap.add_argument("--model", choices=("dense", "moe"), default="dense")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--moe-backend", default="triton")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows: list[tuple[str, dict]] = []
    for label, extra in CELLS:
        r = _run_cell(
            device=args.device,
            cell_args=extra,
            model=args.model,
            n=args.n,
            warmup=args.warmup,
            reps=args.reps,
            backend=args.moe_backend,
        )
        rows.append((label, r))
        print(f"[ok] {label}: {r['e2e_tok_s']:.1f} tok/s", flush=True)

    base = rows[0][1]
    if args.json:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "device": args.device,
                    "n": args.n,
                    "cells": {label: r for label, r in rows},
                },
                ensure_ascii=False,
            )
        )
        return

    print(f"\n# 单卡 sweep — model={args.model} device={args.device} n={args.n}\n")
    print("| 档位 | e2e tok/s | decode tok/s | TTFT ms | TPOT ms | vs eager |")
    print("|---|---:|---:|---:|---:|---:|")
    for label, r in rows:
        speed = r["e2e_tok_s"] / base["e2e_tok_s"] if base["e2e_tok_s"] else float("nan")
        print(
            f"| {label} | {r['e2e_tok_s']:.1f} | {r['decode_tok_s']:.1f} | "
            f"{r['ttft_ms']:.2f} | {r['tpot_ms']:.3f} | {speed:.2f}x |"
        )


if __name__ == "__main__":
    main()
