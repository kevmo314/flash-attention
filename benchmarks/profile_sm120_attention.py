#!/usr/bin/env python3
"""Profile FA2/FA4 forward attention on SM120.

Example:
  ncu --profile-from-start off --set basic --target-processes all \
    python benchmarks/profile_sm120_attention.py --backend fa4 --hdim 128 --seqlen 4096

The script warms the selected backend, optionally prints a timing, then brackets
one measured call with cudaProfilerStart/cudaProfilerStop so Nsight Compute can
ignore Python setup and CuTe JIT compilation.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch


def install_namespace(fa2_root: str | None, fa4_root: str | None) -> None:
    """Allow a FA4 shim package and FA2 extension tree to coexist."""
    roots = [root for root in (fa4_root, fa2_root) if root]
    if not roots:
        return
    for root in roots:
        sys.path.insert(0, root)
    mod = types.ModuleType("flash_attn")
    mod.__path__ = [str(Path(root) / "flash_attn") for root in roots]
    mod.__version__ = "profile-sm120"
    sys.modules["flash_attn"] = mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["fa2", "fa4"], required=True)
    parser.add_argument("--fa2-root", default=None, help="Root containing flash_attn_2_cuda and flash_attn/")
    parser.add_argument("--fa4-root", default=None, help="Root containing flash_attn/cute")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--total-seqlen", type=int, default=32768)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--heads-kv", type=int, default=None)
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--bench", action="store_true", help="Print triton.do_bench timing before profiling")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def logical_tflops(batch: int, seqlen: int, heads: int, hdim: int, causal: bool, ms: float) -> float:
    flops = 4 * batch * seqlen * seqlen * heads * hdim / (2 if causal else 1)
    return flops / (ms * 1e-3) / 1e12


def main() -> None:
    args = parse_args()
    install_namespace(args.fa2_root, args.fa4_root)

    if args.backend == "fa2":
        from flash_attn.flash_attn_interface import flash_attn_func

        def run(q, k, v):
            return flash_attn_func(q, k, v, 0.0, causal=args.causal)

    else:
        from flash_attn.cute.interface import flash_attn_func

        def run(q, k, v):
            out = flash_attn_func(q, k, v, causal=args.causal)
            return out[0] if isinstance(out, tuple) else out

    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    batch = args.batch if args.batch is not None else args.total_seqlen // args.seqlen
    heads = args.heads if args.heads is not None else (16 if args.hdim >= 192 else 32)
    heads_kv = args.heads_kv if args.heads_kv is not None else heads
    q = torch.randn(batch, args.seqlen, heads, args.hdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, args.seqlen, heads_kv, args.hdim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, args.seqlen, heads_kv, args.hdim, device=device, dtype=torch.bfloat16)

    for _ in range(args.warmup):
        run(q, k, v)
    torch.cuda.synchronize()

    if args.bench:
        import triton

        ms = triton.testing.do_bench(lambda: run(q, k, v), warmup=args.warmup, rep=20)
        print(
            f"{args.backend} batch={batch} seqlen={args.seqlen} heads={heads} "
            f"heads_kv={heads_kv} hdim={args.hdim} causal={args.causal} "
            f"ms={ms:.4f} tflops={logical_tflops(batch, args.seqlen, heads, args.hdim, args.causal, ms):.1f}",
            flush=True,
        )

    torch.cuda.cudart().cudaProfilerStart()
    out = run(q, k, v)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    # Keep the result live and make the call observable in eager mode.
    print(f"done backend={args.backend} checksum={float(out.float().sum()):.6f}", flush=True)


if __name__ == "__main__":
    main()
