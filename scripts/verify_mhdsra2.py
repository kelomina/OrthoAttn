import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.dsra.mhdsra2.improved_dsra_mha import (
    MHDSRA2Config,
    MultiHeadDSRA2,
    estimate_attention_memory_bytes,
    format_bytes,
)


def smoke_test():
    torch.manual_seed(0)
    cfg = MHDSRA2Config(
        dim=128,
        heads=4,
        slots=32,
        read_topk=4,
        write_topk=2,
        local_window=32,
        use_local=True,
        use_retrieval=True,
        detach_state=True,
    )
    layer = MultiHeadDSRA2(cfg)
    x1 = torch.randn(2, 16, cfg.dim)
    y1, state, _ = layer(x1, return_aux=True)
    assert y1.shape == x1.shape
    assert state.slot_k.shape == (2, cfg.heads, cfg.slots, cfg.dim // cfg.heads)
    assert state.local_k.shape[2] <= cfg.local_window

    r = 10
    rk = torch.randn(2, cfg.heads, r, cfg.dim // cfg.heads)
    rv = torch.randn(2, cfg.heads, r, cfg.dim // cfg.heads)
    x2 = torch.randn(2, 12, cfg.dim)
    y2, state, aux2 = layer(x2, state=state, retrieved_k=rk, retrieved_v=rv, return_aux=True)
    assert y2.shape == x2.shape
    assert state.local_k.shape[2] <= cfg.local_window
    assert torch.isfinite(y2).all()
    assert torch.isfinite(state.slot_k).all()
    print("[OK] smoke test passed")
    print("gates_mean_after_retrieval [slot, local, retrieval] per head:")
    print(aux2["gates_mean"])


def scaling_report(args):
    mem = estimate_attention_memory_bytes(
        seq_len=args.seq_len,
        batch_size=args.batch,
        dim=args.dim,
        heads=args.heads,
        chunk_size=args.chunk,
        slots=args.slots,
        read_topk=args.read_topk,
        write_topk=args.write_topk,
        local_window=args.local_window,
        retrieval_tokens=args.retrieval_tokens,
        dtype_bytes=args.dtype_bytes,
        page_size=args.page_size,
        keep_full_input_output_on_gpu=args.keep_full_io,
    )
    print("\nEstimated GPU attention working-set memory")
    print("------------------------------------------")
    for k, v in mem.items():
        print(f"{k:28s} {format_bytes(v)}")
    print("------------------------------------------")
    print(f"seq_len={args.seq_len:,}, chunk={args.chunk}, dim={args.dim}, heads={args.heads}")
    print(
        "Note: external exact token memory is assumed to live on CPU/NVMe; "
        "only retrieved tokens and landmarks are counted on GPU."
    )


def micro_benchmark(args):
    if not args.run_bench:
        return
    cfg = MHDSRA2Config(
        dim=args.dim,
        heads=args.heads,
        slots=args.slots,
        read_topk=args.read_topk,
        write_topk=args.write_topk,
        local_window=args.local_window,
        use_local=True,
        use_retrieval=True,
        detach_state=True,
    )
    layer = MultiHeadDSRA2(cfg)
    x = torch.randn(args.batch, args.chunk, args.dim)
    state = None
    rk = torch.randn(args.batch, args.heads, args.retrieval_tokens, args.dim // args.heads)
    rv = torch.randn(args.batch, args.heads, args.retrieval_tokens, args.dim // args.heads)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.steps):
            _, state = layer(x, state=state, retrieved_k=rk, retrieved_v=rv)
    dt = time.perf_counter() - t0
    toks = args.steps * args.batch * args.chunk
    print(f"\nCPU micro-benchmark: {toks:,} tokens in {dt:.3f}s -> {toks / max(dt, 1e-9):.1f} tok/s")
    print("This is only a shape/scaling test; GPU kernels and external retrieval are not benchmarked here.")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=2_000_000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--read-topk", type=int, default=8)
    parser.add_argument("--write-topk", type=int, default=4)
    parser.add_argument("--local-window", type=int, default=512)
    parser.add_argument("--retrieval-tokens", type=int, default=128)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--page-size", type=int, default=1024)
    parser.add_argument("--keep-full-io", action="store_true")
    parser.add_argument("--run-bench", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    smoke_test()
    scaling_report(args)
    micro_benchmark(args)


if __name__ == "__main__":
    main()
