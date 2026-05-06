"""Compare MHDSRA2 vs Standard Attention on tiny LLaMA LM task."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny_llama_shared import LMConfig
from scripts.tiny_llama_mhdsra2 import main_mhdsra2
from scripts.tiny_llama_baseline import main_standard


def run_comparison(config: dict | None = None) -> dict[str, float]:
    """Run both models and return perplexity comparison."""
    cfg = dict(LMConfig)
    if config:
        cfg.update(config)

    results: dict[str, float] = {}

    print("=" * 60)
    print("Training Standard Attention LM (baseline)")
    print("=" * 60)
    t0 = time.time()
    ppl_standard = main_standard(cfg)
    std_time = time.time() - t0
    results["standard_ppl"] = ppl_standard
    results["standard_time_s"] = std_time
    print(f"[Done] Standard PPL: {ppl_standard:.2f}, Time: {std_time:.0f}s\n")

    print("=" * 60)
    print("Training MHDSRA2 LM")
    print("=" * 60)
    t0 = time.time()
    ppl_mhdsra2 = main_mhdsra2(cfg)
    mh_time = time.time() - t0
    results["mhdsra2_ppl"] = ppl_mhdsra2
    results["mhdsra2_time_s"] = mh_time
    print(f"[Done] MHDSRA2 PPL: {ppl_mhdsra2:.2f}, Time: {mh_time:.0f}s\n")

    # Summary
    ratio = ppl_mhdsra2 / ppl_standard if ppl_standard > 0 else float("inf")
    print("=" * 60)
    print("COMPARISON RESULT")
    print("=" * 60)
    print(f"  Standard Attention PPL: {ppl_standard:.2f}")
    print(f"  MHDSRA2 PPL:            {ppl_mhdsra2:.2f}")
    print(f"  Ratio:                  {ratio:.3f}x")
    print(f"  Training Time Std:      {std_time:.0f}s")
    print(f"  Training Time MHDSRA2:  {mh_time:.0f}s")
    if ratio <= 1.2:
        print("  ✅ MHDSRA2 achieves comparable perplexity.")
    elif ratio <= 1.5:
        print("  🟡 MHDSRA2 needs optimization (RoPE, param tuning).")
    else:
        print("  🔴 MHDSRA2 significantly underperforms — investigate bottlenecks.")
    print()

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare MHDSRA2 vs Standard Attention")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=50000, help="Training steps")
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--only", type=str, default=None,
                        choices=["standard", "mhdsra2"],
                        help="Run only one model")
    return parser


def main():
    args = build_parser().parse_args()
    config = {
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "dim": args.dim,
        "heads": args.heads,
        "num_layers": args.layers,
        "device": args.device,
    }

    if args.only == "standard":
        main_standard(config)
    elif args.only == "mhdsra2":
        main_mhdsra2(config)
    else:
        run_comparison(config)


if __name__ == "__main__":
    main()
