"""Profile batch-isolated MHDSRA2 paged retrieval.

This script compares one batch retrieval call with an equivalent loop of
batch=1 repositories. It also prints the internal `PagedExactMemory.retrieve`
profile so exact retrieval optimizations can be evaluated without changing
model code.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dsra.infrastructure.paged_memory_repository import PagedMemoryRepository  # noqa: E402


def _parse_int_list(value: str) -> list[int]:
    """Parse comma-separated positive integers from a CLI argument."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    parsed = [int(item) for item in items]
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def _resolve_device(value: str) -> torch.device:
    """Resolve the requested benchmark device with cuda:0 as the only CUDA target."""
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        value = "cuda:0"
    device = torch.device(value)
    if device.type == "cuda" and device.index not in (0, None):
        raise argparse.ArgumentTypeError("only cuda:0 is supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise argparse.ArgumentTypeError("CUDA requested but not available")
    return torch.device("cuda:0") if device.type == "cuda" else device


def _build_marker_tensors(
    *,
    batch_size: int,
    heads: int,
    tokens: int,
    d_head: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic K/V/query tensors with one marker dimension per sample.

    中文说明:
    - 调用方 / Called by: benchmark runner.
    - 调用对象 / Calls: `torch.zeros`.
    - 作用 / Purpose: 构造每个 batch 样本互不相同的 key/value marker，便于检查是否串样本。
    - 参数 / Parameters: `batch_size/heads/tokens/d_head/device` 控制 benchmark 张量形状。
    - 返回 / Returns: `(key, value, query)`，形状均适合 `PagedMemoryRepository`。
    - 错误处理 / Error handling: `batch_size > d_head` 时 marker 维度不够，抛出 ValueError。
    """
    if batch_size > d_head:
        raise ValueError("d_head must be at least batch_size for marker dimensions")
    key = torch.zeros(batch_size, heads, tokens, d_head, dtype=torch.float32, device=device)
    value = torch.zeros_like(key)
    for sample_idx in range(batch_size):
        key[sample_idx, :, :, sample_idx] = 1.0
        value[sample_idx, :, :, sample_idx] = float(sample_idx + 1)
    query = key[:, :, -1:, :].clone()
    return key, value, query


def _check_no_cross_sample(
    retrieved_v: torch.Tensor | None,
    retrieved_mask: torch.Tensor | None,
    *,
    batch_size: int,
) -> bool:
    """Check that every retrieved value row carries only its own sample marker."""
    if retrieved_v is None or retrieved_mask is None:
        return False
    mask = _mask_as_batch_rows(retrieved_mask, batch_size=batch_size)
    for sample_idx in range(batch_size):
        valid = mask[sample_idx]
        if not bool(valid.any().item()):
            return False
        row = retrieved_v[sample_idx, :, valid, :]
        expected = row[..., sample_idx]
        if not bool(torch.all(expected == float(sample_idx + 1)).item()):
            return False
        leak_dims = [dim for dim in range(batch_size) if dim != sample_idx]
        if leak_dims and not bool(torch.all(row[..., leak_dims] == 0.0).item()):
            return False
    return True


def _mask_as_batch_rows(mask: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    """Normalize batch=1 legacy [R] masks to [1,R] for benchmark reporting."""
    if mask.dim() == 1 and batch_size == 1:
        return mask.unsqueeze(0)
    return mask


def _median_ms(fn, *, repeats: int, warmup: int) -> tuple[float, object]:
    """Return median wall-clock milliseconds and the last callable result."""
    last_result = None
    for _ in range(warmup):
        last_result = fn()
    elapsed: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        last_result = fn()
        elapsed.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(elapsed), last_result


def _mean_profile(profiles: Iterable[dict[str, object] | None]) -> dict[str, object]:
    """Average numeric profile fields and keep common metadata fields."""
    rows = [row for row in profiles if row]
    if not rows:
        return {}
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    averaged = {
        key: statistics.mean(float(row.get(key, 0.0)) for row in rows)
        for key in numeric_keys
    }
    for key in (
        "page_score_mode",
        "batch_size",
        "top_pages",
        "max_tokens",
        "max_retrieved",
        "valid_page_counts",
        "safe_token_counts",
        "retrieved_token_counts",
    ):
        if key in rows[-1]:
            averaged[key] = rows[-1][key]
    return averaged


def _make_repository(
    *,
    page_size: int,
    top_pages: int,
    max_tokens: int,
) -> PagedMemoryRepository:
    return PagedMemoryRepository(
        enabled=True,
        page_size=page_size,
        dtype=torch.float32,
        top_pages=top_pages,
        max_tokens=max_tokens,
    )


def _profile_case(
    *,
    batch_size: int,
    tokens: int,
    page_size: int,
    top_pages: int,
    max_tokens: int,
    repeats: int,
    warmup: int,
    device: torch.device,
) -> dict[str, object]:
    """Profile one batch retrieval configuration and its batch=1 loop equivalent."""
    heads = 2
    d_head = max(8, batch_size)
    key, value, query = _build_marker_tensors(
        batch_size=batch_size,
        heads=heads,
        tokens=tokens,
        d_head=d_head,
        device=device,
    )
    batch_repository = _make_repository(
        page_size=page_size,
        top_pages=top_pages,
        max_tokens=max_tokens,
    )
    batch_repository.append(key, value)
    loop_repositories = []
    for sample_idx in range(batch_size):
        repository = _make_repository(
            page_size=page_size,
            top_pages=top_pages,
            max_tokens=max_tokens,
        )
        repository.append(key[sample_idx : sample_idx + 1], value[sample_idx : sample_idx + 1])
        loop_repositories.append(repository)

    batch_profiles: list[dict[str, object] | None] = []
    loop_profiles: list[dict[str, object] | None] = []

    def batch_call():
        result = batch_repository.retrieve(
            query,
            device=device,
            return_mask=True,
            profile=True,
        )
        batch_profiles.append(batch_repository.memory.last_retrieve_profile)
        return result

    def loop_call():
        rows = []
        for sample_idx, repository in enumerate(loop_repositories):
            rows.append(
                repository.retrieve(
                    query[sample_idx : sample_idx + 1],
                    device=device,
                    return_mask=True,
                    profile=True,
                )
            )
            loop_profiles.append(repository.memory.last_retrieve_profile)
        return rows

    batch_ms, batch_result = _median_ms(batch_call, repeats=repeats, warmup=warmup)
    loop_ms, loop_result = _median_ms(loop_call, repeats=repeats, warmup=warmup)
    retrieved_k, retrieved_v, retrieved_mask = batch_result
    batch_mask_rows = (
        None
        if retrieved_mask is None
        else _mask_as_batch_rows(retrieved_mask, batch_size=batch_size)
    )
    batch_hit_counts = (
        None if batch_mask_rows is None else batch_mask_rows.sum(dim=1).detach().cpu().tolist()
    )
    loop_hit_counts = [
        None if row[2] is None else int(row[2].sum().item())
        for row in loop_result
    ]
    return {
        "batch_size": batch_size,
        "tokens": tokens,
        "page_size": page_size,
        "top_pages": top_pages,
        "max_tokens": max_tokens,
        "device": str(device),
        "batch_retrieval_ms_median": batch_ms,
        "loop_batch1_ms_median": loop_ms,
        "speedup_vs_loop": None if batch_ms <= 0 else loop_ms / batch_ms,
        "passed": _check_no_cross_sample(
            retrieved_v,
            retrieved_mask,
            batch_size=batch_size,
        ),
        "batch_retrieved_shape": None if retrieved_k is None else tuple(retrieved_k.shape),
        "batch_hit_counts": batch_hit_counts,
        "loop_hit_counts": loop_hit_counts,
        "batch_profile": _mean_profile(batch_profiles),
        "loop_profile": _mean_profile(loop_profiles),
    }


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    """Write a compact benchmark summary table."""
    rows = payload["results"]
    lines = [
        "# MHDSRA2 Batched Retrieval Profile",
        "",
        f"- device: `{payload['device']}`",
        f"- repeats: `{payload['repeats']}`",
        f"- warmup: `{payload['warmup']}`",
        "",
        "| B | Tokens | Page | top_pages | max_tokens | batch ms | loop ms | speedup | mode | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        profile = row.get("batch_profile") or {}
        lines.append(
            "| {batch_size} | {tokens} | {page_size} | {top_pages} | {max_tokens} | "
            "{batch_retrieval_ms_median:.3f} | {loop_batch1_ms_median:.3f} | "
            "{speedup:.3f} | {mode} | {passed} |".format(
                **row,
                speedup=float(row["speedup_vs_loop"] or 0.0),
                mode=profile.get("page_score_mode", "n/a"),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=_parse_int_list, default=[1, 4, 8])
    parser.add_argument("--tokens", type=_parse_int_list, default=[64, 256, 1024])
    parser.add_argument("--page-sizes", type=_parse_int_list, default=[16, 64])
    parser.add_argument("--top-pages", type=_parse_int_list, default=[2, 4])
    parser.add_argument("--max-tokens", type=_parse_int_list, default=[4, 16])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = _resolve_device(args.device)
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup must be non-negative")
    results = []
    for batch_size in args.batch_sizes:
        for tokens in args.tokens:
            for page_size in args.page_sizes:
                for top_pages in args.top_pages:
                    for max_tokens in args.max_tokens:
                        results.append(
                            _profile_case(
                                batch_size=batch_size,
                                tokens=tokens,
                                page_size=page_size,
                                top_pages=top_pages,
                                max_tokens=max_tokens,
                                repeats=args.repeats,
                                warmup=args.warmup,
                                device=device,
                            )
                        )
    payload: dict[str, object] = {
        "device": str(device),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "results": results,
        "passed": all(bool(row["passed"]) for row in results),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.markdown_out is not None:
        _write_markdown(args.markdown_out, payload)
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
