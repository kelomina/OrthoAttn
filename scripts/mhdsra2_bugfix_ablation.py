"""Isolated verification for the MHDSRA2 bug-fix bundle.

This script is intentionally small: each check targets one independent fix from
the 2026-06-02 slot/paged-memory patch set, without running long training jobs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dsra.dsra_layer import DSRA_Chunk_Layer  # noqa: E402
from src.dsra.infrastructure.paged_memory_repository import PagedMemoryRepository  # noqa: E402
from src.dsra.mhdsra2.improved_dsra_mha import (  # noqa: E402
    MHDSRA2Config,
    MHDSRA2State,
    MultiHeadDSRA2,
)
from src.dsra.mhdsra2.paged_exact_memory import PagedExactMemory  # noqa: E402


def _check_slot_same_key_overwrite() -> dict[str, object]:
    """Verify same-key writes are not blocked by zero novelty.

    中文说明:
    - 调用方 / Called by: `main`.
    - 调用对象 / Calls: `MultiHeadDSRA2._slot_write`.
    - 作用 / Purpose: 独立验证 slot overwrite-aware 写入门修复。
    - 返回 / Returns: 包含 pass 状态和关键诊断值的字典。
    - 错误处理 / Error handling: 不吞异常；调用方统一捕获并汇总。
    """
    cfg = MHDSRA2Config(
        dim=2,
        heads=1,
        slots=1,
        read_topk=1,
        write_topk=1,
        use_local=False,
        use_retrieval=False,
        detach_state=False,
    )
    layer = MultiHeadDSRA2(cfg)
    with torch.no_grad():
        layer.token_write_gate.weight.zero_()
        layer.token_write_gate.bias.fill_(10.0)

    same_key = torch.tensor([[[[1.0, 0.0]]]])
    state = MHDSRA2State(
        slot_k=same_key.clone(),
        slot_v=torch.tensor([[[[1.0, 0.0]]]]),
        age=torch.zeros(1, 1, 1),
        usage=torch.zeros(1, 1, 1),
        confidence=torch.ones(1, 1, 1),
        position=0,
    )
    next_state = layer._slot_write(
        same_key,
        torch.tensor([[[[0.0, 1.0]]]]),
        state,
        torch.ones(1, 1, 1),
    )
    return {
        "passed": bool(
            layer.last_write_stats["write_mass_max"] > 0
            and layer.last_write_stats["write_gate_max"] > 0
            and next_state.slot_v[0, 0, 0, 1] > state.slot_v[0, 0, 0, 1]
        ),
        "novelty_mean": float(layer.last_write_stats["novelty_mean"]),
        "write_mass_max": float(layer.last_write_stats["write_mass_max"]),
        "write_gate_max": float(layer.last_write_stats["write_gate_max"]),
    }


def _check_paged_single_token_recall() -> dict[str, object]:
    """Verify a single exact token is not filtered out by page mean scoring."""
    memory = PagedExactMemory(page_size=4, dtype=torch.float32)
    key = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [-1.0, 0.0],
                    [-1.0, 0.0],
                    [-1.0, 0.0],
                    [0.2, 0.98],
                    [0.2, 0.98],
                    [0.2, 0.98],
                    [0.2, 0.98],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    value = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1).expand(1, 1, 8, 2)
    memory.append(key, value)
    _, _, positions = memory.retrieve(
        torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32),
        top_pages=1,
        max_tokens=1,
    )
    found = None if positions is None else positions.tolist()
    return {"passed": found == [0], "retrieved_positions": found}


def _check_repository_batch_isolated() -> dict[str, object]:
    """Verify direct repository batch>1 retrieval stays sample-isolated.

    中文说明:
    - 调用方 / Called by: `main`.
    - 调用对象 / Calls: `PagedMemoryRepository.append`, `PagedMemoryRepository.retrieve`.
    - 作用 / Purpose: 独立验证 batch-isolated external memory 不串样本且返回有效 mask。
    - 返回 / Returns: 包含 pass 状态、mask 和两个样本 value marker 的字典。
    - 错误处理 / Error handling: 不吞异常；调用方统一捕获并汇总。
    """
    repository = PagedMemoryRepository(
        enabled=True,
        page_size=2,
        dtype=torch.float32,
        max_tokens=1,
    )
    key = torch.zeros(2, 1, 1, 4)
    value = torch.zeros(2, 1, 1, 4)
    key[0, 0, 0, 0] = 1.0
    key[1, 0, 0, 1] = 1.0
    value[0, 0, 0, 2] = 10.0
    value[1, 0, 0, 3] = 20.0
    repository.append(key, value)
    _, retrieved_v, retrieved_mask = repository.retrieve(
        key,
        device=key.device,
        return_mask=True,
    )
    sample0_marker = None if retrieved_v is None else float(retrieved_v[0, 0, 0, 2])
    sample0_leak = None if retrieved_v is None else float(retrieved_v[0, 0, 0, 3])
    sample1_leak = None if retrieved_v is None else float(retrieved_v[1, 0, 0, 2])
    sample1_marker = None if retrieved_v is None else float(retrieved_v[1, 0, 0, 3])
    return {
        "passed": bool(
            retrieved_v is not None
            and retrieved_mask is not None
            and retrieved_mask.tolist() == [[True], [True]]
            and sample0_marker == 10.0
            and sample0_leak == 0.0
            and sample1_leak == 0.0
            and sample1_marker == 20.0
        ),
        "mask": None if retrieved_mask is None else retrieved_mask.tolist(),
        "sample0_marker": sample0_marker,
        "sample0_leak": sample0_leak,
        "sample1_leak": sample1_leak,
        "sample1_marker": sample1_marker,
    }


def _check_forward_step_single_qkv() -> dict[str, object]:
    """Verify compatibility forward_step computes fast QKV only once."""
    layer = DSRA_Chunk_Layer(dim=8, K=4, kr=1, use_bypass=True)
    call_count = {"qkv": 0}
    original_forward = layer.core.qkv.forward

    def counted_forward(x: torch.Tensor) -> torch.Tensor:
        call_count["qkv"] += 1
        return original_forward(x)

    layer.core.qkv.forward = counted_forward
    out, state, kv_cache = layer.forward_step(torch.randn(1, 1, 8), None, None)
    return {
        "passed": bool(
            call_count["qkv"] == 1
            and out.shape == (1, 1, 8)
            and state.position == 1
            and kv_cache[0] is not None
            and tuple(kv_cache[0].shape) == (1, 1, 8)
        ),
        "qkv_calls": call_count["qkv"],
        "kv_cache_shape": None if kv_cache[0] is None else tuple(kv_cache[0].shape),
        "position": state.position,
    }


def main() -> int:
    checks: dict[str, Callable[[], dict[str, object]]] = {
        "slot_same_key_overwrite": _check_slot_same_key_overwrite,
        "paged_single_token_recall": _check_paged_single_token_recall,
        "repository_batch_isolated": _check_repository_batch_isolated,
        "forward_step_single_qkv": _check_forward_step_single_qkv,
    }
    results: dict[str, dict[str, object]] = {}
    for name, check in checks.items():
        try:
            results[name] = check()
        except Exception as exc:  # pragma: no cover - diagnostic script surface
            results[name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all(bool(row.get("passed")) for row in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
