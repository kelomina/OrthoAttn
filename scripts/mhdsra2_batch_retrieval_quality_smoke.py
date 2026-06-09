"""Smoke-check batch-isolated retrieval quality on synthetic NIAH/JSON probes.

This script is intentionally not a training benchmark.  It builds deterministic
needle-in-a-haystack and JSON-like key/value probes, then verifies that the
external paged exact memory retrieves the right historical tokens per batch row
without cross-sample leakage or future-token leakage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys
import time
from typing import Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dsra.dsra_model import MultiLayerMHDSRA2Model  # noqa: E402
from src.dsra.mhdsra2.paged_exact_memory import PagedExactMemory  # noqa: E402


def _parse_int_list(raw: str | Iterable[int]) -> list[int]:
    """Parse comma-separated positive integers used by the CLI and tests.

    中文说明:
    - 调用方 / Called by: CLI `main` and regression tests.
    - 调用对象 / Calls: string split and integer conversion only.
    - 作用 / Purpose: 将 `1,4,8` 这类命令行参数转成整数列表，避免每个参数重复解析逻辑。
    - 参数 / Parameters: `raw` 是逗号分隔字符串，或测试直接传入的整数序列。
    - 返回 / Returns: positive integer list.
    - 错误处理 / Error handling: 空值或非正整数抛出 `ValueError`，让 CLI 失败可见。

    English documentation:
    Function name:
        _parse_int_list
    Purpose:
        Convert comma-separated positive integers into a list for validation
        grids.
    """
    if isinstance(raw, str):
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [int(item) for item in raw]
    if not values or any(value <= 0 for value in values):
        raise ValueError("integer lists must contain at least one positive value")
    return values


def _resolve_device(raw: str) -> torch.device:
    """Resolve benchmark device while following the project's cuda:0 policy."""
    if raw == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if raw == "cuda":
        return torch.device("cuda:0")
    device = torch.device(raw)
    if device.type == "cuda" and device.index not in (0, None):
        raise argparse.ArgumentTypeError("only cuda:0 is supported")
    return torch.device("cuda:0") if device.type == "cuda" else device


def _one_hot_batch(batch_size: int, heads: int, tokens: int, dim: int) -> torch.Tensor:
    """Create deterministic per-sample token keys with a unique sample channel.

    中文说明:
    - 调用方 / Called by: `_build_niah_probe` and `_build_json_probe`.
    - 调用对象 / Calls: PyTorch tensor allocation only.
    - 作用 / Purpose: 为每个 batch row 构造可区分的 key 通道，使串样本召回会被 marker 检查发现。
    - 参数 / Parameters: batch/head/token/dim define the generated tensor shape.
    - 返回 / Returns: `[B,H,T,d]` float tensor on CPU.
    - 副作用 / Side effects: none.

    English documentation:
    Function name:
        _one_hot_batch
    Purpose:
        Build deterministic sample-specific keys for synthetic retrieval probes.
    """
    key = torch.zeros(batch_size, heads, tokens, dim, dtype=torch.float32)
    for sample_idx in range(batch_size):
        sample_channel = sample_idx % dim
        key[sample_idx, :, :, sample_channel] = 0.25
        for token_idx in range(tokens):
            key[sample_idx, :, token_idx, (sample_channel + token_idx + 3) % dim] += 0.5
            key[sample_idx, :, token_idx, (sample_channel + token_idx * 7 + 5) % dim] -= 0.25
    return key


def _owner_channel(sample_idx: int, dim: int) -> int:
    """Return the sample-owner sentinel channel used in synthetic values."""
    return int(sample_idx) % int(dim)


def _stamp_owner_values(value: torch.Tensor) -> None:
    """Stamp every value token with a per-sample owner sentinel in-place.

    中文说明:
    - 调用方 / Called by: synthetic probe builders.
    - 调用对象 / Calls: tensor assignment only.
    - 作用 / Purpose: 给每个样本的所有 value token 加上 owner 标记；如果 batch
      retrieval 串样本，即使位置命中也能通过 value owner 检查发现。
    - 参数 / Parameters: `value` is `[B,H,T,d]` and is modified in-place.
    - 副作用 / Side effects: writes owner sentinel values into `value`.

    English documentation:
    Function name:
        _stamp_owner_values
    Purpose:
        Mark all values with a sample-owner sentinel so cross-sample value leaks
        cannot hide behind correct positions.
    """
    batch_size, _, _, dim = value.shape
    for sample_idx in range(batch_size):
        value[sample_idx, :, :, _owner_channel(sample_idx, dim)] = 100.0 + sample_idx


def _build_niah_probe(
    *,
    batch_size: int,
    tokens: int,
    heads: int,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a synthetic single-needle retrieval probe.

    中文说明:
    - 调用方 / Called by: `run_quality_smoke`.
    - 调用对象 / Calls: `_one_hot_batch`.
    - 作用 / Purpose: 模拟 NIAH：每个样本在不同位置藏一个唯一 key，query 必须召回本样本的 needle。
    - 返回 / Returns: `(key, value, query, expected_positions)`.
    - 错误处理 / Error handling: 调用方保证 dim 足够容纳 marker 通道。

    English documentation:
    Function name:
        _build_niah_probe
    Purpose:
        Construct one exact-match needle per batch row.
    """
    key = _one_hot_batch(batch_size, heads, tokens, dim)
    value = torch.zeros_like(key)
    expected_positions = torch.empty(batch_size, dtype=torch.long)
    for sample_idx in range(batch_size):
        needle_pos = min(tokens - 1, max(0, tokens // 5 + sample_idx * max(1, tokens // 17)))
        needle_channel = (batch_size + sample_idx) % dim
        key[sample_idx, :, needle_pos, :] = 0.0
        key[sample_idx, :, needle_pos, needle_channel] = 1.0
        value[sample_idx, :, needle_pos, :] = 0.0
        value[sample_idx, :, needle_pos, needle_channel] = 10.0 + sample_idx
        expected_positions[sample_idx] = needle_pos
    _stamp_owner_values(value)
    query = torch.zeros(batch_size, heads, 1, dim, dtype=torch.float32)
    for sample_idx, needle_pos in enumerate(expected_positions.tolist()):
        query[sample_idx, :, 0, :] = key[sample_idx, :, needle_pos, :]
    return key, value, query, expected_positions


def _build_json_probe(
    *,
    batch_size: int,
    tokens: int,
    heads: int,
    dim: int,
    field_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a synthetic JSON-like key/value retrieval probe.

    中文说明:
    - 调用方 / Called by: `run_quality_smoke`.
    - 调用对象 / Calls: `_one_hot_batch`.
    - 作用 / Purpose: 模拟 JSON 对象里重复字段 latest-wins：同一字段先写旧值，再写新值，query 应召回新位置。
    - 返回 / Returns: `(key, value, query, expected_positions)`.
    - 副作用 / Side effects: none.

    English documentation:
    Function name:
        _build_json_probe
    Purpose:
        Construct repeated JSON-like field keys where the newest matching field
        should win ties by position.
    """
    key = _one_hot_batch(batch_size, heads, tokens, dim)
    value = torch.zeros_like(key)
    expected_positions = torch.empty(batch_size, dtype=torch.long)
    stride = max(2, tokens // max(field_count * 3, 1))
    for sample_idx in range(batch_size):
        target_field = sample_idx % field_count
        field_channel = (batch_size * 2 + target_field) % dim
        sample_channel = sample_idx % dim
        old_pos = min(tokens - 2, stride + sample_idx + target_field * stride)
        new_pos = min(tokens - 1, old_pos + max(1, tokens // 3))
        field_key = torch.zeros(dim, dtype=torch.float32)
        field_key[sample_channel] = 0.55
        field_key[field_channel] = 1.0
        key[sample_idx, :, old_pos, :] = field_key
        key[sample_idx, :, new_pos, :] = field_key
        value[sample_idx, :, old_pos, field_channel] = 1.0
        value[sample_idx, :, new_pos, field_channel] = 20.0 + sample_idx
        expected_positions[sample_idx] = new_pos
    _stamp_owner_values(value)
    query = torch.zeros(batch_size, heads, 1, dim, dtype=torch.float32)
    for sample_idx, expected_pos in enumerate(expected_positions.tolist()):
        query[sample_idx, :, 0, :] = key[sample_idx, :, expected_pos, :]
    return key, value, query, expected_positions


def _build_future_cutoff_probe(
    *,
    batch_size: int,
    tokens: int,
    heads: int,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a probe where a stronger future match must be filtered out.

    中文说明:
    - 调用方 / Called by: `run_quality_smoke`.
    - 调用对象 / Calls: `_one_hot_batch`, `_stamp_owner_values`.
    - 作用 / Purpose: 为每个样本设置不同 cutoff，并在 cutoff 后放置同 key future decoy；
      若 `max_position` 失效，latest-wins 会优先召回 future decoy，从而被 smoke 抓到。
    - 返回 / Returns: `(key, value, query, expected_positions, max_positions)`.
    - 错误处理 / Error handling: 调用方保证 token 数足够；默认网格和单测网格均满足。

    English documentation:
    Function name:
        _build_future_cutoff_probe
    Purpose:
        Construct per-sample max_position cutoffs with matching future decoys.
    """
    key = _one_hot_batch(batch_size, heads, tokens, dim)
    value = torch.zeros_like(key)
    expected_positions = torch.empty(batch_size, dtype=torch.long)
    max_positions = torch.empty(batch_size, dtype=torch.long)
    for sample_idx in range(batch_size):
        cutoff = min(tokens - 2, max(4, tokens // 2 + sample_idx))
        valid_pos = max(0, cutoff - 2)
        future_pos = min(tokens - 1, cutoff + 1)
        marker_channel = (batch_size * 3 + sample_idx) % dim
        future_marker_channel = (batch_size * 3 + sample_idx + batch_size) % dim
        query_key = torch.zeros(dim, dtype=torch.float32)
        query_key[marker_channel] = 1.0
        key[sample_idx, :, valid_pos, :] = query_key
        key[sample_idx, :, future_pos, :] = query_key
        value[sample_idx, :, valid_pos, marker_channel] = 30.0 + sample_idx
        value[sample_idx, :, future_pos, future_marker_channel] = 90.0 + sample_idx
        expected_positions[sample_idx] = valid_pos
        max_positions[sample_idx] = cutoff
    _stamp_owner_values(value)
    query = torch.zeros(batch_size, heads, 1, dim, dtype=torch.float32)
    for sample_idx, expected_pos in enumerate(expected_positions.tolist()):
        query[sample_idx, :, 0, :] = key[sample_idx, :, expected_pos, :]
    return key, value, query, expected_positions, max_positions


def _retrieve_once(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor,
    page_size: int,
    top_pages: int,
    max_tokens: int,
    max_position: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Append one probe to memory and retrieve with mask enabled."""
    memory = PagedExactMemory(page_size=page_size, dtype=torch.float32)
    memory.append(key, value)
    return memory.retrieve(
        query,
        top_pages=top_pages,
        max_tokens=max_tokens,
        device=device or query.device,
        max_position=max_position,
        return_mask=True,
    )


def _valid_positions_for_sample(
    positions: torch.Tensor,
    mask: torch.Tensor,
    sample_idx: int,
) -> torch.Tensor:
    """Return valid retrieved positions for one sample across old/new shapes."""
    if positions.dim() == 1:
        if sample_idx != 0:
            raise ValueError("rank-1 positions are only valid for batch_size=1")
        return positions[mask].detach().cpu()
    return positions[sample_idx][mask[sample_idx]].detach().cpu()


def _valid_values_for_sample(
    retrieved_v: torch.Tensor,
    mask: torch.Tensor,
    sample_idx: int,
) -> torch.Tensor:
    """Return valid retrieved values for one sample across old/new shapes."""
    if mask.dim() == 1:
        if sample_idx != 0:
            raise ValueError("rank-1 mask is only valid for batch_size=1")
        return retrieved_v[0, :, mask, :]
    return retrieved_v[sample_idx, :, mask[sample_idx], :]


def _compare_with_batch_one_loop(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor,
    retrieved_v: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    page_size: int,
    top_pages: int,
    max_tokens: int,
    max_position: torch.Tensor | None,
) -> bool:
    """Check batched retrieve positions against equivalent batch=1 loops.

    中文说明:
    - 调用方 / Called by: `run_quality_smoke`.
    - 调用对象 / Calls: `_retrieve_once`.
    - 作用 / Purpose: 用 batch=1 循环作为参考，确认 batch retrieval 没有因打包或 padding 改变召回结果。
    - 参数 / Parameters: batched key/value/query and batched retrieve outputs.
    - 返回 / Returns: True when every row matches its batch=1 reference.
    - 错误处理 / Error handling: 无有效召回时按 mask 和 pos=-1 同步比较。

    English documentation:
    Function name:
        _compare_with_batch_one_loop
    Purpose:
        Verify batch retrieval positions against independent batch=1 retrievals.
    """
    batch_size = key.shape[0]
    for sample_idx in range(batch_size):
        sample_max_position = None
        if max_position is not None:
            sample_max_position = max_position[sample_idx : sample_idx + 1]
        _, loop_v, loop_pos, loop_mask = _retrieve_once(
            key=key[sample_idx : sample_idx + 1],
            value=value[sample_idx : sample_idx + 1],
            query=query[sample_idx : sample_idx + 1],
            page_size=page_size,
            top_pages=top_pages,
            max_tokens=max_tokens,
            max_position=sample_max_position,
            device=query.device,
        )
        row_positions = _valid_positions_for_sample(positions, mask, sample_idx)
        if loop_v is None or loop_pos is None or loop_mask is None:
            if row_positions.numel() != 0:
                return False
            continue
        loop_positions = _valid_positions_for_sample(loop_pos, loop_mask, 0)
        if row_positions.tolist() != loop_positions.tolist():
            return False
        row_values = _valid_values_for_sample(retrieved_v, mask, sample_idx).detach().cpu()
        loop_values = _valid_values_for_sample(loop_v, loop_mask, 0).detach().cpu()
        if row_values.shape != loop_values.shape:
            return False
        if not torch.allclose(row_values, loop_values, atol=1e-6, rtol=1e-6):
            return False
    return True


def _score_probe(
    *,
    scenario: str,
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor,
    expected_positions: torch.Tensor,
    page_size: int,
    top_pages: int,
    max_tokens: int,
    max_position: torch.Tensor | None,
    elapsed_ms: float,
    retrieved_v: torch.Tensor | None,
    positions: torch.Tensor | None,
    mask: torch.Tensor | None,
    batch_loop_match: bool,
) -> dict[str, object]:
    """Summarize one synthetic probe into report-friendly metrics."""
    batch_size = key.shape[0]
    if retrieved_v is None or positions is None or mask is None:
        position_hits = [False for _ in range(batch_size)]
        top1_hits = [False for _ in range(batch_size)]
        marker_hits = [False for _ in range(batch_size)]
        owner_hits = [False for _ in range(batch_size)]
        owner_leaks = [False for _ in range(batch_size)]
        future_leaks = [False for _ in range(batch_size)]
        retrieved_counts = [0 for _ in range(batch_size)]
    else:
        position_hits = []
        top1_hits = []
        marker_hits = []
        owner_hits = []
        owner_leaks = []
        future_leaks = []
        retrieved_counts = []
        for sample_idx in range(batch_size):
            valid_positions = _valid_positions_for_sample(positions, mask, sample_idx)
            retrieved_counts.append(int(valid_positions.numel()))
            expected_pos = int(expected_positions[sample_idx].item())
            position_hits.append(expected_pos in valid_positions.tolist())
            top1_hits.append(
                bool(valid_positions.numel() > 0 and int(valid_positions[0].item()) == expected_pos)
            )
            marker_vector = value[sample_idx, :, expected_pos, :].abs().sum(dim=0)
            marker_channel = int(torch.argmax(marker_vector).item())
            if valid_positions.numel() == 0:
                marker_hits.append(False)
                owner_hits.append(False)
                owner_leaks.append(False)
            else:
                sample_values = _valid_values_for_sample(retrieved_v, mask, sample_idx)
                marker_hits.append(bool((sample_values[..., marker_channel] > 0.0).any().item()))
                owner_channel = _owner_channel(sample_idx, key.shape[-1])
                owner_expected = 100.0 + sample_idx
                owner_values = sample_values[..., owner_channel]
                owner_hits.append(
                    bool(torch.allclose(
                        owner_values,
                        torch.full_like(owner_values, owner_expected),
                        atol=1e-6,
                        rtol=1e-6,
                    ))
                )
                owner_leaks.append(not owner_hits[-1])
            if max_position is None:
                future_leaks.append(False)
            else:
                future_leaks.append(
                    bool((valid_positions >= int(max_position[sample_idx].item())).any().item())
                )
    cross_sample_leak = bool(any(owner_leaks))
    hit_rate = float(mean(position_hits)) if position_hits else 0.0
    top1_hit_rate = float(mean(top1_hits)) if top1_hits else 0.0
    marker_hit_rate = float(mean(marker_hits)) if marker_hits else 0.0
    owner_clean = bool(all(owner_hits)) if owner_hits else False
    no_future_leak = not any(future_leaks)
    passed = bool(
        hit_rate == 1.0
        and top1_hit_rate == 1.0
        and marker_hit_rate == 1.0
        and owner_clean
        and not cross_sample_leak
        and no_future_leak
        and batch_loop_match
    )
    return {
        "scenario": scenario,
        "batch_size": batch_size,
        "tokens": int(key.shape[2]),
        "page_size": int(page_size),
        "top_pages": int(top_pages),
        "max_tokens": int(max_tokens),
        "elapsed_ms": elapsed_ms,
        "retrieved_counts": retrieved_counts,
        "expected_positions": [int(item) for item in expected_positions.tolist()],
        "position_hit_rate": hit_rate,
        "top1_hit_rate": top1_hit_rate,
        "marker_hit_rate": marker_hit_rate,
        "owner_clean": owner_clean,
        "cross_sample_leak": cross_sample_leak,
        "future_leak": not no_future_leak,
        "batch_loop_position_match": bool(batch_loop_match),
        "passed": passed,
    }


def _run_model_call_chain_smoke(device: torch.device) -> dict[str, object]:
    """Verify the multi-layer model still activates retrieval for batch>1.

    中文说明:
    - 调用方 / Called by: `run_quality_smoke`.
    - 调用对象 / Calls: `MultiLayerMHDSRA2Model.forward_selected_logits`.
    - 作用 / Purpose: 在真实多层调用链上确认 batch>1 不再禁用 retrieval，且 mask 被传进核心层。
    - 返回 / Returns: report dict with observed retrieval calls.
    - 错误处理 / Error handling: model exception is allowed to propagate to the smoke runner.

    English documentation:
    Function name:
        _run_model_call_chain_smoke
    Purpose:
        Confirm the model-level call chain passes batched retrieved masks into
        MHDSRA2 layers.
    """
    torch.manual_seed(1234)
    model = MultiLayerMHDSRA2Model(
        vocab_size=48,
        dim=16,
        num_layers=2,
        K=8,
        kr=2,
        chunk_size=8,
        use_retrieval=True,
    ).to(device)
    records: list[dict[str, object]] = []
    originals = []
    for layer_idx, layer in enumerate(model.layers):
        original = layer._retrieval_attention
        originals.append((layer, original))

        def _wrapped(query, retrieved_k, retrieved_v, retrieved_mask=None, *, _idx=layer_idx, _orig=original):
            if retrieved_k is not None:
                records.append(
                    {
                        "layer": _idx,
                        "retrieved_shape": list(retrieved_k.shape),
                        "mask_shape": None if retrieved_mask is None else list(retrieved_mask.shape),
                        "mask_any": None if retrieved_mask is None else bool(retrieved_mask.any().item()),
                    }
                )
            return _orig(query, retrieved_k, retrieved_v, retrieved_mask)

        layer._retrieval_attention = _wrapped
    try:
        tokens = torch.arange(0, 32, device=device, dtype=torch.long).view(1, -1).repeat(4, 1)
        tokens = (tokens + torch.arange(4, device=device).view(-1, 1)) % 48
        logits = model.forward_selected_logits(tokens, torch.full((4,), 31, device=device))
    finally:
        for layer, original in originals:
            layer._retrieval_attention = original
    return {
        "scenario": "model_call_chain",
        "batch_size": 4,
        "retrieval_call_count": len(records),
        "mask_call_count": sum(1 for item in records if item["mask_shape"] is not None),
        "retrieval_records": records,
        "logits_shape": list(logits.shape),
        "passed": bool(
            records
            and any(item["mask_any"] is True for item in records)
            and all(
                item["mask_shape"] is not None
                and item["retrieved_shape"][0] == 4
                and item["mask_shape"][0] == 4
                for item in records
            )
        ),
    }


def run_quality_smoke(
    *,
    batch_sizes: Iterable[int] = (1, 4, 8),
    tokens: Iterable[int] = (256, 1024),
    page_size: int = 64,
    top_pages: int = 4,
    max_tokens: int = 8,
    device: torch.device | str = "auto",
    seed: int = 20260602,
) -> dict[str, object]:
    """Run synthetic retrieval quality probes and return a structured payload.

    中文说明:
    - 调用方 / Called by: CLI `main` and pytest regression tests.
    - 调用对象 / Calls: `PagedExactMemory`, `_run_model_call_chain_smoke`.
    - 作用 / Purpose: 用 NIAH-like 和 JSON-like 小场景验证 batch retrieval 的召回质量、隔离性和调用链。
    - 参数 / Parameters: batch/token/page retrieval grid and device/seed.
    - 返回 / Returns: JSON-serializable dict written by the CLI.
    - 错误处理 / Error handling: invalid grid raises `ValueError`; retrieval failures are encoded as `passed=False`.
    - 副作用 / Side effects: allocates temporary tensors only; no file writes unless `main` persists reports.

    English documentation:
    Function name:
        run_quality_smoke
    Purpose:
        Execute deterministic NIAH-like and JSON-like exact retrieval probes for
        batched paged memory.
    """
    resolved_device = _resolve_device(str(device)) if not isinstance(device, torch.device) else device
    torch.manual_seed(seed)
    batch_values = _parse_int_list(batch_sizes)
    token_values = _parse_int_list(tokens)
    heads = 2
    dim = max(32, max(batch_values) * 4 + 16)
    cases: list[dict[str, object]] = []

    for batch_size in batch_values:
        for token_count in token_values:
            scenarios = (
                ("niah_single_needle", _build_niah_probe),
                ("json_latest_field", _build_json_probe),
                ("future_cutoff", _build_future_cutoff_probe),
            )
            for scenario, builder in scenarios:
                build_kwargs = {
                    "batch_size": batch_size,
                    "tokens": token_count,
                    "heads": heads,
                    "dim": dim,
                }
                if scenario == "json_latest_field":
                    build_kwargs["field_count"] = 6
                built = builder(**build_kwargs)
                if scenario == "future_cutoff":
                    key, value, query, expected_positions, max_position_cpu = built
                else:
                    key, value, query, expected_positions = built
                    max_position_cpu = torch.full(
                        (batch_size,),
                        token_count,
                        dtype=torch.long,
                    )
                key = key.to(resolved_device)
                value = value.to(resolved_device)
                query = query.to(resolved_device)
                expected_positions = expected_positions.to(resolved_device)
                max_position = max_position_cpu.to(resolved_device)
                start = time.perf_counter()
                _, retrieved_v, positions, mask = _retrieve_once(
                    key=key,
                    value=value,
                    query=query,
                    page_size=page_size,
                    top_pages=top_pages,
                    max_tokens=max_tokens,
                    max_position=max_position,
                    device=resolved_device,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                batch_loop_match = False
                if retrieved_v is not None and positions is not None and mask is not None:
                    batch_loop_match = _compare_with_batch_one_loop(
                        key=key,
                        value=value,
                        query=query,
                        retrieved_v=retrieved_v,
                        positions=positions,
                        mask=mask,
                        page_size=page_size,
                        top_pages=top_pages,
                        max_tokens=max_tokens,
                        max_position=max_position,
                    )
                cases.append(
                    _score_probe(
                        scenario=scenario,
                        key=key,
                        value=value,
                        query=query,
                        expected_positions=expected_positions,
                        page_size=page_size,
                        top_pages=top_pages,
                        max_tokens=max_tokens,
                        max_position=max_position,
                        elapsed_ms=elapsed_ms,
                        retrieved_v=retrieved_v,
                        positions=positions,
                        mask=mask,
                        batch_loop_match=batch_loop_match,
                    )
                )

    model_smoke = _run_model_call_chain_smoke(resolved_device)
    passed = bool(all(case["passed"] for case in cases) and model_smoke["passed"])
    return {
        "name": "mhdsra2_batch_retrieval_quality_smoke",
        "seed": int(seed),
        "device": str(resolved_device),
        "config": {
            "batch_sizes": batch_values,
            "tokens": token_values,
            "page_size": int(page_size),
            "top_pages": int(top_pages),
            "max_tokens": int(max_tokens),
        },
        "cases": cases,
        "model_call_chain": model_smoke,
        "summary": {
            "passed": passed,
            "case_count": len(cases),
            "passed_cases": sum(1 for case in cases if case["passed"]),
            "no_cross_sample_leak": not any(case["cross_sample_leak"] for case in cases),
            "no_future_leak": not any(case["future_leak"] for case in cases),
            "all_batch_loop_positions_match": all(
                case["batch_loop_position_match"] for case in cases
            ),
        },
    }


def build_markdown_report(payload: dict[str, object]) -> str:
    """Render the smoke payload as a compact Markdown report."""
    summary = payload["summary"]
    lines = [
        "# MHDSRA2 Batch Retrieval Quality Smoke",
        "",
        f"- device: `{payload['device']}`",
        f"- seed: `{payload['seed']}`",
        f"- passed: `{summary['passed']}`",
        f"- cases: `{summary['passed_cases']}/{summary['case_count']}`",
        f"- no_cross_sample_leak: `{summary['no_cross_sample_leak']}`",
        f"- no_future_leak: `{summary['no_future_leak']}`",
        f"- batch_loop_positions_match: `{summary['all_batch_loop_positions_match']}`",
        "",
        "## Cases",
        "",
        "| scenario | B | T | hit | top1 | marker | owner | counts | loop_match | ms | passed |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---:|---|",
    ]
    for case in payload["cases"]:
        lines.append(
            "| {scenario} | {batch_size} | {tokens} | {position_hit_rate:.3f} | "
            "{top1_hit_rate:.3f} | {marker_hit_rate:.3f} | {owner_clean} | "
            "`{retrieved_counts}` | {batch_loop_position_match} | "
            "{elapsed_ms:.3f} | {passed} |".format(**case)
        )
    model_smoke = payload["model_call_chain"]
    lines.extend(
        [
            "",
            "## Model Call Chain",
            "",
            f"- passed: `{model_smoke['passed']}`",
            f"- retrieval_call_count: `{model_smoke['retrieval_call_count']}`",
            f"- mask_call_count: `{model_smoke['mask_call_count']}`",
            f"- logits_shape: `{model_smoke['logits_shape']}`",
            "",
            "说明：本报告验证 external paged memory 的 batch 隔离、召回位置、latest-wins "
            "和多层调用链 mask 传递；它不是训练后的 NIAH/JSON 任务准确率报告。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the quality smoke command."""
    parser = argparse.ArgumentParser(
        description="Run synthetic MHDSRA2 batch retrieval quality smoke checks."
    )
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--tokens", default="256,1024")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--top-pages", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--json-out", default="reports/mhdsra2_batch_retrieval_quality_smoke.json")
    parser.add_argument("--markdown-out", default="reports/mhdsra2_batch_retrieval_quality_smoke.md")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    """Run CLI command, persist reports, and return the payload for tests."""
    args = build_parser().parse_args(argv)
    payload = run_quality_smoke(
        batch_sizes=_parse_int_list(args.batch_sizes),
        tokens=_parse_int_list(args.tokens),
        page_size=args.page_size,
        top_pages=args.top_pages,
        max_tokens=args.max_tokens,
        device=args.device,
        seed=args.seed,
    )
    json_path = PROJECT_ROOT / args.json_out
    markdown_path = PROJECT_ROOT / args.markdown_out
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(build_markdown_report(payload), encoding="utf-8")
    print(f"Wrote JSON report: {json_path}")
    print(f"Wrote Markdown report: {markdown_path}")
    print(f"passed={payload['summary']['passed']}")
    if not payload["summary"]["passed"]:
        raise SystemExit(1)
    return payload


if __name__ == "__main__":
    main()
