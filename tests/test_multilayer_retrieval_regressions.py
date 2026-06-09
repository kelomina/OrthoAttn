"""Regression tests for MultiLayerMHDSRA2 retrieval wiring."""

from __future__ import annotations

import pytest
import torch

from src.dsra.infrastructure import PagedMemoryRepository
from src.dsra.dsra_model import MultiLayerMHDSRA2Model


def _build_model(*, use_retrieval: bool = True, batch_safe: bool = True) -> MultiLayerMHDSRA2Model:
    """Create a tiny deterministic model for retrieval wiring tests.

    中文说明:
    - 调用方 / Called by: 本文件内 pytest 测试。
    - 调用对象 / Calls: `MultiLayerMHDSRA2Model`。
    - 作用 / Purpose: 构造小规模多层模型，快速验证 chunk 间 retrieval 是否接通。
    - 参数 / Parameters: `use_retrieval` 控制外部分页召回分支；`batch_safe` 保留测试可读性。
    - 返回 / Returns: 可在 CPU 上快速运行的多层 MHDSRA2 token 模型。
    - 错误处理 / Error handling: 底层模型配置错误直接抛出。
    """
    batch_safe = bool(batch_safe)
    del batch_safe
    return MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=2,
        K=8,
        kr=2,
        chunk_size=4,
        use_retrieval=use_retrieval,
    )


def _record_retrieval_calls(model: MultiLayerMHDSRA2Model) -> list[dict[str, object]]:
    """Wrap each layer retrieval branch and return captured call metadata.

    中文说明:
    - 调用方 / Called by: 本文件内 pytest 测试。
    - 调用对象 / Calls: 每层 `MultiHeadDSRA2._retrieval_attention`。
    - 作用 / Purpose: 不改模型输出，只记录 `retrieved_k/retrieved_v` 是否真实传入。
    - 返回 / Returns: 每次 retrieval 分支调用的轻量元数据列表。
    - 副作用 / Side effects: 在测试模型实例上替换方法；仅限本测试局部对象。
    """
    records: list[dict[str, object]] = []
    for layer_idx, layer in enumerate(model.layers):
        original = layer._retrieval_attention

        def wrapped(
            q,
            retrieved_k,
            retrieved_v,
            retrieved_mask=None,
            *,
            _idx=layer_idx,
            _original=original,
        ):
            records.append(
                {
                    "layer_idx": _idx,
                    "retrieved_k_is_none": retrieved_k is None,
                    "retrieved_v_is_none": retrieved_v is None,
                    "retrieved_k_shape": None
                    if retrieved_k is None
                    else tuple(retrieved_k.shape),
                    "retrieved_mask_shape": None
                    if retrieved_mask is None
                    else tuple(retrieved_mask.shape),
                    "retrieved_mask_valid": None
                    if retrieved_mask is None
                    else int(retrieved_mask.sum().item()),
                }
            )
            return _original(q, retrieved_k, retrieved_v, retrieved_mask)

        layer._retrieval_attention = wrapped
    return records


def test_multilayer_forward_retrieves_from_prior_chunks_when_enabled() -> None:
    """`forward` should feed paged K/V retrieval after the first chunk.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `MultiLayerMHDSRA2Model.forward`。
    - 作用 / Purpose: 防止 `use_retrieval=True` 只改配置、不传入真实 retrieval K/V。
    - 错误处理 / Error handling: 第二个 chunk 仍无召回时断言失败。
    """
    torch.manual_seed(20260602)
    model = _build_model(use_retrieval=True)
    records = _record_retrieval_calls(model)
    tokens = torch.arange(12, dtype=torch.long).view(1, 12) % 32

    with torch.no_grad():
        logits = model(tokens)

    assert logits.shape == (1, 12, 32)
    assert len(records) == 6
    assert records[0]["retrieved_k_is_none"] is True
    assert records[1]["retrieved_k_is_none"] is True
    assert any(row["retrieved_k_is_none"] is False for row in records[2:])
    assert all(row["retrieved_v_is_none"] is False for row in records[2:])


def test_multilayer_selected_logits_retrieves_and_matches_full_forward() -> None:
    """Selected logits should keep full-forward semantics with retrieval enabled.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `forward`, `forward_selected_logits`。
    - 作用 / Purpose: 接通 retrieval 后，仍保证 NIAH 省显存 selected-logit 路径与全量前向一致。
    - 错误处理 / Error handling: 数值不一致或未发生召回时断言失败。
    """
    torch.manual_seed(20260602)
    model = _build_model(use_retrieval=True)
    records = _record_retrieval_calls(model)
    tokens = torch.arange(12, dtype=torch.long).view(1, 12) % 32
    positions = torch.tensor([10], dtype=torch.long)

    with torch.no_grad():
        full_logits = model(tokens)
        selected_logits = model.forward_selected_logits(tokens, positions)

    expected = full_logits[torch.arange(tokens.shape[0]), positions]
    torch.testing.assert_close(selected_logits, expected)
    assert any(row["retrieved_k_is_none"] is False for row in records)


def test_multilayer_retrieval_disabled_passes_no_external_kv() -> None:
    """Disabled retrieval should preserve the previous no-external-memory behavior.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `MultiLayerMHDSRA2Model.forward`。
    - 作用 / Purpose: 确认最小修复不改变 `use_retrieval=False` 的旧行为。
    - 错误处理 / Error handling: 禁用状态下传入外部 K/V 时断言失败。
    """
    torch.manual_seed(20260602)
    model = _build_model(use_retrieval=False)
    records = _record_retrieval_calls(model)
    tokens = torch.arange(12, dtype=torch.long).view(1, 12) % 32

    with torch.no_grad():
        logits = model(tokens)

    assert logits.shape == (1, 12, 32)
    assert records
    assert all(row["retrieved_k_is_none"] is True for row in records)
    assert all(row["retrieved_v_is_none"] is True for row in records)


def test_multilayer_retrieval_supports_batch_gt_one_with_masks() -> None:
    """The multilayer retrieval path should keep batch rows isolated.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `MultiLayerMHDSRA2Model.forward`。
    - 作用 / Purpose: 覆盖 batch>1 时 external paged memory 真实启用并携带 mask。
    - 错误处理 / Error handling: 输出 shape、retrieval shape 或 mask 错误都会触发断言。
    """
    torch.manual_seed(20260602)
    model = _build_model(use_retrieval=True)
    records = _record_retrieval_calls(model)
    tokens = torch.arange(24, dtype=torch.long).view(2, 12) % 32

    with torch.no_grad():
        logits = model(tokens)

    assert logits.shape == (2, 12, 32)
    retrieved_records = [row for row in records if row["retrieved_k_is_none"] is False]
    assert retrieved_records
    assert all(row["retrieved_k_shape"][0] == 2 for row in retrieved_records)
    assert all(row["retrieved_mask_shape"][0] == 2 for row in retrieved_records)
    assert all(row["retrieved_mask_valid"] > 0 for row in retrieved_records)


class _CapturingRepository(PagedMemoryRepository):
    """Paged memory repository that records max_position arguments in tests."""

    def __init__(self) -> None:
        super().__init__(enabled=True, page_size=4, dtype=torch.float32, max_tokens=2)
        self.max_position_calls: list[torch.Tensor | int | None] = []

    def retrieve(self, query, device, max_position=None, *, return_mask=False, profile=False):
        if isinstance(max_position, torch.Tensor):
            self.max_position_calls.append(max_position.detach().cpu().clone())
        else:
            self.max_position_calls.append(max_position)
        return super().retrieve(
            query,
            device,
            max_position=max_position,
            return_mask=return_mask,
            profile=profile,
        )


def test_multilayer_retrieval_uses_per_sample_sequence_lengths_for_max_position() -> None:
    """Padded batch retrieval should pass per-sample max_position to paged memory.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `MultiLayerMHDSRA2Model.forward`, `PagedMemoryRepository.retrieve`。
    - 作用 / Purpose: 防止 batch 内样本真实长度不同时，用单个 `state.position` 召回短样本的 pad/future token。
    - 错误处理 / Error handling: 第二个 chunk 未传 `[4, 2]` 这类逐样本边界时断言失败。
    """
    torch.manual_seed(20260602)
    model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=8,
        kr=2,
        chunk_size=4,
        use_retrieval=True,
    )
    repository = _CapturingRepository()
    model._new_retrieval_repositories = lambda: [repository]
    tokens = torch.arange(16, dtype=torch.long).view(2, 8) % 32
    sequence_lengths = torch.tensor([8, 2], dtype=torch.long)

    with torch.no_grad():
        logits = model(tokens, sequence_lengths=sequence_lengths)

    assert logits.shape == (2, 8, 32)
    assert len(repository.max_position_calls) == 2
    assert repository.max_position_calls[0].tolist() == [0, 0]
    assert repository.max_position_calls[1].tolist() == [4, 2]


def test_multilayer_selected_logits_rejects_positions_after_sequence_length() -> None:
    """Selected logits should reject positions outside each sample's true length."""
    model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=8,
        kr=2,
        chunk_size=4,
        use_retrieval=True,
    )
    tokens = torch.arange(16, dtype=torch.long).view(2, 8) % 32

    with pytest.raises(ValueError, match="before each sample"):
        model.forward_selected_logits(
            tokens,
            torch.tensor([7, 3], dtype=torch.long),
            sequence_lengths=torch.tensor([8, 2], dtype=torch.long),
        )
