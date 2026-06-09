from __future__ import annotations

import warnings

import pytest
import torch

from src.dsra.dsra_layer import DSRA_Chunk_Layer
from src.dsra.domain import select_mhdsra2_heads
from src.dsra.dsra_model import MultiLayerMHDSRA2Model, select_mhdsra2_heads as model_heads
from src.dsra.infrastructure.paged_memory_repository import PagedMemoryRepository
from src.dsra.mhdsra2.improved_dsra_mha import MHDSRA2Config, MHDSRA2State, MultiHeadDSRA2
from src.dsra.mhdsra2.paged_exact_memory import PagedExactMemory


def test_paged_exact_memory_reset_and_clear_restart_positions() -> None:
    """Ensure external memory can be cleared between independent streams.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `PagedExactMemory.append`, `reset`, `clear`.
    - 作用 / Purpose: 回归保护底层分页记忆可清空，避免训练序列 K/V 泄漏到后续评估样本。
    - 变量 / Variables: `key/value` 是一个小型 deterministic K/V chunk。
    - 接入 / Integration: 覆盖 `PagedMemoryRepository.reset` 的底层依赖。
    - 错误处理 / Error handling: 断言失败直接暴露回归。
    - 副作用 / Side effects: 仅创建内存对象。
    """
    memory = PagedExactMemory(page_size=2, dtype=torch.float32)
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 4)

    memory.append(key, value)
    assert len(memory) == 2
    assert memory.next_position == 3

    memory.reset()
    assert len(memory) == 0
    assert memory.next_position == 0

    memory.append(key[:, :, :1, :], value[:, :, :1, :])
    assert len(memory) == 1
    assert memory.next_position == 1

    memory.clear()
    assert len(memory) == 0
    assert memory.next_position == 0


def test_repository_reset_keeps_memory_object_and_clears_pages() -> None:
    repository = PagedMemoryRepository(enabled=True, page_size=2, dtype=torch.float32)
    memory_object_id = id(repository.memory)
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 4)

    repository.append(key, value)
    repository.reset()

    assert id(repository.memory) == memory_object_id
    assert len(repository.memory) == 0
    assert repository.memory.next_position == 0


def test_retrieve_max_position_filters_future_tokens() -> None:
    memory = PagedExactMemory(page_size=8, dtype=torch.float32)
    key = torch.zeros(1, 1, 8, 4)
    value = torch.zeros(1, 1, 8, 4)
    key[:, :, 1, 0] = 1.0
    key[:, :, 6, 0] = 1.0
    value[:, :, 1, 1] = 1.0
    value[:, :, 6, 2] = 1.0
    memory.append(key, value)

    query = key[:, :, 6:7, :]
    _, retrieved_v, positions = memory.retrieve(
        query,
        top_pages=1,
        max_tokens=4,
        max_position=4,
    )

    assert positions is not None
    assert retrieved_v is not None
    assert all(int(position) < 4 for position in positions.tolist())
    assert 6 not in positions.tolist()
    assert 1 in positions.tolist()


def test_paged_exact_memory_default_does_not_prune_pages() -> None:
    memory = PagedExactMemory(page_size=1, dtype=torch.float32)
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 4)

    memory.append(key, value)

    assert len(memory) == 3
    assert memory.next_position == 3


def test_paged_exact_memory_max_pages_prunes_oldest_pages() -> None:
    memory = PagedExactMemory(page_size=1, dtype=torch.float32, max_pages=2)
    key = torch.randn(1, 1, 4, 4)
    value = torch.randn(1, 1, 4, 4)

    memory.append(key, value)

    assert len(memory) == 2
    assert memory.next_position == 4
    assert [(page.start, page.end) for page in memory.pages] == [(2, 3), (3, 4)]


def test_paged_exact_memory_max_pages_prefers_invalid_page_pruning() -> None:
    memory = PagedExactMemory(page_size=1, dtype=torch.float32, max_pages=3)
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 4)
    memory.append(key, value)
    memory.invalidate_before(2)

    memory.append(torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4))

    assert len(memory) == 3
    assert memory.next_position == 4
    assert [(page.start, page.end, page.valid) for page in memory.pages] == [
        (1, 2, False),
        (2, 3, True),
        (3, 4, True),
    ]


def test_paged_memory_repository_passes_max_pages_to_memory() -> None:
    repository = PagedMemoryRepository(
        enabled=True,
        page_size=1,
        dtype=torch.float32,
        max_pages=2,
    )
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 4)

    repository.append(key, value)

    assert len(repository.memory) == 2
    assert repository.memory.next_position == 3


def test_slot_write_allows_same_key_value_overwrite() -> None:
    """Same-key corrections must not be blocked by zero novelty.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `MultiHeadDSRA2._slot_write`。
    - 作用 / Purpose: 回归保护同一个 key 写入新 value 时仍能产生写入质量和写入门。
    - 错误处理 / Error handling: 如果 `novelty=0` 再次清零写入门，断言会失败。
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
    old_value = torch.tensor([[[[1.0, 0.0]]]])
    new_value = torch.tensor([[[[0.0, 1.0]]]])
    state = MHDSRA2State(
        slot_k=same_key.clone(),
        slot_v=old_value.clone(),
        age=torch.zeros(1, 1, 1),
        usage=torch.zeros(1, 1, 1),
        confidence=torch.ones(1, 1, 1),
        position=0,
    )

    next_state = layer._slot_write(same_key, new_value, state, torch.ones(1, 1, 1))

    assert float(layer.last_write_stats["novelty_mean"]) == 0.0
    assert float(layer.last_write_stats["write_mass_max"]) > 0.0
    assert float(layer.last_write_stats["write_gate_max"]) > 0.0
    assert float(layer.last_write_stats["overwrite_gate_mean"]) > 0.0
    assert next_state.slot_v[0, 0, 0, 1] > state.slot_v[0, 0, 0, 1]
    assert torch.isfinite(next_state.slot_v).all()


def test_slot_write_protection_uses_topk_gather_without_changing_behavior() -> None:
    cfg = MHDSRA2Config(
        dim=2,
        heads=1,
        slots=1,
        read_topk=1,
        write_topk=1,
        use_local=False,
        use_retrieval=False,
        forget_base=0.0,
        write_protection=4,
        detach_state=False,
    )
    layer = MultiHeadDSRA2(cfg)
    with torch.no_grad():
        layer.token_write_gate.weight.zero_()
        layer.token_write_gate.bias.fill_(10.0)

    state = MHDSRA2State(
        slot_k=torch.tensor([[[[1.0, 0.0]]]]),
        slot_v=torch.tensor([[[[1.0, 0.0]]]]),
        age=torch.zeros(1, 1, 1),
        usage=torch.zeros(1, 1, 1),
        confidence=torch.ones(1, 1, 1),
        position=1,
        protected_until=torch.full((1, 1, 1), 10, dtype=torch.long),
    )

    next_state = layer._slot_write(
        torch.tensor([[[[1.0, 0.0]]]]),
        torch.tensor([[[[0.0, 1.0]]]]),
        state,
        torch.ones(1, 1, 1),
    )

    assert float(layer.last_write_stats["write_mass_max"]) == 0.0
    torch.testing.assert_close(next_state.slot_v, state.slot_v)


def test_paged_exact_memory_finds_single_key_token_diluted_by_page_mean() -> None:
    """A single matching token should survive page-level filtering.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `PagedExactMemory.append`, `PagedExactMemory.retrieve`。
    - 作用 / Purpose: 防止页均值 summary 把页内唯一关键 token 稀释后漏召回。
    - 错误处理 / Error handling: 如果 top page 仍只看 mean summary，会返回错误页位置。
    """
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

    query = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    _, _, positions = memory.retrieve(query, top_pages=1, max_tokens=1)

    assert positions is not None
    assert positions.tolist() == [0]


def test_paged_memory_repository_batch_gt_one_retrieves_isolated_rows() -> None:
    """Batch rows should retrieve only their own external-memory tokens.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `PagedMemoryRepository.append`, `PagedMemoryRepository.retrieve`。
    - 作用 / Purpose: 回归保护 batch-isolated memory，避免 batch 内样本互相召回 K/V。
    - 错误处理 / Error handling: shape、mask 或 value marker 不匹配都会触发断言。
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
    retrieved_k, retrieved_v, retrieved_mask = repository.retrieve(
        key,
        device=key.device,
        return_mask=True,
    )

    assert retrieved_k is not None
    assert retrieved_v is not None
    assert retrieved_mask is not None
    assert retrieved_k.shape == (2, 1, 1, 4)
    assert retrieved_v.shape == (2, 1, 1, 4)
    assert retrieved_mask.tolist() == [[True], [True]]
    assert float(retrieved_v[0, 0, 0, 2]) == 10.0
    assert float(retrieved_v[0, 0, 0, 3]) == 0.0
    assert float(retrieved_v[1, 0, 0, 2]) == 0.0
    assert float(retrieved_v[1, 0, 0, 3]) == 20.0


def test_paged_memory_repository_batch_padding_requires_mask_for_default_api() -> None:
    """The default repository API should not hide padded batch retrieval rows.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `PagedMemoryRepository.retrieve`。
    - 作用 / Purpose: 防止 batch>1 默认无 mask 返回把零 padding 误当有效 retrieval token。
    - 错误处理 / Error handling: 有 padding 时未抛 ValueError，或 return_mask=True 路径失败都会触发断言。
    """
    repository = PagedMemoryRepository(
        enabled=True,
        page_size=2,
        dtype=torch.float32,
        max_tokens=1,
    )
    key = torch.zeros(2, 1, 2, 4)
    value = torch.zeros(2, 1, 2, 4)
    key[0, 0, :, 0] = 1.0
    key[1, 0, :, 1] = 1.0
    value[0, 0, :, 2] = 10.0
    value[1, 0, :, 3] = 20.0
    repository.append(key, value)

    with pytest.raises(ValueError, match="return_mask=True"):
        repository.retrieve(
            key[:, :, -1:, :],
            device=key.device,
            max_position=torch.tensor([0, 2]),
        )

    _, retrieved_v, retrieved_mask = repository.retrieve(
        key[:, :, -1:, :],
        device=key.device,
        max_position=torch.tensor([0, 2]),
        return_mask=True,
    )
    assert retrieved_v is not None
    assert retrieved_mask is not None
    assert retrieved_mask.tolist() == [[False], [True]]


def test_paged_exact_memory_batch_size_change_requires_reset() -> None:
    """A stream cannot silently change batch size without reset.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `PagedExactMemory.append`, `PagedExactMemory.reset`。
    - 作用 / Purpose: 防止同一个外部记忆 stream 在 batch size 改变后串接旧样本。
    - 错误处理 / Error handling: batch size 变化必须抛 ValueError，reset 后才能继续。
    """
    memory = PagedExactMemory(page_size=2, dtype=torch.float32)
    memory.append(torch.randn(2, 1, 1, 4), torch.randn(2, 1, 1, 4))

    with pytest.raises(ValueError, match="reset"):
        memory.append(torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4))

    memory.reset()
    memory.append(torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4))
    assert memory.batch_size == 1
    assert memory.next_position == 1


def test_paged_exact_memory_batch_max_position_returns_padding_mask() -> None:
    """Per-sample max_position should filter future tokens without cross-row leakage.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `PagedExactMemory.append`, `PagedExactMemory.retrieve`。
    - 作用 / Purpose: 验证 batch 内某个样本无有效召回时返回 pos=-1/mask=False padding。
    - 错误处理 / Error handling: 未来 token 泄漏、mask 或 value marker 错误都会触发断言。
    """
    memory = PagedExactMemory(page_size=2, dtype=torch.float32)
    key = torch.zeros(2, 1, 2, 4)
    value = torch.zeros(2, 1, 2, 4)
    key[0, 0, :, 0] = 1.0
    key[1, 0, :, 1] = 1.0
    value[0, 0, :, 2] = 10.0
    value[1, 0, :, 3] = 20.0
    memory.append(key, value)

    _, retrieved_v, positions, retrieved_mask = memory.retrieve(
        key[:, :, -1:, :],
        top_pages=1,
        max_tokens=1,
        max_position=torch.tensor([0, 2]),
        return_mask=True,
    )

    assert retrieved_v is not None
    assert positions is not None
    assert retrieved_mask is not None
    assert positions.tolist() == [[-1], [1]]
    assert retrieved_mask.tolist() == [[False], [True]]
    assert float(retrieved_v[0, 0, 0, 2]) == 0.0
    assert float(retrieved_v[1, 0, 0, 3]) == 20.0


def test_paged_exact_memory_max_pages_prunes_per_batch_sample() -> None:
    """max_pages should be enforced independently for each batch row.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `PagedExactMemory.append`。
    - 作用 / Purpose: 确认 batch-isolated memory 的页裁剪不会跨样本共享计数。
    - 错误处理 / Error handling: 任一样本页数或位置错误都会触发断言。
    """
    memory = PagedExactMemory(page_size=1, dtype=torch.float32, max_pages=2)
    key = torch.randn(2, 1, 4, 4)
    value = torch.randn(2, 1, 4, 4)

    memory.append(key, value)

    assert memory.next_positions == [4, 4]
    assert [len(pages) for pages in memory.pages_by_sample] == [2, 2]
    assert [
        [(page.start, page.end) for page in pages]
        for pages in memory.pages_by_sample
    ] == [
        [(2, 3), (3, 4)],
        [(2, 3), (3, 4)],
    ]


def test_paged_exact_memory_invalidate_before_is_per_batch_sample() -> None:
    """invalidate_before should not invalidate pages across batch rows.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `PagedExactMemory.append`, `PagedExactMemory.invalidate_before`。
    - 作用 / Purpose: 确认每个样本按自己的 position 失效旧页，不会跨样本影响 page.valid。
    - 错误处理 / Error handling: 任一样本 valid 标记错误都会触发断言。
    """
    memory = PagedExactMemory(page_size=1, dtype=torch.float32)
    key = torch.randn(2, 1, 2, 4)
    value = torch.randn(2, 1, 2, 4)
    memory.append(key, value)

    memory.invalidate_before(torch.tensor([1, 0]))

    assert [[page.valid for page in pages] for pages in memory.pages_by_sample] == [
        [False, True],
        [True, True],
    ]


def test_paged_exact_memory_profile_does_not_change_batch_results() -> None:
    """Profiling should be diagnostic only and preserve retrieve outputs.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `PagedExactMemory.retrieve`。
    - 作用 / Purpose: 确认 `profile=True` 只写入 `last_retrieve_profile`，不改变 K/V/pos/mask。
    - 错误处理 / Error handling: 返回值不一致或 profile 字段缺失都会触发断言。
    """
    memory = PagedExactMemory(page_size=2, dtype=torch.float32)
    key = torch.zeros(2, 1, 4, 4)
    value = torch.zeros(2, 1, 4, 4)
    key[0, 0, :, 0] = 1.0
    key[1, 0, :, 1] = 1.0
    value[0, 0, :, 2] = torch.arange(4, dtype=torch.float32)
    value[1, 0, :, 3] = torch.arange(10, 14, dtype=torch.float32)
    memory.append(key, value)

    plain = memory.retrieve(key[:, :, -1:, :], top_pages=2, max_tokens=2, return_mask=True)
    profiled = memory.retrieve(
        key[:, :, -1:, :],
        top_pages=2,
        max_tokens=2,
        return_mask=True,
        profile=True,
    )

    assert memory.last_retrieve_profile is not None
    assert memory.last_retrieve_profile["page_score_mode"] == "vectorized"
    assert memory.last_retrieve_profile["batch_size"] == 2
    assert memory.last_retrieve_profile["retrieved_token_counts"] == [2, 2]
    for plain_item, profiled_item in zip(plain, profiled):
        assert plain_item is not None
        assert profiled_item is not None
        torch.testing.assert_close(plain_item, profiled_item)


def test_paged_exact_memory_vectorized_page_scores_match_fallback() -> None:
    """Vectorized and fallback page scoring should select the same retrieval tokens.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `PagedExactMemory.retrieve`。
    - 作用 / Purpose: 保护向量化 page scoring 不改变 exact retrieval 语义。
    - 错误处理 / Error handling: positions、mask 或 value 不一致都会触发断言。
    """
    memory = PagedExactMemory(page_size=2, dtype=torch.float32)
    key = torch.zeros(2, 1, 5, 4)
    value = torch.zeros(2, 1, 5, 4)
    key[0, 0, :, 0] = torch.tensor([1.0, -1.0, 0.8, 1.0, -0.5])
    key[1, 0, :, 1] = torch.tensor([0.5, 1.0, -1.0, 0.7, 1.0])
    value[0, 0, :, 2] = torch.arange(5, dtype=torch.float32)
    value[1, 0, :, 3] = torch.arange(20, 25, dtype=torch.float32)
    memory.append(key, value)
    query = torch.zeros(2, 1, 1, 4)
    query[0, 0, 0, 0] = 1.0
    query[1, 0, 0, 1] = 1.0

    vectorized = memory.retrieve(
        query,
        top_pages=2,
        max_tokens=3,
        max_position=torch.tensor([5, 4]),
        return_mask=True,
        profile=True,
    )
    assert memory.last_retrieve_profile is not None
    assert memory.last_retrieve_profile["page_score_mode"] == "vectorized"
    original_limit = memory._VECTOR_PAGE_SCORE_MAX_ELEMENTS
    memory._VECTOR_PAGE_SCORE_MAX_ELEMENTS = 1
    try:
        fallback = memory.retrieve(
            query,
            top_pages=2,
            max_tokens=3,
            max_position=torch.tensor([5, 4]),
            return_mask=True,
            profile=True,
        )
    finally:
        memory._VECTOR_PAGE_SCORE_MAX_ELEMENTS = original_limit

    assert memory.last_retrieve_profile is not None
    assert memory.last_retrieve_profile["page_score_mode"] == "sample_loop"
    for vectorized_item, fallback_item in zip(vectorized, fallback):
        assert vectorized_item is not None
        assert fallback_item is not None
        torch.testing.assert_close(vectorized_item, fallback_item)


def test_paged_exact_memory_query_pooling_default_is_mean() -> None:
    """Default retrieval query pooling should preserve the historical mean path."""
    memory = PagedExactMemory(page_size=2, dtype=torch.float32)

    assert memory.query_pooling == "mean"


def test_paged_exact_memory_max_token_query_pooling_uses_strongest_query_token() -> None:
    """max_token pooling should let one query token select its matching page."""
    memory = PagedExactMemory(page_size=2, dtype=torch.float32, query_pooling="max_token")
    key = torch.tensor(
        [[
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]]
    )
    value = key.clone()
    memory.append(key, value)
    query = torch.tensor(
        [[
            [1.0, 0.0],
            [0.0, -1.0],
        ]]
    )

    _, _, positions = memory.retrieve(query, top_pages=1, max_tokens=1)

    assert positions is not None
    assert int(positions[0].item()) in {0, 1}


def test_paged_exact_memory_rejects_unknown_query_pooling() -> None:
    with pytest.raises(ValueError, match="query_pooling"):
        PagedExactMemory(page_size=2, dtype=torch.float32, query_pooling="bad")


def test_dsra_chunk_layer_forward_step_uses_single_fast_qkv_projection() -> None:
    """Compatibility forward_step should compute fast QKV once.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `DSRA_Chunk_Layer.forward_step`。
    - 作用 / Purpose: 防止逐 token 解码路径重复执行 `core.qkv(x_t)`。
    - 错误处理 / Error handling: 调用次数、输出形状或外部 memory 写入异常都会触发断言。
    """
    torch.manual_seed(0)
    layer = DSRA_Chunk_Layer(dim=8, K=4, kr=1, use_bypass=True)
    call_count = {"qkv": 0}
    original_forward = layer.core.qkv.forward

    def counted_forward(x: torch.Tensor) -> torch.Tensor:
        call_count["qkv"] += 1
        return original_forward(x)

    layer.core.qkv.forward = counted_forward
    out, state, kv_cache = layer.forward_step(torch.randn(1, 1, 8), None, None)

    assert call_count["qkv"] == 1
    assert out.shape == (1, 1, 8)
    assert state.position == 1
    assert kv_cache[0] is not None
    assert kv_cache[1] is not None
    assert kv_cache[0].shape == (1, 1, 8)
    assert kv_cache[1].shape == (1, 1, 8)
    assert layer.memory_repository.memory.next_position == 1


def test_dsra_chunk_layer_forward_step_rejects_multi_token_input() -> None:
    layer = DSRA_Chunk_Layer(dim=8, K=4, kr=1, use_bypass=True)

    with pytest.raises(ValueError, match="one token"):
        layer.forward_step(torch.randn(1, 2, 8), None, None)


def test_forward_step_kv_cache_does_not_mutate_input_state() -> None:
    layer = DSRA_Chunk_Layer(dim=8, K=4, kr=1, use_bypass=True)
    state = layer._coerce_state(None, 1, torch.device("cpu"), torch.float32)
    cached_k = torch.randn(1, layer.core.heads, 1, layer.core.d_head)
    cached_v = torch.randn(1, layer.core.heads, 1, layer.core.d_head)

    layer.forward_step(torch.randn(1, 1, 8), state, (cached_k, cached_v))

    assert state.local_k is None
    assert state.local_v is None


def test_dsra_chunk_layer_batch_gt_one_external_memory_is_active() -> None:
    layer = DSRA_Chunk_Layer(dim=8, K=4, kr=1, use_bypass=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out, state, kv_cache, _ = layer(torch.randn(2, 3, 8))

    assert out.shape == (2, 3, 8)
    assert state.position == 3
    assert kv_cache[0] is not None
    assert not any("batch_size>1" in str(item.message) for item in caught)
    assert layer.memory_repository.memory.next_positions == [3, 3]
    assert [len(pages) for pages in layer.memory_repository.memory.pages_by_sample] == [1, 1]
    assert layer.last_external_memory_diagnostic["status"] == "active"
    assert layer.last_external_memory_diagnostic["batch_size"] == 2


def test_dsra_chunk_layer_auto_resets_external_memory_for_new_sequence() -> None:
    torch.manual_seed(0)
    layer = DSRA_Chunk_Layer(dim=8, K=4, kr=1, use_bypass=False)
    first_sequence = torch.randn(1, 3, 8)
    second_sequence = torch.randn(1, 2, 8)

    layer(first_sequence, chunk_idx=0)
    assert layer.memory_repository.memory.next_position == 3

    layer(second_sequence, chunk_idx=0)
    assert layer.memory_repository.memory.next_position == 2


def test_shared_head_selector_wrappers_match_domain_helper() -> None:
    assert model_heads(32) == select_mhdsra2_heads(32)
    assert DSRA_Chunk_Layer(dim=32, K=4, kr=1).core.heads == select_mhdsra2_heads(32)


def test_multilayer_update_momentum_calls_each_layer() -> None:
    model = MultiLayerMHDSRA2Model(
        vocab_size=16,
        dim=16,
        num_layers=2,
        K=4,
        kr=1,
        chunk_size=2,
        mhdsra2_config_override={"momentum_qkv": True},
    )
    called = [0, 0]

    for idx, layer in enumerate(model.layers):
        def _mark_call(index: int = idx) -> None:
            called[index] += 1

        layer.update_momentum = _mark_call

    model.update_momentum()

    assert called == [1, 1]


def test_batch_retrieval_quality_smoke_core_passes_on_small_grid() -> None:
    """Synthetic quality smoke should detect no cross-sample or future leakage.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `scripts.mhdsra2_batch_retrieval_quality_smoke.run_quality_smoke`.
    - 作用 / Purpose: 保护新增质量验收脚本的核心判定，确保短网格下 NIAH-like、
      JSON-like 和多层模型调用链都能通过。
    - 错误处理 / Error handling: 任一 case 未通过、串样本或未来泄漏都会触发断言。

    English documentation:
    Function name:
        test_batch_retrieval_quality_smoke_core_passes_on_small_grid
    Purpose:
        Keep the synthetic batch retrieval quality smoke executable and strict.
    """
    from scripts.mhdsra2_batch_retrieval_quality_smoke import run_quality_smoke

    payload = run_quality_smoke(
        batch_sizes=(1, 2),
        tokens=(64,),
        page_size=16,
        top_pages=2,
        max_tokens=4,
        device=torch.device("cpu"),
    )

    assert payload["summary"]["passed"] is True
    assert payload["summary"]["no_cross_sample_leak"] is True
    assert payload["summary"]["no_future_leak"] is True
    assert payload["summary"]["all_batch_loop_positions_match"] is True
    assert payload["model_call_chain"]["passed"] is True
    assert {case["scenario"] for case in payload["cases"]} == {
        "niah_single_needle",
        "json_latest_field",
        "future_cutoff",
    }
    assert all(case["top1_hit_rate"] == 1.0 for case in payload["cases"])
    assert all(case["owner_clean"] is True for case in payload["cases"])
