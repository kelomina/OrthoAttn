from __future__ import annotations

import torch

from src.dsra.dsra_layer import DSRA_Chunk_Layer
from src.dsra.infrastructure.paged_memory_repository import PagedMemoryRepository
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


def test_dsra_chunk_layer_auto_resets_external_memory_for_new_sequence() -> None:
    torch.manual_seed(0)
    layer = DSRA_Chunk_Layer(dim=8, K=4, kr=1, use_bypass=False)
    first_sequence = torch.randn(1, 3, 8)
    second_sequence = torch.randn(1, 2, 8)

    layer(first_sequence, chunk_idx=0)
    assert layer.memory_repository.memory.next_position == 3

    layer(second_sequence, chunk_idx=0)
    assert layer.memory_repository.memory.next_position == 2

