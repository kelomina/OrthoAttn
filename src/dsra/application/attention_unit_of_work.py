"""Application unit-of-work boundary for streaming attention calls."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..infrastructure.paged_memory_repository import PagedMemoryRepository
from ..mhdsra2.improved_dsra_mha import MHDSRA2State


class StreamingAttentionUnitOfWork:
    """Coordinate state, local cache, and external memory for one forward call.

    中文说明:
    - 调用方 / Called by: `DSRA_Chunk_Layer.forward`
    - 调用对象 / Calls: `PagedMemoryRepository.retrieve`
    - 作用 / Purpose: 明确应用层工作单元边界，把一次流式前向的状态读取与提交集中管理
    - 变量 / Variables:
      `state` 是 MHDSRA2 流式状态, `kv_cache` 是旧接口兼容局部缓存,
      `time_state` 是旧 timestamps 兼容字段, `memory_repository` 是外部记忆仓储
    - 接入 / Integration: 使用 `with StreamingAttentionUnitOfWork(...) as uow` 包裹前向调用
    - 错误处理 / Error handling: 当前工作单元不吞异常，底层错误会向上抛出
    - 关键词 / Keywords:
      unit_of_work|application|state|cache|memory|forward|streaming|commit|retrieve|工作单元
    """

    def __init__(
        self,
        *,
        state: Optional[MHDSRA2State],
        kv_cache: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
        time_state: Optional[torch.Tensor],
        memory_repository: PagedMemoryRepository,
    ) -> None:
        """Create one streaming attention unit of work.

        中文说明:
        - 调用方 / Called by: `DSRA_Chunk_Layer.forward`
        - 调用对象 / Calls: 无外部函数
        - 作用 / Purpose: 捕获一次前向调用开始时的状态、缓存和外部记忆仓储
        - 变量 / Variables:
          `state/kv_cache/time_state` 是旧接口入参, `memory_repository` 是基础设施仓储
        - 接入 / Integration: 构造后进入上下文管理器并在成功前向后调用 `commit_forward`
        - 错误处理 / Error handling: 不验证张量内容，维度错误交给注意力层处理
        - 关键词 / Keywords:
          init|unit_of_work|state|kv_cache|time_state|memory_repository|application|forward|context|初始化
        """
        self.state = state
        self.kv_cache = kv_cache
        self.time_state = time_state
        self.memory_repository = memory_repository

    def __enter__(self) -> "StreamingAttentionUnitOfWork":
        """Enter the forward-call transaction boundary.

        中文说明:
        - 调用方 / Called by: Python `with` 语句
        - 调用对象 / Calls: 无外部函数
        - 作用 / Purpose: 返回当前工作单元供模型层读取状态与召回记忆
        - 变量 / Variables: `self` 是当前工作单元实例
        - 接入 / Integration: `with ... as uow` 后通过 `uow.retrieve(...)` 和字段访问接入
        - 错误处理 / Error handling: 进入阶段不吞异常
        - 关键词 / Keywords:
          enter|context|unit_of_work|application|transaction|state|memory|forward|with|进入
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        """Leave the forward-call transaction boundary.

        中文说明:
        - 调用方 / Called by: Python `with` 语句
        - 调用对象 / Calls: 无外部函数
        - 作用 / Purpose: 明确工作单元不拦截异常，保证错误向测试和调用方暴露
        - 变量 / Variables:
          `exc_type/exc_value/traceback` 是上下文管理器标准异常参数
        - 接入 / Integration: 上层无需手动调用，退出 `with` 时自动执行
        - 错误处理 / Error handling: 始终返回 `False`，不忽略任何异常
        - 关键词 / Keywords:
          exit|context|unit_of_work|exception|propagate|application|forward|transaction|with|退出
        """
        return False

    def retrieve(
        self, query: torch.Tensor, device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Retrieve external memory candidates through the repository.

        中文说明:
        - 调用方 / Called by: `DSRA_Chunk_Layer.forward`
        - 调用对象 / Calls: `PagedMemoryRepository.retrieve`
        - 作用 / Purpose: 在应用层统一发起外部记忆召回
        - 变量 / Variables:
          `query` 是 head-space query, `device` 是召回张量目标设备
        - 接入 / Integration: 返回值直接传给 MHDSRA2 retrieval 分支
        - 错误处理 / Error handling: 仓储会把禁用、空记忆和不支持批大小映射为空召回
        - 关键词 / Keywords:
          retrieve|unit_of_work|repository|query|device|external_memory|kv|application|recall|检索
        """
        return self.memory_repository.retrieve(query, device)

    def commit_forward(
        self,
        *,
        state: MHDSRA2State,
        kv_cache: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
        time_state: Optional[torch.Tensor],
    ) -> None:
        """Commit successful forward outputs into the unit of work.

        中文说明:
        - 调用方 / Called by: `DSRA_Chunk_Layer.forward`
        - 调用对象 / Calls: 无外部函数
        - 作用 / Purpose: 将前向成功后的流式状态、局部缓存和时间状态作为工作单元结果
        - 变量 / Variables:
          `state` 是新 MHDSRA2 状态, `kv_cache` 是旧接口兼容缓存,
          `time_state` 是 timestamps 兼容状态
        - 接入 / Integration: 调用后从 `uow.state/uow.kv_cache/uow.time_state` 读取返回值
        - 错误处理 / Error handling: 仅在成功前向后调用；异常前不会提交半成品
        - 关键词 / Keywords:
          commit|unit_of_work|state|kv_cache|time_state|forward|application|streaming|result|提交
        """
        self.state = state
        self.kv_cache = kv_cache
        self.time_state = time_state
