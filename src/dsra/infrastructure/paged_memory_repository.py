"""Infrastructure repository around CPU-side paged exact memory."""

from __future__ import annotations

from typing import Optional

import torch

from ..mhdsra2.paged_exact_memory import PagedExactMemory


class PagedMemoryRepository:
    """Own paged K/V memory behind a small repository boundary.

    中文说明:
    - 调用方 / Called by: `StreamingAttentionUnitOfWork`, `DSRA_Chunk_Layer`
    - 调用对象 / Calls: `PagedExactMemory.append`, `PagedExactMemory.retrieve`
    - 作用 / Purpose: 将外部精确记忆归入基础设施层，避免模型层直接管理分页存储细节
    - 变量 / Variables:
      `memory` 是底层分页记忆, `enabled` 控制是否启用检索, `top_pages/max_tokens`
      控制每次召回规模
    - 接入 / Integration: 模型层传入 head-space K/V 后调用 `append` 与 `retrieve`
    - 错误处理 / Error handling: 禁用或空记忆时返回空召回；同一 stream batch size
      变化时由底层显式抛出，避免串样本
    - 关键词 / Keywords:
      repository|paged_memory|infrastructure|retrieve|append|kv|mhdsra2|exact|memory|基础设施
    """

    def __init__(
        self,
        *,
        enabled: bool,
        page_size: int = 1024,
        dtype: torch.dtype = torch.float16,
        top_pages: int = 4,
        max_tokens: int = 128,
        max_pages: Optional[int] = None,
        query_pooling: str = "mean",
    ) -> None:
        """Create a paged memory repository.

        中文说明:
        - 调用方 / Called by: `DSRA_Chunk_Layer.__init__`
        - 调用对象 / Calls: `PagedExactMemory`
        - 作用 / Purpose: 初始化基础设施层分页记忆与召回参数
        - 变量 / Variables:
          `enabled` 决定是否创建有效召回路径, `page_size/dtype/max_pages` 传给底层分页记忆,
          `top_pages/max_tokens` 是检索预算
        - 接入 / Integration: 作为 `DSRA_Chunk_Layer.memory_repository` 持有
        - 错误处理 / Error handling: 底层 `PagedExactMemory` 对非法页大小沿用自身异常
        - 关键词 / Keywords:
          init|repository|paged|memory|enabled|page_size|dtype|top_pages|max_tokens|初始化
        """
        self.enabled = enabled
        self.memory = PagedExactMemory(
            page_size=page_size,
            dtype=dtype,
            max_pages=max_pages,
            query_pooling=query_pooling,
        )
        self.top_pages = int(top_pages)
        self.max_tokens = int(max_tokens)
        self.query_pooling = str(query_pooling)

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append one chunk of head-space K/V into external memory.

        中文说明:
        - 调用方 / Called by: `DSRA_Chunk_Layer.forward`
        - 调用对象 / Calls: `PagedExactMemory.append`
        - 作用 / Purpose: 把当前 chunk 的 K/V 写入 CPU 分页记忆，供后续 chunk 精确召回
        - 变量 / Variables:
          `key/value` 是 `[B,H,T,d]` 或 `[H,T,d]` head-space 张量
        - 接入 / Integration: 前向完成后调用，避免当前 token 召回自身未来信息
        - 错误处理 / Error handling: 禁用时直接返回；batch size 在同一 stream 内变化时由底层抛出 ValueError
        - 关键词 / Keywords:
          append|kv|paged_memory|external|cpu|chunk|repository|write|mhdsra2|写入
        """
        if not self.enabled:
            return
        self.memory.append(key, value)

    def retrieve(
        self,
        query: torch.Tensor,
        device: torch.device,
        max_position: Optional[int | torch.Tensor | list[int] | tuple[int, ...]] = None,
        *,
        return_mask: bool = False,
        return_metadata: bool = False,
        profile: bool = False,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ] | tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Retrieve exact K/V candidates for the current query.

        中文说明:
        - 调用方 / Called by: `StreamingAttentionUnitOfWork.retrieve`
        - 调用对象 / Calls: `PagedExactMemory.retrieve`
        - 作用 / Purpose: 从基础设施分页记忆中召回最相关 K/V，并隐藏位置元数据
        - 变量 / Variables:
          `query` 是当前 head-space query, `device` 是召回张量应返回到的设备
        - 接入 / Integration: 将返回值传给 `MultiHeadDSRA2.forward(retrieved_k=..., retrieved_v=...)`
        - 错误处理 / Error handling: 禁用或空记忆返回空召回；batch size 变化由底层显式抛出
        - 关键词 / Keywords:
          retrieve|query|kv|paged_memory|external|repository|device|recall|mhdsra2|召回
        """
        if not self.enabled:
            if return_metadata:
                if return_mask:
                    return None, None, None, None
                return None, None, None
            if return_mask:
                return None, None, None
            return None, None
        result = self.memory.retrieve(
            query,
            top_pages=self.top_pages,
            max_tokens=self.max_tokens,
            device=device,
            max_position=max_position,
            return_mask=True,
            return_metadata=return_metadata,
            profile=profile,
        )
        if return_metadata:
            retrieved_k, retrieved_v, positions, retrieved_mask, metadata = result
        else:
            retrieved_k, retrieved_v, positions, retrieved_mask = result
            metadata = None
        if return_mask:
            if return_metadata:
                return retrieved_k, retrieved_v, retrieved_mask, metadata
            return retrieved_k, retrieved_v, retrieved_mask
        if (
            retrieved_k is not None
            and retrieved_mask is not None
            and retrieved_mask.dim() == 2
            and retrieved_mask.shape[0] > 1
            and not bool(retrieved_mask.all().item())
        ):
            raise ValueError(
                "batch>1 retrieval returned padded rows; call retrieve(..., return_mask=True) "
                "and pass the mask to MultiHeadDSRA2.forward"
            )
        if return_metadata:
            return retrieved_k, retrieved_v, metadata
        return retrieved_k, retrieved_v

    def reset(self) -> None:
        """Clear all paged memory records.

        中文说明:
        - 调用方 / Called by: tests or long-lived model owners between independent streams
        - 调用对象 / Calls: `PagedExactMemory`
        - 作用 / Purpose: 为新的独立序列清理基础设施记忆，避免跨样本泄漏
        - 变量 / Variables: 无输入变量，仅重建 `memory`
        - 接入 / Integration: 长生命周期服务在请求结束后调用
        - 错误处理 / Error handling: 无外部资源释放，重建对象不会抛出额外异常
        - 关键词 / Keywords:
          reset|clear|memory|repository|paged|stream|request|lifecycle|cleanup|清理
        """
        self.memory.reset()
