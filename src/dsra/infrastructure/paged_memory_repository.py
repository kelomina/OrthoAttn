"""Infrastructure repository around CPU-side paged exact memory."""

from __future__ import annotations

from typing import Optional, Tuple

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
    - 错误处理 / Error handling: 禁用或批大小不受支持时跳过基础设施调用并返回空召回
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
    ) -> None:
        """Create a paged memory repository.

        中文说明:
        - 调用方 / Called by: `DSRA_Chunk_Layer.__init__`
        - 调用对象 / Calls: `PagedExactMemory`
        - 作用 / Purpose: 初始化基础设施层分页记忆与召回参数
        - 变量 / Variables:
          `enabled` 决定是否创建有效召回路径, `page_size/dtype` 传给底层分页记忆,
          `top_pages/max_tokens` 是检索预算
        - 接入 / Integration: 作为 `DSRA_Chunk_Layer.memory_repository` 持有
        - 错误处理 / Error handling: 底层 `PagedExactMemory` 对非法页大小沿用自身异常
        - 关键词 / Keywords:
          init|repository|paged|memory|enabled|page_size|dtype|top_pages|max_tokens|初始化
        """
        self.enabled = enabled
        self.memory = PagedExactMemory(page_size=page_size, dtype=dtype)
        self.top_pages = int(top_pages)
        self.max_tokens = int(max_tokens)

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append one chunk of head-space K/V into external memory.

        中文说明:
        - 调用方 / Called by: `DSRA_Chunk_Layer.forward`
        - 调用对象 / Calls: `PagedExactMemory.append`
        - 作用 / Purpose: 把当前 chunk 的 K/V 写入 CPU 分页记忆，供后续 chunk 精确召回
        - 变量 / Variables:
          `key/value` 是 `[B,H,T,d]` 或 `[H,T,d]` head-space 张量
        - 接入 / Integration: 前向完成后调用，避免当前 token 召回自身未来信息
        - 错误处理 / Error handling: 禁用时直接返回；非 batch=1 的参考实现暂时跳过
        - 关键词 / Keywords:
          append|kv|paged_memory|external|cpu|chunk|repository|write|mhdsra2|写入
        """
        if not self.enabled:
            return
        if key.dim() == 4 and key.shape[0] != 1:
            return
        self.memory.append(key, value)

    def retrieve(
        self, query: torch.Tensor, device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Retrieve exact K/V candidates for the current query.

        中文说明:
        - 调用方 / Called by: `StreamingAttentionUnitOfWork.retrieve`
        - 调用对象 / Calls: `PagedExactMemory.retrieve`
        - 作用 / Purpose: 从基础设施分页记忆中召回最相关 K/V，并隐藏位置元数据
        - 变量 / Variables:
          `query` 是当前 head-space query, `device` 是召回张量应返回到的设备
        - 接入 / Integration: 将返回值传给 `MultiHeadDSRA2.forward(retrieved_k=..., retrieved_v=...)`
        - 错误处理 / Error handling: 禁用、空记忆或不支持批大小时返回 `(None, None)`
        - 关键词 / Keywords:
          retrieve|query|kv|paged_memory|external|repository|device|recall|mhdsra2|召回
        """
        if not self.enabled:
            return None, None
        if query.dim() == 4 and query.shape[0] != 1:
            return None, None
        retrieved_k, retrieved_v, _ = self.memory.retrieve(
            query,
            top_pages=self.top_pages,
            max_tokens=self.max_tokens,
            device=device,
        )
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
        self.memory = PagedExactMemory(page_size=self.memory.page_size, dtype=self.memory.dtype)
