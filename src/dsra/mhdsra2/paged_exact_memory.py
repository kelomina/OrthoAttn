"""CPU-side paged exact memory for MHDSRA2.

This is a minimal reference implementation. It keeps all token-level K/V on CPU
and only returns a small retrieved subset to the GPU-side attention layer.
For production, replace the exact page scoring with FAISS/ScaNN or a custom ANN
index and store token K/V in pinned CPU memory, mmap, or NVMe.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class PageRecord:
    key: torch.Tensor
    value: torch.Tensor
    summary: torch.Tensor
    start: int
    end: int
    valid: bool = True
    version: int = 0


@dataclass
class _PreparedRetrieveSample:
    q_summary: torch.Tensor
    valid_pages: list[PageRecord]
    safe_page_keys: list[torch.Tensor]
    safe_summaries: list[torch.Tensor]
    max_position: Optional[int]


class PagedExactMemory:
    """Batch-isolated paged token memory with simple top-page retrieval.

    The module is intentionally CPU-side. It gives DSRA-v2 a precise recall path
    without keeping 2M-token KV on GPU. For batch>1, each batch row owns an
    independent page list so one sample can never retrieve another sample's K/V.
    """

    _BATCH_CHANGED_MESSAGE = (
        "PagedExactMemory stream batch_size changed from {old} to {new}; "
        "call reset() before reusing memory with a different batch size."
    )
    _VECTOR_PAGE_SCORE_MAX_ELEMENTS = 8_000_000
    _QUERY_POOLING_MODES = {"mean", "max_token"}

    def __init__(
        self,
        page_size: int = 1024,
        dtype: torch.dtype = torch.float16,
        max_pages: Optional[int] = None,
        query_pooling: Literal["mean", "max_token"] = "mean",
    ):
        self.page_size = int(page_size)
        self.dtype = dtype
        self.max_pages = None if max_pages is None else int(max_pages)
        self.query_pooling = str(query_pooling)
        if self.page_size < 1:
            raise ValueError("page_size must be positive")
        if self.max_pages is not None and self.max_pages < 1:
            raise ValueError("max_pages must be positive or None")
        if self.query_pooling not in self._QUERY_POOLING_MODES:
            allowed = ", ".join(sorted(self._QUERY_POOLING_MODES))
            raise ValueError(f"query_pooling must be one of: {allowed}")
        self._batch_size: Optional[int] = None
        self._pages_by_sample: list[list[PageRecord]] = []
        self._next_positions: list[int] = []
        self.last_retrieve_profile: Optional[dict[str, object]] = None

    @property
    def batch_size(self) -> Optional[int]:
        """Return the locked stream batch size, or None before first use."""
        return self._batch_size

    @property
    def pages_by_sample(self) -> list[list[PageRecord]]:
        """Return per-sample page lists for batch-isolation diagnostics."""
        return self._pages_by_sample

    @property
    def next_positions(self) -> list[int]:
        """Return per-sample next global positions."""
        return self._next_positions

    @property
    def pages(self) -> list[PageRecord]:
        """Batch=1 compatibility view over sample 0 pages."""
        if not self._pages_by_sample:
            return []
        return self._pages_by_sample[0]

    @property
    def next_position(self) -> int:
        """Batch=1 compatibility view over sample 0 next position."""
        if not self._next_positions:
            return 0
        return self._next_positions[0]

    def __len__(self) -> int:
        return len(self.pages)

    def reset(self) -> None:
        """Clear all stored pages and restart global token positions.

        中文说明:
        - 调用方 / Called by: `PagedMemoryRepository.reset`, tests, long-lived stream owners.
        - 调用对象 / Calls: list clear operation only.
        - 作用 / Purpose: 在独立序列或评估样本之间清空 CPU 分页 K/V，防止跨样本记忆泄漏。
        - 参数 / Parameters: 无。
        - 返回 / Returns: None。
        - 错误处理 / Error handling: 无外部资源；原地清空不吞异常。
        - 副作用 / Side effects: 删除所有 page records，并解除 batch_size 锁。

        English documentation:
        Function name:
            reset
        Purpose:
            Clear external paged K/V memory between independent streams.
        """
        self._batch_size = None
        self._pages_by_sample.clear()
        self._next_positions.clear()
        self.last_retrieve_profile = None

    def clear(self) -> None:
        """Alias for `reset` for callers that use collection-style naming."""
        self.reset()

    def _ensure_batch_size(self, batch_size: int) -> None:
        """Lock or validate the stream batch size.

        中文说明:
        - 调用方 / Called by: `append`, `retrieve`
        - 调用对象 / Calls: list allocation only
        - 作用 / Purpose: 保证同一条外部分页记忆 stream 内 batch size 不变化；
          如果需要从 batch=4 切换到 batch=1，调用方必须先 `reset()`
        - 参数 / Parameters: `batch_size` 是当前输入的 batch 行数
        - 返回 / Returns: None
        - 错误处理 / Error handling: batch size 变化时抛出 ValueError，避免跨 stream 串记忆
        - 关键词 / Keywords:
          batch_size|stream|reset|paged_memory|isolation|external_memory|隔离
        """
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self._batch_size is None:
            self._batch_size = batch_size
            self._pages_by_sample = [[] for _ in range(batch_size)]
            self._next_positions = [0 for _ in range(batch_size)]
            return
        if self._batch_size != batch_size:
            raise ValueError(
                self._BATCH_CHANGED_MESSAGE.format(old=self._batch_size, new=batch_size)
            )

    def _normalize_batch_tensor(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.dim() == 3:
            return tensor.unsqueeze(0)
        if tensor.dim() == 4:
            return tensor
        raise ValueError(f"{name} must be [H,T,d] or [B,H,T,d]")

    def _position_for_sample(
        self,
        position: Optional[int | torch.Tensor | list[int] | tuple[int, ...]],
        sample_idx: int,
    ) -> Optional[int]:
        if position is None:
            return None
        if isinstance(position, torch.Tensor):
            flat = position.detach().cpu().flatten()
            if flat.numel() == 1:
                return int(flat.item())
            if self._batch_size is not None and flat.numel() != self._batch_size:
                raise ValueError(
                    f"expected max_position for {self._batch_size} samples, got {flat.numel()}"
                )
            return int(flat[sample_idx].item())
        if isinstance(position, (list, tuple)):
            if len(position) == 1:
                return int(position[0])
            if self._batch_size is not None and len(position) != self._batch_size:
                raise ValueError(
                    f"expected max_position for {self._batch_size} samples, got {len(position)}"
                )
            return int(position[sample_idx])
        return int(position)

    def _enforce_max_pages(self, sample_idx: int) -> None:
        """Prune old pages for one sample when an explicit page cap is configured."""
        if self.max_pages is None:
            return
        pages = self._pages_by_sample[sample_idx]
        if len(pages) <= self.max_pages:
            return

        overflow = len(pages) - self.max_pages
        invalid_pages = [page for page in pages if not page.valid]
        invalid_to_drop = set(id(page) for page in invalid_pages[:overflow])
        if invalid_to_drop:
            pages = [page for page in pages if id(page) not in invalid_to_drop]

        if len(pages) > self.max_pages:
            pages = pages[-self.max_pages :]
        self._pages_by_sample[sample_idx] = pages

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append chunk K/V for one or more independent batch samples.

        key/value shape: [B,H,T,d] or [H,T,d]. T may be larger than page_size.
        Each batch row is stored in its own page list. When max_pages is
        configured, old page records are pruned per sample after append while
        each sample's next_position keeps a monotonic global token coordinate.
        """
        key_b = self._normalize_batch_tensor(key, "key")
        value_b = self._normalize_batch_tensor(value, "value")
        if key_b.shape != value_b.shape:
            raise ValueError("key and value must have the same shape")
        batch_size, _, token_count, _ = key_b.shape
        self._ensure_batch_size(batch_size)

        for sample_idx in range(batch_size):
            sample_key = key_b[sample_idx]
            sample_value = value_b[sample_idx]
            next_position = self._next_positions[sample_idx]
            for start in range(0, token_count, self.page_size):
                end = min(start + self.page_size, token_count)
                k_page = sample_key[:, start:end, :].detach().to(
                    "cpu", dtype=self.dtype
                ).contiguous()
                v_page = sample_value[:, start:end, :].detach().to(
                    "cpu", dtype=self.dtype
                ).contiguous()
                summary = F.normalize(k_page.float().mean(dim=1), dim=-1).to(
                    dtype=self.dtype
                )
                self._pages_by_sample[sample_idx].append(
                    PageRecord(
                        k_page,
                        v_page,
                        summary,
                        next_position + start,
                        next_position + end,
                    )
                )
            self._next_positions[sample_idx] = next_position + token_count
            self._enforce_max_pages(sample_idx)

    def invalidate_before(self, position: int | torch.Tensor | list[int] | tuple[int, ...]) -> None:
        """Forget obsolete pages independently for each sample."""
        if self._batch_size is None:
            return
        for sample_idx, pages in enumerate(self._pages_by_sample):
            sample_position = self._position_for_sample(position, sample_idx)
            if sample_position is None:
                continue
            for page in pages:
                if page.end <= sample_position:
                    page.valid = False
                    page.version += 1

    @staticmethod
    def _rank_by_score_then_position(scores: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rank candidates by score first and recency second."""
        position_order = torch.argsort(positions, descending=True, stable=True)
        score_order = torch.argsort(scores[position_order], descending=True, stable=True)
        return position_order[score_order]

    def _prepare_retrieve_sample(
        self,
        query_h: torch.Tensor,
        pages: list[PageRecord],
        max_position: Optional[int],
    ) -> _PreparedRetrieveSample:
        """Prepare per-sample valid pages and safe page tensors for scoring."""
        query_float = query_h.float()
        if self.query_pooling == "max_token":
            query_norm = F.normalize(query_float, dim=-1)
            q_summary = F.normalize(query_norm, dim=-1).cpu()
        else:
            q_summary = F.normalize(query_float.mean(dim=1), dim=-1).cpu()
        valid_pages = [page for page in pages if page.valid]
        if max_position is not None:
            max_position = int(max_position)
            valid_pages = [page for page in valid_pages if page.start < max_position]
        if not valid_pages:
            return _PreparedRetrieveSample(q_summary, [], [], [], max_position)

        safe_page_keys: list[torch.Tensor] = []
        safe_summaries: list[torch.Tensor] = []
        if max_position is None:
            safe_page_keys = [page.key.float() for page in valid_pages]
            safe_summaries = [page.summary.float() for page in valid_pages]
        else:
            filtered_pages = []
            for page in valid_pages:
                safe_end = max(0, min(max_position, page.end) - page.start)
                if safe_end <= 0:
                    continue
                safe_key = page.key[:, :safe_end, :].float()
                safe_page_keys.append(safe_key)
                safe_summaries.append(F.normalize(safe_key.mean(dim=1), dim=-1))
                filtered_pages.append(page)
            if not safe_summaries:
                return _PreparedRetrieveSample(q_summary, [], [], [], max_position)
            valid_pages = filtered_pages
        return _PreparedRetrieveSample(
            q_summary,
            valid_pages,
            safe_page_keys,
            safe_summaries,
            max_position,
        )

    def _score_pages_one(self, prepared: _PreparedRetrieveSample) -> Optional[torch.Tensor]:
        """Score pages for one sample with the reference sample-loop implementation."""
        if not prepared.valid_pages:
            return None
        summaries = torch.stack(prepared.safe_summaries, dim=1)
        if prepared.q_summary.dim() == 3:
            page_mean_scores = torch.einsum(
                "htd,hpd->htp",
                prepared.q_summary,
                summaries,
            ).mean(dim=0).max(dim=0).values
        else:
            page_mean_scores = torch.einsum(
                "hd,hpd->hp",
                prepared.q_summary,
                summaries,
            ).mean(dim=0)
        per_page_max_scores = []
        for page_key in prepared.safe_page_keys:
            normalized_page_key = F.normalize(page_key, dim=-1)
            if prepared.q_summary.dim() == 3:
                token_scores = torch.einsum(
                    "hqd,htd->hqt",
                    prepared.q_summary,
                    normalized_page_key,
                ).mean(dim=0).max(dim=0).values
            else:
                token_scores = torch.einsum(
                    "hd,htd->ht",
                    prepared.q_summary,
                    normalized_page_key,
                ).mean(dim=0)
            per_page_max_scores.append(token_scores.max())
        page_token_scores = torch.stack(per_page_max_scores, dim=0)
        return torch.maximum(page_mean_scores, page_token_scores)

    def _score_pages_vectorized(
        self,
        prepared_samples: list[_PreparedRetrieveSample],
    ) -> Optional[list[Optional[torch.Tensor]]]:
        """Score page summaries and per-page max tokens across a batch.

        中文说明:
        - 调用方 / Called by: `retrieve`.
        - 调用对象 / Calls: `torch.stack`, `F.normalize`, `torch.einsum`.
        - 作用 / Purpose: 将 batch 内可变页数和页长打包成 padded tensor，一次性计算
          `[B,P]` 页分数；超过安全阈值时返回 None 让调用方走逐样本 fallback。
        - 返回 / Returns: 每个样本的 page score tensor；无有效页的样本为 None。
        - 错误处理 / Error handling: 临时张量规模超过阈值时不抛错，返回 None。
        """
        page_counts = [len(sample.valid_pages) for sample in prepared_samples]
        if not any(page_counts):
            return [None for _ in prepared_samples]
        max_pages = max(page_counts)
        max_page_tokens = max(
            (
                int(page_key.shape[1])
                for sample in prepared_samples
                for page_key in sample.safe_page_keys
            ),
            default=0,
        )
        if max_page_tokens <= 0:
            return [None for _ in prepared_samples]

        batch_size = len(prepared_samples)
        if self.query_pooling == "max_token":
            return None
        heads, d_head = prepared_samples[0].q_summary.shape
        element_count = batch_size * max_pages * heads * max_page_tokens * d_head
        if element_count > int(self._VECTOR_PAGE_SCORE_MAX_ELEMENTS):
            return None

        q_summary = torch.stack([sample.q_summary for sample in prepared_samples], dim=0)
        summaries = torch.zeros(
            batch_size,
            max_pages,
            heads,
            d_head,
            dtype=torch.float32,
        )
        page_keys = torch.zeros(
            batch_size,
            max_pages,
            heads,
            max_page_tokens,
            d_head,
            dtype=torch.float32,
        )
        page_mask = torch.zeros(batch_size, max_pages, dtype=torch.bool)
        token_mask = torch.zeros(batch_size, max_pages, max_page_tokens, dtype=torch.bool)

        for sample_idx, sample in enumerate(prepared_samples):
            for page_idx, (summary, page_key) in enumerate(
                zip(sample.safe_summaries, sample.safe_page_keys)
            ):
                token_count = int(page_key.shape[1])
                summaries[sample_idx, page_idx] = summary
                page_keys[sample_idx, page_idx, :, :token_count, :] = page_key
                page_mask[sample_idx, page_idx] = True
                token_mask[sample_idx, page_idx, :token_count] = True

        page_mean_scores = (q_summary[:, None, :, :] * summaries).sum(dim=-1).mean(dim=-1)
        token_scores = torch.einsum(
            "bhd,bphtd->bpht",
            q_summary,
            F.normalize(page_keys, dim=-1),
        ).mean(dim=2)
        token_scores = token_scores.masked_fill(
            ~token_mask,
            torch.finfo(token_scores.dtype).min,
        )
        page_token_scores = token_scores.max(dim=-1).values
        page_scores = torch.maximum(page_mean_scores, page_token_scores)
        page_scores = page_scores.masked_fill(~page_mask, torch.finfo(page_scores.dtype).min)
        return [
            None if page_count == 0 else page_scores[sample_idx, :page_count].clone()
            for sample_idx, page_count in enumerate(page_counts)
        ]

    def _retrieve_from_prepared(
        self,
        prepared: _PreparedRetrieveSample,
        page_scores: Optional[torch.Tensor],
        *,
        top_pages: int,
        max_tokens: int,
        query_dtype: torch.dtype,
        device: torch.device,
        profile_times: Optional[dict[str, float]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Retrieve K/V for one sample after page scores are available."""
        if page_scores is None or not prepared.valid_pages:
            return None, None, None

        page_select_start = time.perf_counter()
        selected_page_count = min(top_pages, page_scores.numel())
        page_positions = torch.tensor(
            [page.end for page in prepared.valid_pages],
            dtype=torch.long,
        )
        top_page_idx = self._rank_by_score_then_position(
            page_scores, page_positions
        )[:selected_page_count]
        chosen = [prepared.valid_pages[int(index)] for index in top_page_idx.tolist()]
        if profile_times is not None:
            profile_times["page_select_ms"] += (time.perf_counter() - page_select_start) * 1000.0

        k_cat = torch.cat([page.key.float() for page in chosen], dim=1)
        v_cat = torch.cat([page.value.float() for page in chosen], dim=1)
        pos = torch.cat([torch.arange(page.start, page.end) for page in chosen], dim=0)
        if prepared.max_position is not None:
            allowed = pos < prepared.max_position
            if not bool(allowed.any().item()):
                return None, None, None
            k_cat = k_cat[:, allowed, :]
            v_cat = v_cat[:, allowed, :]
            pos = pos[allowed]

        token_select_start = time.perf_counter()
        normalized_k_cat = F.normalize(k_cat, dim=-1)
        if prepared.q_summary.dim() == 3:
            token_scores = torch.einsum(
                "hqd,hrd->hqr",
                prepared.q_summary,
                normalized_k_cat,
            ).mean(dim=0).max(dim=0).values
        else:
            token_scores = torch.einsum(
                "hd,hrd->hr",
                prepared.q_summary,
                normalized_k_cat,
            ).mean(dim=0)
        selected_token_count = min(max_tokens, token_scores.numel())
        tok_idx = self._rank_by_score_then_position(
            token_scores, pos
        )[:selected_token_count]
        if profile_times is not None:
            profile_times["token_select_ms"] += (time.perf_counter() - token_select_start) * 1000.0

        materialize_start = time.perf_counter()
        k_out = k_cat[:, tok_idx, :].to(device=device, dtype=query_dtype)
        v_out = v_cat[:, tok_idx, :].to(device=device, dtype=query_dtype)
        pos_out = pos[tok_idx].to(device=device)
        if profile_times is not None:
            profile_times["materialize_ms"] += (time.perf_counter() - materialize_start) * 1000.0
        return k_out, v_out, pos_out

    def retrieve(
        self,
        query: torch.Tensor,
        top_pages: int = 4,
        max_tokens: int = 128,
        device: Optional[torch.device] = None,
        max_position: Optional[int | torch.Tensor | list[int] | tuple[int, ...]] = None,
        *,
        return_mask: bool = False,
        return_metadata: bool = False,
        profile: bool = False,
    ):
        """Return retrieved K/V, positions, and optionally a validity mask/metadata.

        query shape: [B,H,T,d] or [H,T,d]. For batch=1, returned K/V keep the
        historical [1,H,R,d] shape and positions are [R]. For batch>1, returned
        K/V are [B,H,R,d], positions are [B,R], and `return_mask=True` returns a
        [B,R] mask so padded retrieval slots cannot affect attention. Metadata
        is opt-in and duplicates position/mask/count data without changing the
        historical tuple when disabled.
        """
        total_start = time.perf_counter()
        self.last_retrieve_profile = None
        profile_times: Optional[dict[str, float]] = None
        if profile:
            profile_times = {
                "page_filter_pack_ms": 0.0,
                "page_score_ms": 0.0,
                "page_select_ms": 0.0,
                "token_select_ms": 0.0,
                "materialize_ms": 0.0,
            }

        query_b = self._normalize_batch_tensor(query, "query")
        batch_size, heads, _, d_head = query_b.shape
        device = device or query.device
        self._ensure_batch_size(batch_size)

        prepare_start = time.perf_counter()
        prepared_samples: list[_PreparedRetrieveSample] = []
        for sample_idx in range(batch_size):
            sample_max_position = self._position_for_sample(max_position, sample_idx)
            prepared_samples.append(
                self._prepare_retrieve_sample(
                    query_b[sample_idx],
                    self._pages_by_sample[sample_idx],
                    sample_max_position,
                )
            )
        if profile_times is not None:
            profile_times["page_filter_pack_ms"] += (time.perf_counter() - prepare_start) * 1000.0

        score_start = time.perf_counter()
        page_scores_by_sample = self._score_pages_vectorized(prepared_samples)
        page_score_mode = "vectorized"
        if page_scores_by_sample is None:
            page_score_mode = "sample_loop"
            page_scores_by_sample = [
                self._score_pages_one(prepared)
                for prepared in prepared_samples
            ]
        if profile_times is not None:
            profile_times["page_score_ms"] += (time.perf_counter() - score_start) * 1000.0

        results: list[
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
        ] = [
            self._retrieve_from_prepared(
                prepared,
                page_scores,
                top_pages=top_pages,
                max_tokens=max_tokens,
                query_dtype=query.dtype,
                device=device,
                profile_times=profile_times,
            )
            for prepared, page_scores in zip(prepared_samples, page_scores_by_sample)
        ]

        valid_counts = [0 if row[0] is None else int(row[0].shape[1]) for row in results]
        max_retrieved = max(valid_counts, default=0)
        if max_retrieved <= 0:
            if profile:
                self.last_retrieve_profile = {
                    **(profile_times or {}),
                    "total_ms": (time.perf_counter() - total_start) * 1000.0,
                    "page_score_mode": page_score_mode,
                    "batch_size": batch_size,
                    "valid_page_counts": [len(sample.valid_pages) for sample in prepared_samples],
                    "retrieved_token_counts": valid_counts,
                    "max_retrieved": 0,
                    "top_pages": int(top_pages),
                    "max_tokens": int(max_tokens),
                }
            if return_metadata:
                empty_shape = (0,) if batch_size == 1 else (batch_size, 0)
                empty_positions = torch.empty(empty_shape, device=device, dtype=torch.long)
                metadata = {
                    "positions": empty_positions,
                    "mask": torch.empty(empty_shape, device=device, dtype=torch.bool),
                    "retrieved_token_counts": torch.zeros(
                        batch_size,
                        device=device,
                        dtype=torch.long,
                    ),
                    "max_position": max_position,
                }
                if return_mask:
                    return None, None, None, None, metadata
                return None, None, None, metadata
            if return_mask:
                return None, None, None, None
            return None, None, None

        batch_materialize_start = time.perf_counter()
        k_batch = torch.zeros(
            batch_size,
            heads,
            max_retrieved,
            d_head,
            device=device,
            dtype=query.dtype,
        )
        v_batch = torch.zeros_like(k_batch)
        pos_batch = torch.full(
            (batch_size, max_retrieved),
            -1,
            device=device,
            dtype=torch.long,
        )
        mask_batch = torch.zeros(
            batch_size,
            max_retrieved,
            device=device,
            dtype=torch.bool,
        )

        for sample_idx, (k_out, v_out, pos_out) in enumerate(results):
            if k_out is None or v_out is None or pos_out is None:
                continue
            token_count = k_out.shape[1]
            k_batch[sample_idx, :, :token_count, :] = k_out
            v_batch[sample_idx, :, :token_count, :] = v_out
            pos_batch[sample_idx, :token_count] = pos_out
            mask_batch[sample_idx, :token_count] = True

        if profile_times is not None:
            profile_times["materialize_ms"] += (
                time.perf_counter() - batch_materialize_start
            ) * 1000.0
            self.last_retrieve_profile = {
                **profile_times,
                "total_ms": (time.perf_counter() - total_start) * 1000.0,
                "page_score_mode": page_score_mode,
                "batch_size": batch_size,
                "valid_page_counts": [len(sample.valid_pages) for sample in prepared_samples],
                "safe_token_counts": [
                    int(sum(page_key.shape[1] for page_key in sample.safe_page_keys))
                    for sample in prepared_samples
                ],
                "retrieved_token_counts": valid_counts,
                "max_retrieved": int(max_retrieved),
                "top_pages": int(top_pages),
                "max_tokens": int(max_tokens),
            }

        if batch_size == 1:
            token_count = valid_counts[0]
            k_single = k_batch[:, :, :token_count, :]
            v_single = v_batch[:, :, :token_count, :]
            pos_single = pos_batch[0, :token_count]
            mask_single = mask_batch[0, :token_count]
            if return_metadata:
                metadata = {
                    "positions": pos_single,
                    "mask": mask_single,
                    "retrieved_token_counts": torch.tensor(
                        [token_count],
                        device=device,
                        dtype=torch.long,
                    ),
                    "max_position": max_position,
                }
                if return_mask:
                    return k_single, v_single, pos_single, mask_single, metadata
                return k_single, v_single, pos_single, metadata
            if return_mask:
                return k_single, v_single, pos_single, mask_single
            return k_single, v_single, pos_single

        if return_metadata:
            metadata = {
                "positions": pos_batch,
                "mask": mask_batch,
                "retrieved_token_counts": torch.tensor(
                    valid_counts,
                    device=device,
                    dtype=torch.long,
                ),
                "max_position": max_position,
            }
            if return_mask:
                return k_batch, v_batch, pos_batch, mask_batch, metadata
            return k_batch, v_batch, pos_batch, metadata
        if return_mask:
            return k_batch, v_batch, pos_batch, mask_batch
        return k_batch, v_batch, pos_batch


if __name__ == "__main__":
    torch.manual_seed(0)
    mem = PagedExactMemory(page_size=16)
    k = torch.randn(1, 4, 64, 32)
    v = torch.randn(1, 4, 64, 32)
    mem.append(k, v)
    q = k[:, :, 30:34, :] + 0.01 * torch.randn(1, 4, 4, 32)
    rk, rv, pos = mem.retrieve(q, top_pages=2, max_tokens=8)
    assert rk is not None and rv is not None and pos is not None
    print("[OK] PagedExactMemory smoke test passed")
    print("retrieved positions:", pos.tolist())
