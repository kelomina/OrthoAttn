"""CPU-side paged exact memory for MHDSRA2.

This is a minimal reference implementation. It keeps all token-level K/V on CPU
and only returns a small retrieved subset to the GPU-side attention layer.
For production, replace the exact page scoring with FAISS/ScaNN or a custom ANN
index and store token K/V in pinned CPU memory, mmap, or NVMe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

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


class PagedExactMemory:
    """Append-only paged token memory with simple top-page retrieval.

    The module is intentionally CPU-side. It gives DSRA-v2 a precise recall path
    without keeping 2M-token KV on GPU.
    """

    def __init__(self, page_size: int = 1024, dtype: torch.dtype = torch.float16):
        self.page_size = int(page_size)
        self.dtype = dtype
        self.pages: List[PageRecord] = []
        self.next_position = 0

    def __len__(self) -> int:
        return len(self.pages)

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append chunk K/V.

        key/value shape: [1,H,T,d] or [H,T,d]. T may be larger than page_size.
        Only batch_size=1 is supported in this reference implementation.
        """
        if key.dim() == 4:
            if key.shape[0] != 1:
                raise ValueError("PagedExactMemory reference code supports batch_size=1")
            key = key[0]
            value = value[0]
        if key.dim() != 3:
            raise ValueError("key/value must be [H,T,d] or [1,H,T,d]")
        _, t, _ = key.shape
        for s in range(0, t, self.page_size):
            e = min(s + self.page_size, t)
            k_page = key[:, s:e, :].detach().to("cpu", dtype=self.dtype).contiguous()
            v_page = value[:, s:e, :].detach().to("cpu", dtype=self.dtype).contiguous()
            summary = F.normalize(k_page.float().mean(dim=1), dim=-1).to(dtype=self.dtype)
            self.pages.append(PageRecord(k_page, v_page, summary, self.next_position + s, self.next_position + e))
        self.next_position += t

    def invalidate_before(self, position: int) -> None:
        """Forget obsolete pages without rewriting memory."""
        for p in self.pages:
            if p.end <= position:
                p.valid = False
                p.version += 1

    def retrieve(
        self,
        query: torch.Tensor,
        top_pages: int = 4,
        max_tokens: int = 128,
        device: Optional[torch.device] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return retrieved_k, retrieved_v, positions.

        query shape: [1,H,T,d] or [H,T,d]. Returned K/V shape: [1,H,R,d].
        Retrieval is chunk-level: mean query selects pages, then mean query selects
        top tokens inside those pages. Per-token retrieval can be added on top of
        this interface but costs more memory.
        """
        if len(self.pages) == 0:
            return None, None, None
        if query.dim() == 4:
            if query.shape[0] != 1:
                raise ValueError("PagedExactMemory reference code supports batch_size=1")
            query_h = query[0]
        elif query.dim() == 3:
            query_h = query
        else:
            raise ValueError("query must be [H,T,d] or [1,H,T,d]")

        device = device or query.device
        q_summary = F.normalize(query_h.float().mean(dim=1), dim=-1).cpu()
        valid_pages = [p for p in self.pages if p.valid]
        if not valid_pages:
            return None, None, None
        summaries = torch.stack([p.summary.float() for p in valid_pages], dim=1)
        page_scores = torch.einsum("hd,hpd->hp", q_summary, summaries).mean(dim=0)
        n_pages = min(top_pages, page_scores.numel())
        top_page_idx = torch.topk(page_scores, n_pages).indices.tolist()
        chosen = [valid_pages[i] for i in top_page_idx]

        k_cat = torch.cat([p.key.float() for p in chosen], dim=1)
        v_cat = torch.cat([p.value.float() for p in chosen], dim=1)
        pos = torch.cat([torch.arange(p.start, p.end) for p in chosen], dim=0)
        token_scores = torch.einsum("hd,hrd->hr", q_summary, F.normalize(k_cat, dim=-1)).mean(dim=0)
        r = min(max_tokens, token_scores.numel())
        tok_idx = torch.topk(token_scores, r).indices
        k_out = k_cat[:, tok_idx, :].unsqueeze(0).to(device=device, dtype=query.dtype)
        v_out = v_cat[:, tok_idx, :].unsqueeze(0).to(device=device, dtype=query.dtype)
        pos_out = pos[tok_idx].to(device=device)
        return k_out, v_out, pos_out


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
