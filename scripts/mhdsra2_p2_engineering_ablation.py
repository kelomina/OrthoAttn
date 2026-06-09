"""Small isolated checks for the MHDSRA2 P2/P3 engineering fixes.

This script avoids training. Each check targets one independent guardrail so a
future regression can be diagnosed without running the full benchmark suite.
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
from src.dsra.mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2  # noqa: E402
from src.dsra.mhdsra2.paged_exact_memory import PagedExactMemory  # noqa: E402


def _check_max_pages_pruning() -> dict[str, object]:
    """Verify explicit max_pages bounds stored page records.

    中文说明:
    - 调用方 / Called by: `main`
    - 调用对象 / Calls: `PagedExactMemory.append`
    - 作用 / Purpose: 独立验证分页记忆上限只在显式配置后生效
    - 返回 / Returns: 包含 pass 状态、页面数和保留位置的字典
    - 错误处理 / Error handling: 不吞异常；调用方统一捕获并汇总
    """
    memory = PagedExactMemory(page_size=1, dtype=torch.float32, max_pages=2)
    memory.append(torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
    kept = [(page.start, page.end) for page in memory.pages]
    return {
        "passed": len(memory) == 2 and kept == [(2, 3), (3, 4)] and memory.next_position == 4,
        "page_count": len(memory),
        "kept_pages": kept,
        "next_position": memory.next_position,
    }


def _check_retrieval_gate_bias() -> dict[str, object]:
    """Verify opt-in retrieval quality bias raises retrieval gate mass."""
    torch.manual_seed(0)
    cfg = MHDSRA2Config(
        dim=8,
        heads=2,
        slots=4,
        read_topk=1,
        write_topk=1,
        use_local=False,
        use_retrieval=True,
        retrieval_quality_gate_bias=0.0,
        detach_state=False,
    )
    biased_cfg = MHDSRA2Config(**{**cfg.__dict__, "retrieval_quality_gate_bias": 2.0})
    base = MultiHeadDSRA2(cfg)
    biased = MultiHeadDSRA2(biased_cfg)
    biased.load_state_dict(base.state_dict())
    x = torch.randn(1, 2, 8)
    q_proj, _, _ = base.qkv(x).chunk(3, dim=-1)
    retrieved_k = base._to_heads(q_proj[:, :1, :]).detach()
    retrieved_v = torch.randn_like(retrieved_k)
    _, _, base_aux = base(x, retrieved_k=retrieved_k, retrieved_v=retrieved_v, return_aux=True)
    _, _, biased_aux = biased(x, retrieved_k=retrieved_k, retrieved_v=retrieved_v, return_aux=True)
    return {
        "passed": bool(biased_aux["gate_retrieval_mean"] > base_aux["gate_retrieval_mean"]),
        "base_gate_retrieval_mean": float(base_aux["gate_retrieval_mean"]),
        "biased_gate_retrieval_mean": float(biased_aux["gate_retrieval_mean"]),
        "retrieved_token_count": int(base_aux["retrieved_token_count"]),
    }


def _check_context_film_hidden() -> dict[str, object]:
    """Verify CCFM FiLM hidden width scales and keeps legacy override."""
    scaled = MultiHeadDSRA2(MHDSRA2Config(dim=64, heads=4, use_context_film=True))
    legacy = MultiHeadDSRA2(
        MHDSRA2Config(dim=64, heads=4, use_context_film=True, context_film_hidden=8)
    )
    return {
        "passed": scaled.film_net[0].out_features == 16 and legacy.film_net[0].out_features == 8,
        "scaled_hidden": scaled.film_net[0].out_features,
        "legacy_hidden": legacy.film_net[0].out_features,
    }


def _check_mask_cache_and_state_copy() -> dict[str, object]:
    """Verify local mask caching and non-mutating cache injection."""
    torch.manual_seed(0)
    cfg = MHDSRA2Config(
        dim=8,
        heads=2,
        slots=4,
        read_topk=1,
        write_topk=1,
        local_window=4,
        use_local=True,
        use_retrieval=False,
        detach_state=False,
    )
    layer = MultiHeadDSRA2(cfg)
    x = torch.randn(1, 2, 8)
    layer(x)
    first_mask = layer._local_mask_cache
    layer(x)
    reused_mask = first_mask is not None and layer._local_mask_cache is first_mask

    compat = DSRA_Chunk_Layer(dim=8, K=4, kr=1, use_bypass=True)
    state = compat._coerce_state(None, 1, torch.device("cpu"), torch.float32)
    cached_k = torch.randn(1, compat.core.heads, 1, compat.core.d_head)
    cached_v = torch.randn(1, compat.core.heads, 1, compat.core.d_head)
    compat.forward_step(torch.randn(1, 1, 8), state, (cached_k, cached_v))
    state_clean = state.local_k is None and state.local_v is None

    return {
        "passed": bool(reused_mask and state_clean),
        "mask_reused": bool(reused_mask),
        "input_state_local_cache_clean": bool(state_clean),
    }


def main() -> int:
    checks: dict[str, Callable[[], dict[str, object]]] = {
        "max_pages_pruning": _check_max_pages_pruning,
        "retrieval_gate_bias": _check_retrieval_gate_bias,
        "context_film_hidden": _check_context_film_hidden,
        "mask_cache_and_state_copy": _check_mask_cache_and_state_copy,
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
