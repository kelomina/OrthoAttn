from __future__ import annotations

import torch

from scripts.diagnostic_memory_benchmark import MODEL_LABELS, MODEL_ORDER, _build_mhdsra2_layer
from src.dsra.mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2


def test_diagnostic_model_order_separates_forced_and_learned_gate() -> None:
    assert "mhdsra2_with_paged_recall_forced_gate" in MODEL_ORDER
    assert "mhdsra2_with_paged_recall_learned_gate" in MODEL_ORDER
    assert "forced gate" in MODEL_LABELS["mhdsra2_with_paged_recall_forced_gate"]
    assert "learned gate" in MODEL_LABELS["mhdsra2_with_paged_recall_learned_gate"]


def test_diagnostic_forced_gate_is_opt_in() -> None:
    forced = _build_mhdsra2_layer(
        dim=8,
        slots=2,
        use_retrieval=True,
        key_count=2,
        force_retrieval_gate=True,
    )
    learned = _build_mhdsra2_layer(
        dim=8,
        slots=2,
        use_retrieval=True,
        key_count=2,
        force_retrieval_gate=False,
    )

    forced_bias = forced.fuse_gate.bias.detach()
    assert float(forced_bias[2]) > float(forced_bias[1])
    assert learned.fuse_gate.weight.abs().sum().item() != 0.0


def test_gate_quality_bias_default_keeps_existing_gate_path() -> None:
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
    layer = MultiHeadDSRA2(cfg)
    x = torch.randn(1, 2, 8)
    q_proj, _, _ = layer.qkv(x).chunk(3, dim=-1)
    retrieved_k = layer._to_heads(q_proj[:, :1, :]).detach()
    retrieved_v = torch.randn_like(retrieved_k)

    _, _, aux = layer(
        x,
        retrieved_k=retrieved_k,
        retrieved_v=retrieved_v,
        return_aux=True,
    )

    assert aux["retrieval_available"] is True
    assert aux["retrieved_token_count"] == 1
    assert aux["slot_confidence_mean"].shape == ()
    assert aux["gate_slot_mean"].shape == ()
    assert aux["gate_local_mean"].shape == ()
    assert aux["gate_retrieval_mean"].shape == ()
    assert torch.isfinite(aux["gates_mean"]).all()
    assert float(aux["gate_retrieval_mean"]) > 0.0


def test_gate_quality_bias_increases_retrieval_gate_when_retrieval_exists() -> None:
    torch.manual_seed(0)
    base_cfg = MHDSRA2Config(
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
    biased_cfg = MHDSRA2Config(**{**base_cfg.__dict__, "retrieval_quality_gate_bias": 2.0})
    base = MultiHeadDSRA2(base_cfg)
    biased = MultiHeadDSRA2(biased_cfg)
    biased.load_state_dict(base.state_dict())
    x = torch.randn(1, 2, 8)
    q_proj, _, _ = base.qkv(x).chunk(3, dim=-1)
    retrieved_k = base._to_heads(q_proj[:, :1, :]).detach()
    retrieved_v = torch.randn_like(retrieved_k)

    _, _, base_aux = base(x, retrieved_k=retrieved_k, retrieved_v=retrieved_v, return_aux=True)
    _, _, biased_aux = biased(x, retrieved_k=retrieved_k, retrieved_v=retrieved_v, return_aux=True)

    assert biased_aux["gate_retrieval_mean"] > base_aux["gate_retrieval_mean"]


def test_gate_quality_bias_does_not_enable_missing_retrieval() -> None:
    cfg = MHDSRA2Config(
        dim=8,
        heads=2,
        slots=4,
        read_topk=1,
        write_topk=1,
        use_local=False,
        use_retrieval=True,
        retrieval_quality_gate_bias=10.0,
        detach_state=False,
    )
    layer = MultiHeadDSRA2(cfg)

    _, _, aux = layer(torch.randn(1, 2, 8), return_aux=True)

    assert aux["retrieval_available"] is False
    assert aux["retrieved_token_count"] == 0
    assert float(aux["gate_retrieval_mean"]) == 0.0


def test_retrieval_mask_disables_empty_batch_row() -> None:
    """Masked retrieval padding should not contribute to attention or gates.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `MultiHeadDSRA2._retrieval_attention`, `MultiHeadDSRA2.forward`。
    - 作用 / Purpose: 验证 batch 中无有效召回的样本不会被 padding token 污染。
    - 错误处理 / Error handling: attention 输出非零、gate 诊断错误或非 finite 都会触发断言。
    """
    torch.manual_seed(0)
    cfg = MHDSRA2Config(
        dim=8,
        heads=2,
        slots=4,
        read_topk=1,
        write_topk=1,
        use_local=False,
        use_retrieval=True,
        detach_state=False,
    )
    layer = MultiHeadDSRA2(cfg)
    x = torch.randn(2, 1, 8)
    q_proj, _, _ = layer.qkv(x).chunk(3, dim=-1)
    query_heads = layer._to_heads(q_proj)
    retrieved_k = torch.randn(2, 2, 1, 4)
    retrieved_v = torch.randn(2, 2, 1, 4)
    retrieved_mask = torch.tensor([[True], [False]])

    retrieval_out = layer._retrieval_attention(
        query_heads,
        retrieved_k,
        retrieved_v,
        retrieved_mask,
    )
    _, _, aux = layer(
        x,
        retrieved_k=retrieved_k,
        retrieved_v=retrieved_v,
        retrieved_mask=retrieved_mask,
        return_aux=True,
    )

    assert torch.isfinite(retrieval_out).all()
    assert float(retrieval_out[0].detach().abs().sum().item()) > 0.0
    assert float(retrieval_out[1].detach().abs().sum().item()) == 0.0
    assert aux["retrieval_available"] is True
    assert float(aux["retrieval_available_ratio"]) == 0.5
    assert float(aux["retrieved_token_count_mean"]) == 0.5
    assert float(aux["retrieved_token_count_max"]) == 1.0
    assert float(aux["gate_retrieval_mean"]) > 0.0
    assert aux["gate_retrieval_by_sample"].shape == (2,)
    assert float(aux["gate_retrieval_by_sample"][0]) > 0.0
    assert float(aux["gate_retrieval_by_sample"][1]) == 0.0


def test_retrieval_mask_is_finite_in_half_precision() -> None:
    """Half precision masking should not overflow while hiding invalid tokens.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `MultiHeadDSRA2._retrieval_attention`。
    - 作用 / Purpose: 防止 fp16 retrieval mask 使用过小常量导致 `masked_fill` 溢出。
    - 错误处理 / Error handling: RuntimeError、非 finite 或无效行非零都会触发断言。
    """
    cfg = MHDSRA2Config(
        dim=8,
        heads=2,
        slots=4,
        read_topk=1,
        write_topk=1,
        use_local=False,
        use_retrieval=True,
        detach_state=False,
    )
    layer = MultiHeadDSRA2(cfg).half()
    q = torch.randn(2, 2, 1, 4, dtype=torch.float16)
    retrieved_k = torch.randn(2, 2, 1, 4, dtype=torch.float16)
    retrieved_v = torch.randn(2, 2, 1, 4, dtype=torch.float16)
    retrieved_mask = torch.tensor([[True], [False]])

    retrieval_out = layer._retrieval_attention(q, retrieved_k, retrieved_v, retrieved_mask)

    assert retrieval_out.dtype == torch.float16
    assert torch.isfinite(retrieval_out).all()
    assert float(retrieval_out[1].detach().abs().sum().item()) == 0.0


def test_context_film_hidden_scales_and_allows_legacy_width() -> None:
    scaled = MultiHeadDSRA2(
        MHDSRA2Config(dim=64, heads=4, use_context_film=True)
    )
    legacy = MultiHeadDSRA2(
        MHDSRA2Config(dim=64, heads=4, use_context_film=True, context_film_hidden=8)
    )

    assert scaled.film_net[0].out_features == 16
    assert legacy.film_net[0].out_features == 8


def test_local_mask_cache_reuses_last_shape_without_changing_output() -> None:
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
    cached = MultiHeadDSRA2(cfg)
    reference = MultiHeadDSRA2(cfg)
    reference.load_state_dict(cached.state_dict())
    x = torch.randn(1, 2, 8)

    y_cached, _ = cached(x)
    first_mask = cached._local_mask_cache
    y_cached_again, _ = cached(x)
    second_mask = cached._local_mask_cache

    y_ref, _ = reference(x)
    y_ref_again, _ = reference(x)

    assert first_mask is not None
    assert second_mask is first_mask
    torch.testing.assert_close(y_cached, y_ref)
    torch.testing.assert_close(y_cached_again, y_ref_again)
