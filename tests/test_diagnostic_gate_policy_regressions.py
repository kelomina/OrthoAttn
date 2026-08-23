from __future__ import annotations

import pytest
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


def test_retrieval_quality_features_single_candidate_margin_is_finite() -> None:
    """A single valid candidate should not produce sentinel-sized score margins."""
    torch.manual_seed(123)
    cfg = MHDSRA2Config(
        dim=8,
        heads=2,
        slots=4,
        read_topk=1,
        write_topk=1,
        use_local=False,
        use_retrieval=True,
        retrieval_quality_gate_adapter=True,
        detach_state=False,
    )
    layer = MultiHeadDSRA2(cfg)
    x = torch.randn(1, 2, 8)
    q_proj, _, _ = layer.qkv(x).chunk(3, dim=-1)
    q = layer._to_heads(q_proj)
    retrieved_k = torch.randn(1, 2, 1, 4)
    retrieved_mask = torch.tensor([[True]])

    features = layer._retrieval_quality_features(q, retrieved_k, retrieved_mask)

    assert features is not None
    assert torch.isfinite(features).all()
    assert float(features[0, 2].item()) == 0.0


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


def test_retrieval_attention_topk_default_preserves_weights() -> None:
    """Default-off retrieval top-k should preserve historical softmax behavior.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `MultiHeadDSRA2._retrieval_attention`。
    - 作用 / Purpose: 确认 `retrieval_attention_topk=None` 不改变旧路径，避免
      默认实验结果因为新配置项漂移。
    - 错误处理 / Error handling: 权重或输出不一致会触发断言。
    """
    torch.manual_seed(7)
    cfg = MHDSRA2Config(
        dim=8,
        heads=1,
        slots=4,
        use_local=False,
        use_retrieval=True,
        retrieval_tau=8.0,
        retrieval_attention_topk=None,
    )
    layer = MultiHeadDSRA2(cfg)
    q = torch.randn(1, 1, 2, 8)
    retrieved_k = torch.randn(1, 1, 6, 8)
    retrieved_v = torch.randn(1, 1, 6, 8)
    retrieved_mask = torch.tensor([[True, True, True, True, False, True]])

    output, weights = layer._retrieval_attention(
        q,
        retrieved_k,
        retrieved_v,
        retrieved_mask,
        return_weights=True,
    )

    logits = torch.einsum("bhtd,bhrd->bhtr", q, retrieved_k) * (8 ** -0.5)
    valid_view = retrieved_mask.view(1, 1, 1, 6)
    manual_logits = (logits * cfg.retrieval_tau).masked_fill(
        ~valid_view,
        torch.finfo(logits.dtype).min,
    )
    manual_weights = torch.softmax(manual_logits, dim=-1) * valid_view.to(dtype=q.dtype)
    manual_weights = manual_weights / manual_weights.sum(dim=-1, keepdim=True).clamp_min(
        cfg.eps
    )
    manual_output = torch.einsum("bhtr,bhrd->bhtd", manual_weights, retrieved_v)

    torch.testing.assert_close(weights, manual_weights)
    torch.testing.assert_close(output, manual_output)


def test_retrieval_attention_topk_concentrates_weight_without_dropping_recall() -> None:
    """Top-k retrieval attention should reduce softmax dilution after broad recall.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `MultiHeadDSRA2._retrieval_attention`。
    - 作用 / Purpose: 模拟 1 个 exact match + 127 个干扰候选；保留 128 个
      retrieved token，但 softmax 只在最高分 top-k 上归一化，验证关键 token
      权重明显上升。
    - 错误处理 / Error handling: top-k 未提高 exact-match 权重、无效候选有权重
      或非 top-k 候选仍参与归一化都会触发断言。
    """
    base_cfg = MHDSRA2Config(
        dim=8,
        heads=1,
        slots=4,
        use_local=False,
        use_retrieval=True,
        retrieval_tau=8.0,
        retrieval_attention_topk=None,
    )
    topk_cfg = MHDSRA2Config(
        **{**base_cfg.__dict__, "retrieval_attention_topk": 16}
    )
    base = MultiHeadDSRA2(base_cfg)
    topk = MultiHeadDSRA2(topk_cfg)
    q = torch.zeros(1, 1, 1, 8)
    q[..., 0] = 1.0
    retrieved_k = torch.zeros(1, 1, 128, 8)
    retrieved_k[..., 0, 0] = 1.0
    retrieved_k[..., 1:, 1] = 1.0
    retrieved_v = torch.randn(1, 1, 128, 8)
    retrieved_mask = torch.ones(1, 128, dtype=torch.bool)

    _, base_weights = base._retrieval_attention(
        q,
        retrieved_k,
        retrieved_v,
        retrieved_mask,
        return_weights=True,
    )
    _, topk_weights = topk._retrieval_attention(
        q,
        retrieved_k,
        retrieved_v,
        retrieved_mask,
        return_weights=True,
    )

    assert float(base_weights[0, 0, 0, 0]) < 0.2
    assert float(topk_weights[0, 0, 0, 0]) > 0.5
    assert int((topk_weights[0, 0, 0] > 0).sum().item()) == 16
    torch.testing.assert_close(topk_weights[0, 0, 0].sum(), torch.tensor(1.0))


def test_retrieval_attention_topk_respects_mask_and_half_precision() -> None:
    """Top-k retrieval attention should stay finite with padding and fp16.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `MultiHeadDSRA2._retrieval_attention`。
    - 作用 / Purpose: 保护 top-k mask 与 batch padding mask 的组合，尤其是
      fp16 下不能恢复 `-1e9` 一类溢出风险。
    - 错误处理 / Error handling: invalid row 非零、无效候选有权重、非 finite
      或 dtype 漂移都会触发断言。
    """
    cfg = MHDSRA2Config(
        dim=8,
        heads=2,
        slots=4,
        use_local=False,
        use_retrieval=True,
        retrieval_attention_topk=2,
        detach_state=False,
    )
    layer = MultiHeadDSRA2(cfg).half()
    q = torch.randn(2, 2, 1, 4, dtype=torch.float16)
    retrieved_k = torch.randn(2, 2, 4, 4, dtype=torch.float16)
    retrieved_v = torch.randn(2, 2, 4, 4, dtype=torch.float16)
    retrieved_mask = torch.tensor(
        [
            [True, False, True, True],
            [False, False, False, False],
        ]
    )

    output, weights = layer._retrieval_attention(
        q,
        retrieved_k,
        retrieved_v,
        retrieved_mask,
        return_weights=True,
    )

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    assert float(output[1].detach().abs().sum().item()) == 0.0
    assert float(weights[0, :, :, 1].detach().abs().sum().item()) == 0.0
    assert int((weights[0, 0, 0] > 0).sum().item()) <= 2
    assert int((weights[1] > 0).sum().item()) == 0


def test_retrieval_attention_topk_does_not_truncate_returned_scores() -> None:
    """Returned retrieval scores should remain broad diagnostics, not top-k logits.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `MultiHeadDSRA2._retrieval_attention`。
    - 作用 / Purpose: `retrieval_attention_topk` 只应影响 softmax 权重，不能把
      `return_scores=True` 暴露给训练/诊断的有效候选分数改成 top-k 后的
      sentinel，否则 evidence loss 或报告会把未入选候选误判成无效。
    - 错误处理 / Error handling: 有效但未入选 top-k 的候选分数若变成 dtype
      最小值会触发断言。
    """
    cfg = MHDSRA2Config(
        dim=8,
        heads=1,
        slots=4,
        use_local=False,
        use_retrieval=True,
        retrieval_tau=8.0,
        retrieval_attention_topk=2,
    )
    layer = MultiHeadDSRA2(cfg)
    q = torch.zeros(1, 1, 1, 8)
    q[..., 0] = 1.0
    retrieved_k = torch.zeros(1, 1, 6, 8)
    retrieved_k[..., 0, 0] = 1.0
    retrieved_k[..., 1:, 1] = 1.0
    retrieved_v = torch.randn(1, 1, 6, 8)

    _, weights, scores = layer._retrieval_attention(
        q,
        retrieved_k,
        retrieved_v,
        torch.ones(1, 6, dtype=torch.bool),
        return_weights=True,
        return_scores=True,
    )

    assert int((weights[0, 0, 0] > 0).sum().item()) == 2
    assert torch.isfinite(scores).all()
    assert float(scores[0, 0, 0, 2].item()) > torch.finfo(scores.dtype).min / 2


def test_retrieval_attention_topk_handles_token_specific_candidates() -> None:
    """Rank-5 retrieval candidates should apply top-k per query token.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `MultiHeadDSRA2._retrieval_attention`。
    - 作用 / Purpose: 覆盖 `retrieved_k` 形状为 `[B,H,T,R,d]` 的 token-specific
      retrieval 路径，确认每个 query token 都按自己的候选分数做 top-k，不会
      由于 `[B,1,T,R]` mask 广播而误用其它 token 的候选集合。
    - 错误处理 / Error handling: 每个 token 的 exact-match 权重不集中、padding
      token 有权重、或 top-k 数量不符合预期都会触发断言。
    """
    cfg = MHDSRA2Config(
        dim=8,
        heads=1,
        slots=4,
        use_local=False,
        use_retrieval=True,
        retrieval_tau=8.0,
        retrieval_attention_topk=1,
    )
    layer = MultiHeadDSRA2(cfg)
    q = torch.zeros(1, 1, 2, 8)
    q[:, :, 0, 0] = 1.0
    q[:, :, 1, 1] = 1.0
    retrieved_k = torch.zeros(1, 1, 2, 4, 8)
    retrieved_k[:, :, 0, 0, 0] = 1.0
    retrieved_k[:, :, 0, 1:, 2] = 1.0
    retrieved_k[:, :, 1, 0, 2] = 1.0
    retrieved_k[:, :, 1, 1, 1] = 1.0
    retrieved_k[:, :, 1, 2:, 3] = 1.0
    retrieved_v = torch.randn(1, 1, 2, 4, 8)
    retrieved_mask = torch.tensor([[[True, True, True, False], [True, True, True, True]]])

    _, weights = layer._retrieval_attention(
        q,
        retrieved_k,
        retrieved_v,
        retrieved_mask,
        return_weights=True,
    )

    assert weights.shape == (1, 1, 2, 4)
    assert int((weights[0, 0, 0] > 0).sum().item()) == 1
    assert int((weights[0, 0, 1] > 0).sum().item()) == 1
    assert float(weights[0, 0, 0, 0].item()) == 1.0
    assert float(weights[0, 0, 0, 3].item()) == 0.0
    assert float(weights[0, 0, 1, 1].item()) == 1.0


def test_retrieval_attention_topk_rejects_non_positive_value() -> None:
    """Invalid retrieval attention top-k config should fail before training.

    中文说明:
    - 调用方 / Called by: pytest。
    - 调用对象 / Calls: `MHDSRA2Config.__post_init__`。
    - 作用 / Purpose: 防止 `retrieval_attention_topk=0` 静默关闭或产生空
      softmax 候选，确保配置错误在模型构造前暴露。
    - 错误处理 / Error handling: 未抛出 `ValueError` 会触发断言。
    """
    with pytest.raises(ValueError, match="retrieval_attention_topk"):
        MHDSRA2Config(dim=8, heads=1, retrieval_attention_topk=0)


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
