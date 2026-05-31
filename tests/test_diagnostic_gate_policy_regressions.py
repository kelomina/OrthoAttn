from __future__ import annotations

from scripts.diagnostic_memory_benchmark import MODEL_LABELS, MODEL_ORDER, _build_mhdsra2_layer


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
