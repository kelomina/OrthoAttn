from __future__ import annotations

import inspect

import pytest

from scripts.json_retrieval_test import run_json_retrieval_generalization_test
from scripts.mhdsra2_quality_improvement_ablation import (
    DEFAULT_GROUPS,
    DEFAULT_REPORT_NAME,
    append_checkpoint_row,
    build_parser,
    build_run_rows,
    group_capability,
    load_checkpoint_rows,
    group_override,
    row_key,
    run_ablation,
    save_reports,
    summarize_slot_collision_diagnostics,
)
from scripts.needle_in_haystack_test import compute_retrieval_evidence_gate_loss
from scripts.toy_task_associative_recall import MHDSRA2CompatChunkLayer
from src.dsra.dsra_model import MultiLayerMHDSRA2Model
from src.dsra.mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2
from src.dsra.mhdsra2.paged_exact_memory import PagedExactMemory


def test_quality_ablation_group_overrides_are_explicit() -> None:
    assert group_override("baseline") == {}
    assert group_override("evidence_hit_supervision") == {}
    assert group_override("learned_retrieval_gate") == {
        "retrieval_quality_gate_adapter": True
    }
    assert group_override("evidence_plus_gate") == {
        "retrieval_quality_gate_adapter": True
    }
    assert group_override("retrieval_query_pooling") == {
        "retrieval_query_pooling": "max_token"
    }
    assert group_override("retrieval_gate_quality") == {
        "retrieval_quality_gate_bias": 2.0
    }
    assert group_override("combined") == {
        "retrieval_query_pooling": "max_token",
        "retrieval_quality_gate_bias": 2.0,
    }
    assert DEFAULT_GROUPS == (
        "baseline",
        "evidence_hit_supervision",
        "learned_retrieval_gate",
        "evidence_plus_gate",
    )
    assert group_capability("baseline", "retrieval_evidence_loss_alpha") == 0.0
    assert group_capability("evidence_hit_supervision", "retrieval_evidence_loss_alpha") > 0.0
    assert group_capability("evidence_plus_gate", "json_evidence_loss_weight") > 0.0


def test_quality_ablation_dry_run_expands_requested_matrix() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dry-run",
            "--tasks",
            "smoke,niah,json,two_digit",
            "--groups",
            "baseline,combined",
            "--niah-seq-lengths",
            "8192",
            "--niah-seeds",
            "101,202",
            "--json-task-seed-roots",
            "7",
            "--two-digit-seeds",
            "101",
        ]
    )

    rows = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)
    payload = run_ablation(args)

    assert len(rows) == 1 + 2 * 1 * 2 + 2 * 1 + 1 * 2 * 1 * 1 * 4
    assert payload["config"]["dry_run"] is True
    assert payload["rows"] == []
    assert len(payload["planned_rows"]) == len(rows)
    assert payload["planned_rows"][0]["task"] == "smoke"
    assert payload["planned_rows"][0]["status"] == "planned"
    assert "config" in payload["planned_rows"][0]


def test_quality_ablation_smoke_run_is_reported_as_row(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--reports-dir",
            str(tmp_path),
            "--checkpoint-path",
            str(tmp_path / "checkpoint.jsonl"),
            "--tasks",
            "smoke",
            "--groups",
            "baseline",
            "--device",
            "cpu",
            "--smoke-batch-sizes",
            "1",
            "--smoke-tokens",
            "64",
            "--smoke-page-size",
            "16",
            "--smoke-top-pages",
            "2",
            "--smoke-max-tokens",
            "4",
        ]
    )

    payload = run_ablation(args)

    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["group"] == "shared"
    assert row["task"] == "smoke"
    assert row["status"] == "passed"
    assert row["config"]["page_size"] == 16
    assert row["validation_metrics"]["passed"] is True
    assert row["test_metrics"] == {}


def test_quality_ablation_report_name_controls_output_paths(tmp_path) -> None:
    payload = {
        "name": "custom_evidence_report",
        "config": {
            "device": "cpu",
            "groups": ["baseline"],
            "tasks": ["smoke"],
            "dry_run": True,
        },
        "rows": [],
        "success_summary": {},
    }

    json_path, markdown_path = save_reports(payload, tmp_path)

    assert json_path.name == "custom_evidence_report.json"
    assert markdown_path.name == "custom_evidence_report.md"
    assert json_path.exists()
    assert markdown_path.exists()


def test_quality_ablation_parser_defaults_to_evidence_short_grid() -> None:
    args = build_parser().parse_args(["--dry-run"])

    assert args.report_name == DEFAULT_REPORT_NAME
    assert args.tasks == ("smoke", "niah", "json")
    assert args.groups == DEFAULT_GROUPS
    assert args.niah_seq_lengths == (8192,)
    assert args.niah_seeds == (101, 202, 303)
    assert args.niah_epochs == 60
    assert args.niah_robust_eval_interval == 20
    assert args.json_task_seed_roots == (7, 11, 19)
    assert args.two_digit_layers == (4, 8)
    assert args.two_digit_steps == (512,)
    assert args.two_digit_learning_rates == (0.01,)

    rows = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)
    assert len(rows) == 1 + 4 * 1 * 3 + 4 * 3
    evidence_niah_rows = [
        row
        for row in rows
        if row["task"] == "niah" and row["group"] == "evidence_hit_supervision"
    ]
    assert evidence_niah_rows
    assert all(
        row["config"]["retrieval_evidence_loss_alpha"] > 0.0
        for row in evidence_niah_rows
    )
    evidence_json_rows = [
        row
        for row in rows
        if row["task"] == "json" and row["group"] == "evidence_plus_gate"
    ]
    assert evidence_json_rows
    assert all(row["config"]["evidence_loss_weight"] > 0.0 for row in evidence_json_rows)


def test_quality_ablation_invalid_query_pooling_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="retrieval_query_pooling"):
        MultiLayerMHDSRA2Model(
            vocab_size=16,
            dim=16,
            num_layers=1,
            K=4,
            kr=1,
            chunk_size=8,
            mhdsra2_config_override={"retrieval_query_pooling": "invalid"},
        )

    with pytest.raises(ValueError, match="retrieval_query_pooling"):
        MHDSRA2CompatChunkLayer(
            dim=16,
            K=4,
            kr=1,
            local_window=8,
            mhdsra2_config_override={"retrieval_query_pooling": "invalid"},
        )


def test_json_generalization_entry_accepts_mhdsra2_override() -> None:
    signature = inspect.signature(run_json_retrieval_generalization_test)

    assert "mhdsra2_config_override" in signature.parameters
    assert signature.parameters["mhdsra2_config_override"].default is None


def test_quality_ablation_slot_collision_diagnostic_summarizes_usage() -> None:
    import torch

    summary = summarize_slot_collision_diagnostics(
        {
            "slot_usage": torch.tensor([[[10.0, 0.0, 0.0, 0.0]]]),
            "slot_confidence": torch.tensor([[[0.5, 0.0, 0.0, 0.0]]]),
        }
    )

    assert summary["available"] is True
    assert summary["effective_slot_count"] == pytest.approx(1.0)
    assert summary["top1_usage_share"] == pytest.approx(1.0)
    assert summary["collision_risk"] == "high"


def test_multilayer_selected_logits_return_aux_is_opt_in() -> None:
    import torch

    model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=4,
        kr=1,
        chunk_size=4,
        use_retrieval=True,
    )
    tokens = torch.arange(8, dtype=torch.long).view(1, 8) % 32
    positions = torch.tensor([7], dtype=torch.long)

    with torch.no_grad():
        logits = model.forward_selected_logits(tokens, positions)
        logits_with_aux, aux = model.forward_selected_logits(
            tokens,
            positions,
            return_aux=True,
        )

    torch.testing.assert_close(logits_with_aux, logits)
    assert aux["last_layer"] is not None
    assert "slot_usage" in aux["last_layer"]
    assert "selected_gate_retrieval_by_sample" in aux["last_layer"]
    torch.testing.assert_close(
        aux["last_layer"]["selected_gate_retrieval_by_sample"],
        aux["last_layer"]["gate_retrieval_by_token"][0, 3].view(1),
    )
    if "retrieval_metadata" in aux["last_layer"]:
        assert "selected_retrieval_metadata" in aux["last_layer"]


def test_paged_exact_memory_metadata_is_opt_in_and_keeps_legacy_tuple() -> None:
    import torch

    memory = PagedExactMemory(page_size=4, max_pages=8, dtype=torch.float32)
    keys = torch.randn(1, 2, 6, 8)
    values = torch.randn(1, 2, 6, 8)
    memory.append(keys, values)
    query = keys[:, :, 2:3, :]

    legacy_result = memory.retrieve(query, top_pages=2, max_tokens=3, return_mask=True)
    assert len(legacy_result) == 4

    metadata_result = memory.retrieve(
        query,
        top_pages=2,
        max_tokens=3,
        return_mask=True,
        return_metadata=True,
    )
    assert len(metadata_result) == 5
    _, _, positions, mask, metadata = metadata_result
    assert metadata["positions"].shape == positions.shape
    assert metadata["mask"].shape == mask.shape
    assert metadata["retrieved_token_counts"].shape == (1,)


def test_retrieval_evidence_gate_loss_uses_train_aux_metadata_only() -> None:
    import torch

    gate = torch.tensor([0.75], requires_grad=True)
    token_weights = torch.tensor([[0.20, 0.30, 0.50]], requires_grad=True)
    aux = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([1, 2, 3]),
                "mask": torch.tensor([True, True, True]),
            },
            "selected_gate_retrieval_by_sample_for_loss": gate,
            "selected_retrieval_token_weight_by_sample_for_loss": token_weights,
        },
        "validation_metrics": {"generation_exact_match_rate": 0.0},
        "test_metrics": {"generation_exact_match_rate": 1.0},
    }

    loss, metrics = compute_retrieval_evidence_gate_loss(
        aux,
        torch.tensor([2]),
        device=torch.device("cpu"),
    )
    loss.backward()

    assert metrics["available"] is True
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["evidence_weight_mean"] == pytest.approx(0.30)
    assert metrics["ranking_loss"] > 0.0
    assert gate.grad is not None
    assert abs(float(gate.grad.item())) > 0.0
    assert token_weights.grad is not None
    assert float(token_weights.grad[0, 1].item()) < 0.0


def test_retrieval_evidence_loss_prefers_selected_metadata() -> None:
    import torch

    gate = torch.tensor([0.60], requires_grad=True)
    token_weights = torch.tensor([[0.10, 0.90]], requires_grad=True)
    aux = {
        "last_layer": {
            "retrieval_metadata": {
                "positions": torch.tensor([[99, 100], [2, 3]]),
                "mask": torch.tensor([[True, True], [True, True]]),
            },
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[2, 3]]),
                "mask": torch.tensor([[True, True]]),
            },
            "selected_gate_retrieval_by_sample_for_loss": gate,
            "selected_retrieval_token_weight_by_sample_for_loss": token_weights,
        }
    }

    loss, metrics = compute_retrieval_evidence_gate_loss(
        aux,
        torch.tensor([3]),
        device=torch.device("cpu"),
    )
    loss.backward()

    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["evidence_weight_mean"] == pytest.approx(0.90)
    assert gate.grad is not None
    assert token_weights.grad is not None


def test_retrieval_evidence_loss_requires_selected_metadata() -> None:
    import torch

    aux = {
        "last_layer": {
            "retrieval_metadata": {
                "positions": torch.tensor([[3, 4]]),
                "mask": torch.tensor([[True, True]]),
            },
            "selected_gate_retrieval_by_sample_for_loss": torch.tensor(
                [0.5],
                requires_grad=True,
            ),
            "selected_retrieval_token_weight_by_sample_for_loss": torch.tensor(
                [[0.5, 0.5]],
                requires_grad=True,
            ),
        }
    }

    loss, metrics = compute_retrieval_evidence_gate_loss(
        aux,
        torch.tensor([3]),
        device=torch.device("cpu"),
    )

    assert loss.item() == pytest.approx(0.0)
    assert metrics["available"] is False
    assert metrics["unavailable_reason"] == "missing_selected_metadata_or_gate"


def test_retrieval_evidence_loss_rejects_selected_batch_mismatch() -> None:
    import torch

    aux = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[3, 4], [5, 6]]),
                "mask": torch.tensor([[True, True], [True, True]]),
            },
            "selected_gate_retrieval_by_sample_for_loss": torch.tensor(
                [0.5],
                requires_grad=True,
            ),
            "selected_retrieval_token_weight_by_sample_for_loss": torch.tensor(
                [[0.5, 0.5]],
                requires_grad=True,
            ),
        }
    }

    loss, metrics = compute_retrieval_evidence_gate_loss(
        aux,
        torch.tensor([3, 6]),
        device=torch.device("cpu"),
    )

    assert loss.item() == pytest.approx(0.0)
    assert metrics["available"] is False
    assert metrics["unavailable_reason"] == "gate_batch_mismatch"


def test_retrieval_evidence_loss_aligns_full_batch_evidence_by_selected_indices() -> None:
    import torch

    gate = torch.tensor([0.70], requires_grad=True)
    token_weights = torch.tensor([[0.25, 0.75]], requires_grad=True)
    aux = {
        "last_layer": {
            "selected_batch_indices": torch.tensor([1]),
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[8, 9]]),
                "mask": torch.tensor([[True, True]]),
            },
            "selected_gate_retrieval_by_sample_for_loss": gate,
            "selected_retrieval_token_weight_by_sample_for_loss": token_weights,
        }
    }

    loss, metrics = compute_retrieval_evidence_gate_loss(
        aux,
        torch.tensor([4, 9, 12]),
        device=torch.device("cpu"),
    )
    loss.backward()

    assert metrics["available"] is True
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["evidence_weight_mean"] == pytest.approx(0.75)
    assert gate.grad is not None
    assert token_weights.grad is not None


def test_zero_initialized_retrieval_gate_adapter_does_not_change_default_logits() -> None:
    import torch

    torch.manual_seed(123)
    cfg = MHDSRA2Config(dim=16, heads=2, slots=4, read_topk=2)
    baseline = MultiHeadDSRA2(cfg)
    adapted = MultiHeadDSRA2(
        MHDSRA2Config(
            dim=16,
            heads=2,
            slots=4,
            read_topk=2,
            retrieval_quality_gate_adapter=True,
        )
    )
    adapted.load_state_dict(baseline.state_dict(), strict=False)

    x = torch.randn(2, 3, 16)
    retrieved_k = torch.randn(2, 2, 4, 8)
    retrieved_v = torch.randn(2, 2, 4, 8)
    retrieved_mask = torch.ones(2, 4, dtype=torch.bool)

    with torch.no_grad():
        baseline_out, _, baseline_aux = baseline(
            x,
            retrieved_k=retrieved_k,
            retrieved_v=retrieved_v,
            retrieved_mask=retrieved_mask,
            return_aux=True,
        )
        adapted_out, _, adapted_aux = adapted(
            x,
            retrieved_k=retrieved_k,
            retrieved_v=retrieved_v,
            retrieved_mask=retrieved_mask,
            return_aux=True,
        )

    torch.testing.assert_close(adapted_out, baseline_out)
    assert "retrieval_quality_adapter_delta" not in baseline_aux or baseline_aux[
        "retrieval_quality_adapter_delta"
    ] is None
    assert adapted_aux["retrieval_quality_adapter_delta"] is not None
    torch.testing.assert_close(
        adapted_aux["retrieval_quality_adapter_delta"],
        torch.zeros_like(adapted_aux["retrieval_quality_adapter_delta"]),
    )


def test_retrieval_aux_exposes_token_weights_only_when_requested() -> None:
    import torch

    torch.manual_seed(456)
    layer = MultiHeadDSRA2(MHDSRA2Config(dim=16, heads=2, slots=4, read_topk=2))
    x = torch.randn(1, 2, 16)
    retrieved_k = torch.randn(1, 2, 3, 8)
    retrieved_v = torch.randn(1, 2, 3, 8)
    retrieved_mask = torch.tensor([[True, True, False]])

    with torch.no_grad():
        output, _ = layer(
            x,
            retrieved_k=retrieved_k,
            retrieved_v=retrieved_v,
            retrieved_mask=retrieved_mask,
            return_aux=False,
        )
        output_with_aux, _, aux = layer(
            x,
            retrieved_k=retrieved_k,
            retrieved_v=retrieved_v,
            retrieved_mask=retrieved_mask,
            return_aux=True,
        )

    torch.testing.assert_close(output_with_aux, output)
    token_weights = aux["retrieval_token_weight_by_sample"]
    assert token_weights.shape == (1, 3)
    assert token_weights[0, 2].item() == pytest.approx(0.0, abs=1e-6)
    assert token_weights[0, :2].sum().item() == pytest.approx(1.0, rel=1e-5)


def test_quality_ablation_checkpoint_rows_round_trip(tmp_path) -> None:
    row = {
        "group": "baseline",
        "task": "json",
        "seed": 7,
        "status": "completed",
        "config": {"epochs": 1, "mhdsra2_config_override": {}},
        "validation_metrics": {"validation_generation_exact_match_rate": 1.0},
        "test_metrics": {},
    }
    checkpoint_path = tmp_path / "rows.jsonl"

    append_checkpoint_row(checkpoint_path, row)
    loaded = load_checkpoint_rows(checkpoint_path)

    assert loaded[row_key(row)]["status"] == "completed"
    assert loaded[row_key(row)]["validation_metrics"][
        "validation_generation_exact_match_rate"
    ] == 1.0
