from __future__ import annotations

import inspect

import pytest
import torch

from scripts import mhdsra2_quality_improvement_ablation as ablation
from scripts import json_retrieval_test as json_task
from scripts import needle_in_haystack_test as niah_task
from scripts.json_retrieval_test import run_json_retrieval_generalization_test
from scripts.mhdsra2_quality_improvement_ablation import (
    DEFAULT_GROUPS,
    DEFAULT_REPORT_NAME,
    append_checkpoint_row,
    build_parser,
    build_run_rows,
    group_capability,
    group_uses_niah_retrieval,
    json_group_capabilities,
    load_checkpoint_rows,
    group_override,
    row_key,
    run_ablation,
    run_json_row,
    save_reports,
    split_json_row_metrics,
    summarize_slot_collision_diagnostics,
)
from scripts.needle_in_haystack_test import (
    compute_query_evidence_alignment_loss,
    compute_retrieval_span_predictor_loss,
    compute_retrieval_evidence_gate_loss,
    compute_retrieval_projection_contrastive_loss,
    summarize_retrieval_span_predictor_step_metrics,
    summarize_retrieval_evidence_step_metrics,
    summarize_retrieval_projection_step_metrics,
)
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
    assert group_override("evidence_rank_margin") == {}
    assert group_override("evidence_rank_margin_needle_copy") == {}
    assert group_override("evidence_score_margin") == {}
    assert group_override("evidence_score_margin_needle_copy") == {}
    assert group_override("retrieval_neighbor_span") == {"retrieval_neighbor_span": 1}
    assert group_override("retrieval_neighbor_span_needle_copy") == {
        "retrieval_neighbor_span": 1
    }
    assert group_override("evidence_score_margin_neighbor_span_needle_copy") == {
        "retrieval_neighbor_span": 1
    }
    assert group_override("retrieval_neighbor_span_pair_copy") == {
        "retrieval_neighbor_span": 1
    }
    assert group_override("evidence_score_margin_neighbor_span_pair_copy") == {
        "retrieval_neighbor_span": 1
    }
    assert group_override("evidence_key_score_margin_neighbor_span_pair_copy") == {
        "retrieval_neighbor_span": 1
    }
    assert group_override("slot_readout_bias") == {}
    assert group_override("evidence_slot_readout") == {}
    assert group_override("extract_compose_readout") == {}
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
    assert group_override("retrieval_bidirectional_structured_span_predictor") == {
        "retrieval_neighbor_span": 1,
        "retrieval_neighbor_direction": "both",
    }
    assert group_override("retrieval_page_local_neighbor_span_predictor") == {
        "retrieval_neighbor_span": 1,
        "retrieval_neighbor_direction": "both",
        "retrieval_neighbor_seed_multiplier": 4,
    }
    assert group_override("retrieval_structured_feature_span_predictor") == {
        "retrieval_neighbor_span": 1,
        "retrieval_neighbor_direction": "both",
        "retrieval_neighbor_seed_multiplier": 4,
    }
    assert group_override("retrieval_compact_page_local_span_predictor") == {
        "retrieval_neighbor_span": 1,
        "retrieval_neighbor_direction": "both",
        "retrieval_neighbor_seed_multiplier": 4,
        "retrieval_max_tokens": 32,
    }
    assert group_override("retrieval_pair_aware_page_local_span_predictor") == {
        "retrieval_neighbor_span": 1,
        "retrieval_neighbor_direction": "both",
        "retrieval_neighbor_seed_multiplier": 4,
        "retrieval_neighbor_budget_mode": "pair_aware",
    }
    assert DEFAULT_GROUPS == (
        "baseline",
        "evidence_hit_supervision",
        "learned_retrieval_gate",
        "evidence_plus_gate",
    )
    assert group_capability("baseline", "retrieval_evidence_loss_alpha") == 0.0
    assert group_capability("baseline", "json_slot_decoder_loss_weight") == 0.0
    assert group_capability("evidence_hit_supervision", "retrieval_evidence_loss_alpha") > 0.0
    assert group_capability("evidence_hit_supervision", "retrieval_evidence_rank_margin") == 0.0
    assert group_capability("evidence_rank_margin", "retrieval_evidence_loss_alpha") > 0.0
    assert group_capability("evidence_rank_margin", "retrieval_evidence_rank_margin") > 0.0
    assert group_capability("evidence_rank_margin", "retrieval_evidence_score_margin") == 0.0
    assert group_capability("evidence_score_margin", "retrieval_evidence_loss_alpha") > 0.0
    assert group_capability("evidence_score_margin", "retrieval_evidence_rank_margin") == 0.0
    assert group_capability("evidence_score_margin", "retrieval_evidence_score_margin") > 0.0
    assert group_capability("baseline", "neighbor_span") == 0
    assert group_capability("retrieval_neighbor_span", "neighbor_span") == 1
    assert group_capability("retrieval_neighbor_span", "retrieval_evidence_loss_alpha") == 0.0
    assert group_capability("retrieval_neighbor_span_needle_copy", "neighbor_span") == 1
    assert (
        group_capability("retrieval_neighbor_span_needle_copy", "niah_readout_mode")
        == "needle_copy"
    )
    assert (
        group_capability("retrieval_neighbor_span_pair_copy", "niah_readout_mode")
        == "needle_pair_copy"
    )
    assert group_capability("baseline", "query_evidence_alignment_alpha", 0.0) == 0.0
    assert (
        group_capability(
            "query_evidence_alignment_pair_copy",
            "query_evidence_alignment_alpha",
        )
        > 0.0
    )
    assert group_capability("baseline", "retrieval_projection_contrastive_alpha", 0.0) == 0.0
    assert group_capability("baseline", "retrieval_span_predictor_alpha", 0.0) == 0.0
    assert group_capability("baseline", "neighbor_direction", "right") == "right"
    assert group_capability(
        "retrieval_page_local_neighbor_span_predictor",
        "neighbor_seed_multiplier",
    ) == 4
    assert (
        group_capability(
            "retrieval_projection_contrastive",
            "retrieval_projection_contrastive_alpha",
        )
        > 0.0
    )
    assert (
        group_capability(
            "retrieval_projection_pair_copy",
            "retrieval_projection_contrastive_alpha",
        )
        > 0.0
    )
    assert (
        group_capability("retrieval_projection_pair_copy", "niah_readout_mode")
        == "needle_pair_copy"
    )
    assert (
        group_capability("retrieval_span_predictor", "retrieval_span_predictor_alpha")
        > 0.0
    )
    assert (
        group_capability("retrieval_span_predictor", "niah_readout_mode")
        == "span_predictor"
    )
    assert (
        group_capability("retrieval_span_predictor", "niah_span_candidate_filter")
        == "all"
    )
    assert (
        group_capability(
            "retrieval_structured_span_predictor",
            "retrieval_span_predictor_alpha",
        )
        > 0.0
    )
    assert (
        group_capability(
            "retrieval_structured_span_predictor",
            "niah_span_candidate_filter",
        )
        == "key_value_pair"
    )
    assert (
        group_capability(
            "retrieval_prefer_structured_span_predictor",
            "niah_span_candidate_filter",
        )
        == "prefer_key_value_pair"
    )
    assert (
        group_capability(
            "retrieval_multi_positive_span_predictor",
            "niah_span_loss_mode",
        )
        == "multi_positive"
    )
    assert (
        group_capability(
            "retrieval_bidirectional_structured_span_predictor",
            "neighbor_direction",
        )
        == "both"
    )
    assert (
        group_capability(
            "retrieval_structured_feature_span_predictor",
            "retrieval_span_structure_features",
            False,
        )
        is True
    )
    assert group_capability("baseline", "retrieval_max_tokens", 128) == 128
    assert (
        group_capability(
            "retrieval_compact_page_local_span_predictor",
            "retrieval_max_tokens",
        )
        == 32
    )
    assert group_capability("baseline", "neighbor_budget_mode", "unbounded") == "unbounded"
    assert (
        group_capability(
            "retrieval_pair_aware_page_local_span_predictor",
            "neighbor_budget_mode",
        )
        == "pair_aware"
    )
    assert group_capability("evidence_hit_supervision", "retrieval_evidence_target_offset") == 1
    assert (
        group_capability(
            "evidence_key_score_margin_neighbor_span_pair_copy",
            "retrieval_evidence_target_offset",
        )
        == 0
    )
    assert group_capability("evidence_plus_gate", "json_evidence_loss_weight") > 0.0
    assert group_capability("slot_readout_bias", "json_slot_decoder_loss_weight") > 0.0
    assert group_capability("slot_readout_bias", "json_slot_decoder_logit_bias") > 0.0
    assert group_capability("evidence_slot_readout", "json_evidence_loss_weight") > 0.0
    assert group_capability("extract_compose_readout", "json_evidence_loss_weight") > 0.0
    assert (
        group_capability("extract_compose_readout", "json_generation_readout_mode")
        == "extract_then_compose"
    )
    assert group_capability("needle_copy_readout", "niah_readout_mode") == "needle_copy"
    assert (
        group_capability("evidence_needle_copy_readout", "niah_readout_mode")
        == "needle_copy"
    )
    assert (
        group_capability(
            "evidence_needle_copy_readout",
            "retrieval_evidence_loss_alpha",
        )
        > 0.0
    )
    assert (
        group_capability(
            "evidence_rank_margin_needle_copy",
            "retrieval_evidence_rank_margin",
        )
        > 0.0
    )
    assert (
        group_capability(
            "evidence_score_margin_needle_copy",
            "retrieval_evidence_score_margin",
        )
        > 0.0
    )
    assert (
        group_capability("evidence_score_margin_needle_copy", "niah_readout_mode")
        == "needle_copy"
    )
    assert "slot_readout_bias" not in DEFAULT_GROUPS
    assert "evidence_slot_readout" not in DEFAULT_GROUPS
    assert "extract_compose_readout" not in DEFAULT_GROUPS
    assert "needle_copy_readout" not in DEFAULT_GROUPS
    assert "evidence_needle_copy_readout" not in DEFAULT_GROUPS
    assert "evidence_rank_margin" not in DEFAULT_GROUPS
    assert "evidence_rank_margin_needle_copy" not in DEFAULT_GROUPS
    assert "evidence_score_margin" not in DEFAULT_GROUPS
    assert "evidence_score_margin_needle_copy" not in DEFAULT_GROUPS
    assert "retrieval_neighbor_span" not in DEFAULT_GROUPS
    assert "retrieval_neighbor_span_needle_copy" not in DEFAULT_GROUPS
    assert "evidence_score_margin_neighbor_span_needle_copy" not in DEFAULT_GROUPS
    assert "retrieval_neighbor_span_pair_copy" not in DEFAULT_GROUPS
    assert "evidence_score_margin_neighbor_span_pair_copy" not in DEFAULT_GROUPS
    assert "evidence_key_score_margin_neighbor_span_pair_copy" not in DEFAULT_GROUPS
    assert "query_evidence_alignment_pair_copy" not in DEFAULT_GROUPS
    assert "retrieval_projection_contrastive" not in DEFAULT_GROUPS
    assert "retrieval_projection_pair_copy" not in DEFAULT_GROUPS
    assert "retrieval_span_predictor" not in DEFAULT_GROUPS
    assert "retrieval_structured_span_predictor" not in DEFAULT_GROUPS
    assert "retrieval_prefer_structured_span_predictor" not in DEFAULT_GROUPS
    assert "retrieval_multi_positive_span_predictor" not in DEFAULT_GROUPS
    assert "retrieval_bidirectional_structured_span_predictor" not in DEFAULT_GROUPS
    assert group_uses_niah_retrieval("baseline") is False
    assert group_uses_niah_retrieval("evidence_hit_supervision") is True
    assert group_uses_niah_retrieval("evidence_rank_margin") is True
    assert group_uses_niah_retrieval("learned_retrieval_gate") is True
    assert group_uses_niah_retrieval("slot_readout_bias") is False
    assert group_uses_niah_retrieval("needle_copy_readout") is True
    assert group_uses_niah_retrieval("evidence_needle_copy_readout") is True
    assert group_uses_niah_retrieval("evidence_rank_margin_needle_copy") is True
    assert group_uses_niah_retrieval("evidence_score_margin") is True
    assert group_uses_niah_retrieval("evidence_score_margin_needle_copy") is True
    assert group_uses_niah_retrieval("retrieval_neighbor_span") is True
    assert group_uses_niah_retrieval("retrieval_neighbor_span_needle_copy") is True
    assert group_uses_niah_retrieval("evidence_score_margin_neighbor_span_needle_copy") is True
    assert group_uses_niah_retrieval("retrieval_neighbor_span_pair_copy") is True
    assert group_uses_niah_retrieval("evidence_score_margin_neighbor_span_pair_copy") is True
    assert group_uses_niah_retrieval("evidence_key_score_margin_neighbor_span_pair_copy") is True
    assert group_uses_niah_retrieval("query_evidence_alignment_pair_copy") is True
    assert group_uses_niah_retrieval("retrieval_projection_contrastive") is True
    assert group_uses_niah_retrieval("retrieval_projection_pair_copy") is True
    assert group_uses_niah_retrieval("retrieval_span_predictor") is True
    assert group_uses_niah_retrieval("retrieval_structured_span_predictor") is True
    assert group_uses_niah_retrieval("retrieval_prefer_structured_span_predictor") is True
    assert group_uses_niah_retrieval("retrieval_multi_positive_span_predictor") is True
    assert group_uses_niah_retrieval(
        "retrieval_bidirectional_structured_span_predictor"
    ) is True


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
    niah_rows = [row for row in payload["planned_rows"] if row["task"] == "niah"]
    assert niah_rows
    assert all(row["config"]["test_batches_per_depth"] == 0 for row in niah_rows)


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
    baseline_niah_rows = [
        row
        for row in rows
        if row["task"] == "niah" and row["group"] == "baseline"
    ]
    assert baseline_niah_rows
    assert all(row["config"]["use_retrieval"] is False for row in baseline_niah_rows)
    assert all(row["config"]["use_retrieval"] is True for row in evidence_niah_rows)
    evidence_json_rows = [
        row
        for row in rows
        if row["task"] == "json" and row["group"] == "evidence_plus_gate"
    ]
    assert evidence_json_rows
    assert all(row["config"]["evidence_loss_weight"] > 0.0 for row in evidence_json_rows)


def test_quality_ablation_json_readout_groups_expand_explicit_config() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dry-run",
            "--tasks",
            "json",
            "--groups",
            "baseline,slot_readout_bias,evidence_slot_readout",
            "--json-task-seed-roots",
            "7",
        ]
    )

    rows = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)
    by_group = {row["group"]: row for row in rows}

    assert by_group["baseline"]["config"]["slot_decoder_loss_weight"] == 0.0
    assert by_group["baseline"]["config"]["slot_decoder_logit_bias"] == 0.0
    assert by_group["baseline"]["config"]["evidence_hint_weight"] == 0.0
    assert by_group["baseline"]["config"]["generation_readout_mode"] == "model"
    assert by_group["slot_readout_bias"]["config"]["evidence_loss_weight"] == 0.0
    assert by_group["slot_readout_bias"]["config"]["slot_decoder_loss_weight"] > 0.0
    assert by_group["slot_readout_bias"]["config"]["slot_decoder_logit_bias"] > 0.0
    assert by_group["evidence_slot_readout"]["config"]["evidence_loss_weight"] > 0.0
    assert by_group["evidence_slot_readout"]["config"][
        "slot_decoder_loss_weight"
    ] == group_capability("slot_readout_bias", "json_slot_decoder_loss_weight")
    assert json_group_capabilities("slot_readout_bias") == {
        "evidence_loss_weight": 0.0,
        "slot_decoder_loss_weight": group_capability(
            "slot_readout_bias",
            "json_slot_decoder_loss_weight",
        ),
        "slot_decoder_logit_bias": group_capability(
            "slot_readout_bias",
            "json_slot_decoder_logit_bias",
        ),
        "evidence_hint_weight": 0.0,
        "generation_readout_mode": "model",
    }


def test_quality_ablation_extract_compose_readout_group_is_explicit() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dry-run",
            "--tasks",
            "json",
            "--groups",
            "baseline,extract_compose_readout",
            "--json-task-seed-roots",
            "7",
        ]
    )

    rows = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)
    by_group = {row["group"]: row for row in rows}

    assert by_group["baseline"]["config"]["generation_readout_mode"] == "model"
    assert by_group["baseline"]["config"]["answer_template_mode"] == "canonical"
    extract_config = by_group["extract_compose_readout"]["config"]
    assert extract_config["generation_readout_mode"] == "extract_then_compose"
    assert extract_config["distractor_records_per_case"] == 0
    assert extract_config["answer_template_mode"] == "canonical"
    assert extract_config["evidence_loss_weight"] > 0.0
    assert extract_config["slot_decoder_loss_weight"] == 0.0
    assert extract_config["slot_decoder_logit_bias"] == 0.0


def test_json_generation_readout_adapter_registry_is_default_off() -> None:
    assert json_task.get_generation_readout_adapter_config("model") is None
    assert json_task.DEFAULT_GENERATION_READOUT_MODE in json_task.ALLOWED_GENERATION_READOUT_MODES

    adapter_config = json_task.get_generation_readout_adapter_config("extract_then_compose")

    assert adapter_config["diagnostic_key"] == "extract_then_compose"
    assert adapter_config["metric_prefix"] == "extract_then_compose"
    assert adapter_config["slot_metric_names"] == json_task.ANSWER_SLOT_NAMES
    assert set(json_task.ALLOWED_GENERATION_READOUT_MODES) == {
        "model",
        *json_task.GENERATION_READOUT_ADAPTER_CONFIGS.keys(),
    }
    with pytest.raises(ValueError, match="Unsupported generation_readout_mode"):
        json_task.get_generation_readout_adapter_config("unknown_adapter")


def test_json_single_case_model_readout_does_not_call_adapter(monkeypatch) -> None:
    case = {
        "sample_bytes": b"context",
        "question_bytes": b"question",
        "expected_answer_bytes": b"ANSWER",
        "metadata": {
            "question": "question",
            "museum": "Palace Museum",
            "artifact": "Autumn Lantern Procession",
            "artist": "Lin Qiao",
            "dynasty": "Tang",
            "expected_answer_text": "ANSWER",
            "needle_position_pct": 0.5,
        },
    }

    def fail_adapter(*_args, **_kwargs):
        raise AssertionError("default model readout must not call structured adapter")

    monkeypatch.setattr(json_task, "evaluate_generation_readout_adapter", fail_adapter)
    monkeypatch.setattr(
        json_task,
        "build_training_example",
        lambda _case: (
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
        ),
    )
    monkeypatch.setattr(
        json_task,
        "forward_json_retrieval",
        lambda *_args, **_kwargs: (
            torch.zeros(1, 1, json_task.VOCAB_SIZE),
            torch.zeros(1, 1, 4),
            {},
        ),
    )
    monkeypatch.setattr(
        json_task,
        "evaluate_teacher_forced",
        lambda *_args, **_kwargs: {
            "exact_byte_match": False,
            "sequence_accuracy": 0.0,
            "prefix_match_length": 0,
            "entity_span_metrics": {},
        },
    )
    monkeypatch.setattr(
        json_task,
        "evaluate_generation",
        lambda *_args, **_kwargs: {
            "exact_byte_match": False,
            "sequence_accuracy": 0.0,
            "prefix_match_length": 0,
            "entity_span_metrics": {},
        },
    )
    monkeypatch.setattr(json_task, "evaluate_entity_auxiliary", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(json_task, "evaluate_evidence_decoder", lambda *_args, **_kwargs: {"available": False})
    monkeypatch.setattr(json_task, "evaluate_slot_decoder", lambda *_args, **_kwargs: {"available": False})
    monkeypatch.setattr(json_task, "get_answer_entity_spans", lambda _case: {})

    result = json_task.evaluate_single_case(
        model=object(),
        case=case,
        device=torch.device("cpu"),
        generation_readout_mode="model",
    )

    assert "extract_then_compose" not in result
    assert result["generation"]["exact_byte_match"] is False


def test_quality_ablation_needle_copy_readout_group_is_explicit() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dry-run",
            "--tasks",
            "niah",
            "--groups",
            "baseline,needle_copy_readout,evidence_needle_copy_readout",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
        ]
    )

    rows = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)
    by_group = {row["group"]: row for row in rows}

    assert by_group["baseline"]["config"]["niah_readout_mode"] == "model"
    assert by_group["baseline"]["config"]["use_retrieval"] is False
    copy_config = by_group["needle_copy_readout"]["config"]
    assert copy_config["niah_readout_mode"] == "needle_copy"
    assert copy_config["use_retrieval"] is True
    assert copy_config["retrieval_evidence_loss_alpha"] == 0.0
    evidence_copy_config = by_group["evidence_needle_copy_readout"]["config"]
    assert evidence_copy_config["niah_readout_mode"] == "needle_copy"
    assert evidence_copy_config["use_retrieval"] is True
    assert evidence_copy_config["retrieval_evidence_loss_alpha"] > 0.0


def test_quality_ablation_rank_margin_group_is_explicit() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dry-run",
            "--tasks",
            "niah",
            "--groups",
            "baseline,evidence_rank_margin,evidence_rank_margin_needle_copy",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
        ]
    )

    rows = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)
    by_group = {row["group"]: row for row in rows}

    baseline_config = by_group["baseline"]["config"]
    assert baseline_config["retrieval_evidence_rank_margin"] == 0.0
    assert baseline_config["use_retrieval"] is False
    rank_config = by_group["evidence_rank_margin"]["config"]
    assert rank_config["retrieval_evidence_loss_alpha"] > 0.0
    assert rank_config["retrieval_evidence_rank_margin"] > 0.0
    assert rank_config["niah_readout_mode"] == "model"
    assert rank_config["use_retrieval"] is True
    copy_config = by_group["evidence_rank_margin_needle_copy"]["config"]
    assert copy_config["retrieval_evidence_rank_margin"] == rank_config[
        "retrieval_evidence_rank_margin"
    ]
    assert copy_config["niah_readout_mode"] == "needle_copy"
    assert copy_config["use_retrieval"] is True


def test_quality_ablation_score_margin_group_is_explicit() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dry-run",
            "--tasks",
            "niah",
            "--groups",
            "baseline,evidence_score_margin,evidence_score_margin_needle_copy",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
        ]
    )

    rows = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)
    by_group = {row["group"]: row for row in rows}

    baseline_config = by_group["baseline"]["config"]
    assert baseline_config["retrieval_evidence_score_margin"] == 0.0
    assert baseline_config["use_retrieval"] is False
    score_config = by_group["evidence_score_margin"]["config"]
    assert score_config["retrieval_evidence_loss_alpha"] > 0.0
    assert score_config["retrieval_evidence_rank_margin"] == 0.0
    assert score_config["retrieval_evidence_score_margin"] > 0.0
    assert score_config["niah_readout_mode"] == "model"
    assert score_config["use_retrieval"] is True
    copy_config = by_group["evidence_score_margin_needle_copy"]["config"]
    assert copy_config["retrieval_evidence_score_margin"] == score_config[
        "retrieval_evidence_score_margin"
    ]
    assert copy_config["niah_readout_mode"] == "needle_copy"
    assert copy_config["use_retrieval"] is True


def test_quality_ablation_neighbor_span_group_is_explicit() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dry-run",
            "--tasks",
            "niah",
            "--groups",
            (
                "baseline,retrieval_neighbor_span,"
                "retrieval_neighbor_span_needle_copy,"
                "evidence_score_margin_neighbor_span_needle_copy,"
                "retrieval_neighbor_span_pair_copy,"
                "evidence_score_margin_neighbor_span_pair_copy,"
                "evidence_key_score_margin_neighbor_span_pair_copy,"
                "query_evidence_alignment_pair_copy,"
                "retrieval_projection_contrastive,"
                "retrieval_projection_pair_copy,"
                "retrieval_span_predictor,"
                "retrieval_structured_span_predictor,"
                "retrieval_prefer_structured_span_predictor,"
                "retrieval_multi_positive_span_predictor,"
                    "retrieval_bidirectional_structured_span_predictor,"
                    "retrieval_page_local_neighbor_span_predictor,"
                    "retrieval_structured_feature_span_predictor,"
                    "retrieval_compact_page_local_span_predictor,"
                    "retrieval_pair_aware_page_local_span_predictor"
                ),
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
        ]
    )

    rows = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)
    by_group = {row["group"]: row for row in rows}

    baseline_config = by_group["baseline"]["config"]
    assert baseline_config["retrieval_neighbor_span"] == 0
    assert baseline_config["use_retrieval"] is False

    neighbor_config = by_group["retrieval_neighbor_span"]["config"]
    assert neighbor_config["retrieval_neighbor_span"] == 1
    assert neighbor_config["use_retrieval"] is True
    assert neighbor_config["mhdsra2_config_override"] == {"retrieval_neighbor_span": 1}
    assert neighbor_config["retrieval_evidence_loss_alpha"] == 0.0
    assert neighbor_config["niah_readout_mode"] == "model"

    copy_config = by_group["retrieval_neighbor_span_needle_copy"]["config"]
    assert copy_config["retrieval_neighbor_span"] == 1
    assert copy_config["niah_readout_mode"] == "needle_copy"
    assert copy_config["use_retrieval"] is True

    combined_config = by_group["evidence_score_margin_neighbor_span_needle_copy"]["config"]
    assert combined_config["retrieval_neighbor_span"] == 1
    assert combined_config["retrieval_evidence_score_margin"] > 0.0
    assert combined_config["retrieval_evidence_loss_alpha"] > 0.0
    assert combined_config["niah_readout_mode"] == "needle_copy"

    pair_config = by_group["retrieval_neighbor_span_pair_copy"]["config"]
    assert pair_config["retrieval_neighbor_span"] == 1
    assert pair_config["retrieval_evidence_loss_alpha"] == 0.0
    assert pair_config["niah_readout_mode"] == "needle_pair_copy"
    assert pair_config["use_retrieval"] is True

    evidence_pair_config = by_group[
        "evidence_score_margin_neighbor_span_pair_copy"
    ]["config"]
    assert evidence_pair_config["retrieval_neighbor_span"] == 1
    assert evidence_pair_config["retrieval_evidence_loss_alpha"] > 0.0
    assert evidence_pair_config["retrieval_evidence_score_margin"] > 0.0
    assert evidence_pair_config["retrieval_evidence_target_offset"] == 1
    assert evidence_pair_config["niah_readout_mode"] == "needle_pair_copy"
    assert evidence_pair_config["use_retrieval"] is True

    key_pair_config = by_group[
        "evidence_key_score_margin_neighbor_span_pair_copy"
    ]["config"]
    assert key_pair_config["retrieval_neighbor_span"] == 1
    assert key_pair_config["retrieval_evidence_loss_alpha"] > 0.0
    assert key_pair_config["retrieval_evidence_score_margin"] > 0.0
    assert key_pair_config["retrieval_evidence_target_offset"] == 0
    assert key_pair_config["niah_readout_mode"] == "needle_pair_copy"
    assert key_pair_config["use_retrieval"] is True

    alignment_config = by_group["query_evidence_alignment_pair_copy"]["config"]
    assert alignment_config["retrieval_neighbor_span"] == 1
    assert alignment_config["retrieval_evidence_loss_alpha"] == 0.0
    assert alignment_config["query_evidence_alignment_alpha"] > 0.0
    assert alignment_config["retrieval_evidence_target_offset"] == 1
    assert alignment_config["niah_readout_mode"] == "needle_pair_copy"
    assert alignment_config["use_retrieval"] is True

    projection_config = by_group["retrieval_projection_contrastive"]["config"]
    assert projection_config["retrieval_neighbor_span"] == 0
    assert projection_config["retrieval_evidence_loss_alpha"] == 0.0
    assert projection_config["retrieval_projection_contrastive_alpha"] > 0.0
    assert projection_config["retrieval_projection_temperature"] > 0.0
    assert projection_config["niah_readout_mode"] == "model"
    assert projection_config["use_retrieval"] is True

    projection_pair_config = by_group["retrieval_projection_pair_copy"]["config"]
    assert projection_pair_config["retrieval_neighbor_span"] == 1
    assert projection_pair_config["retrieval_projection_contrastive_alpha"] > 0.0
    assert projection_pair_config["niah_readout_mode"] == "needle_pair_copy"
    assert projection_pair_config["use_retrieval"] is True

    span_config = by_group["retrieval_span_predictor"]["config"]
    assert span_config["retrieval_neighbor_span"] == 1
    assert span_config["retrieval_span_predictor_alpha"] > 0.0
    assert span_config["retrieval_projection_contrastive_alpha"] == 0.0
    assert span_config["niah_readout_mode"] == "span_predictor"
    assert span_config["niah_span_candidate_filter"] == "all"
    assert span_config["use_retrieval"] is True

    structured_span_config = by_group["retrieval_structured_span_predictor"]["config"]
    assert structured_span_config["retrieval_neighbor_span"] == 1
    assert structured_span_config["retrieval_span_predictor_alpha"] > 0.0
    assert structured_span_config["retrieval_projection_contrastive_alpha"] == 0.0
    assert structured_span_config["niah_readout_mode"] == "span_predictor"
    assert structured_span_config["niah_span_candidate_filter"] == "key_value_pair"
    assert structured_span_config["use_retrieval"] is True

    prefer_structured_span_config = by_group[
        "retrieval_prefer_structured_span_predictor"
    ]["config"]
    assert prefer_structured_span_config["retrieval_neighbor_span"] == 1
    assert prefer_structured_span_config["retrieval_span_predictor_alpha"] > 0.0
    assert prefer_structured_span_config["retrieval_projection_contrastive_alpha"] == 0.0
    assert prefer_structured_span_config["niah_readout_mode"] == "span_predictor"
    assert (
        prefer_structured_span_config["niah_span_candidate_filter"]
        == "prefer_key_value_pair"
    )
    assert prefer_structured_span_config["niah_span_loss_mode"] == "single_positive"
    assert prefer_structured_span_config["use_retrieval"] is True

    multi_positive_config = by_group["retrieval_multi_positive_span_predictor"][
        "config"
    ]
    assert multi_positive_config["retrieval_neighbor_span"] == 1
    assert multi_positive_config["retrieval_span_predictor_alpha"] > 0.0
    assert multi_positive_config["niah_readout_mode"] == "span_predictor"
    assert multi_positive_config["niah_span_candidate_filter"] == "prefer_key_value_pair"
    assert multi_positive_config["niah_span_loss_mode"] == "multi_positive"
    assert multi_positive_config["use_retrieval"] is True

    bidirectional_config = by_group[
        "retrieval_bidirectional_structured_span_predictor"
    ]["config"]
    assert bidirectional_config["retrieval_neighbor_span"] == 1
    assert bidirectional_config["retrieval_neighbor_direction"] == "both"
    assert bidirectional_config["retrieval_span_predictor_alpha"] > 0.0
    assert bidirectional_config["niah_readout_mode"] == "span_predictor"
    assert (
        bidirectional_config["niah_span_candidate_filter"]
        == "prefer_key_value_pair"
    )
    assert bidirectional_config["niah_span_loss_mode"] == "multi_positive"
    assert bidirectional_config["mhdsra2_config_override"] == {
        "retrieval_neighbor_span": 1,
        "retrieval_neighbor_direction": "both",
    }
    assert bidirectional_config["use_retrieval"] is True

    page_local_config = by_group["retrieval_page_local_neighbor_span_predictor"][
        "config"
    ]
    assert page_local_config["retrieval_neighbor_span"] == 1
    assert page_local_config["retrieval_neighbor_direction"] == "both"
    assert page_local_config["retrieval_neighbor_seed_multiplier"] == 4
    assert page_local_config["niah_readout_mode"] == "span_predictor"
    assert page_local_config["niah_span_candidate_filter"] == "prefer_key_value_pair"
    assert page_local_config["niah_span_loss_mode"] == "multi_positive"
    assert page_local_config["mhdsra2_config_override"] == {
        "retrieval_neighbor_span": 1,
        "retrieval_neighbor_direction": "both",
        "retrieval_neighbor_seed_multiplier": 4,
    }
    assert page_local_config["use_retrieval"] is True

    structured_feature_config = by_group["retrieval_structured_feature_span_predictor"][
        "config"
    ]
    assert structured_feature_config["retrieval_neighbor_span"] == 1
    assert structured_feature_config["retrieval_neighbor_direction"] == "both"
    assert structured_feature_config["retrieval_neighbor_seed_multiplier"] == 4
    assert structured_feature_config["retrieval_span_structure_features"] is True
    assert structured_feature_config["niah_readout_mode"] == "span_predictor"
    assert structured_feature_config["niah_span_candidate_filter"] == "prefer_key_value_pair"
    assert structured_feature_config["niah_span_loss_mode"] == "multi_positive"
    assert structured_feature_config["mhdsra2_config_override"] == {
        "retrieval_neighbor_span": 1,
        "retrieval_neighbor_direction": "both",
        "retrieval_neighbor_seed_multiplier": 4,
    }
    assert structured_feature_config["use_retrieval"] is True

    compact_config = by_group["retrieval_compact_page_local_span_predictor"]["config"]
    assert compact_config["retrieval_neighbor_span"] == 1
    assert compact_config["retrieval_neighbor_direction"] == "both"
    assert compact_config["retrieval_neighbor_seed_multiplier"] == 4
    assert compact_config["retrieval_max_tokens"] == 32
    assert compact_config["niah_readout_mode"] == "span_predictor"
    assert compact_config["niah_span_candidate_filter"] == "prefer_key_value_pair"
    assert compact_config["niah_span_loss_mode"] == "multi_positive"
    assert compact_config["mhdsra2_config_override"] == {
        "retrieval_neighbor_span": 1,
        "retrieval_neighbor_direction": "both",
        "retrieval_neighbor_seed_multiplier": 4,
        "retrieval_max_tokens": 32,
    }
    assert compact_config["use_retrieval"] is True

    pair_aware_config = by_group["retrieval_pair_aware_page_local_span_predictor"][
        "config"
    ]
    assert pair_aware_config["retrieval_neighbor_span"] == 1
    assert pair_aware_config["retrieval_neighbor_direction"] == "both"
    assert pair_aware_config["retrieval_neighbor_seed_multiplier"] == 4
    assert pair_aware_config["retrieval_neighbor_budget_mode"] == "pair_aware"
    assert pair_aware_config["retrieval_max_tokens"] == 128
    assert pair_aware_config["niah_readout_mode"] == "span_predictor"
    assert pair_aware_config["niah_span_candidate_filter"] == "prefer_key_value_pair"
    assert pair_aware_config["niah_span_loss_mode"] == "multi_positive"
    assert pair_aware_config["mhdsra2_config_override"] == {
        "retrieval_neighbor_span": 1,
        "retrieval_neighbor_direction": "both",
        "retrieval_neighbor_seed_multiplier": 4,
        "retrieval_neighbor_budget_mode": "pair_aware",
    }
    assert pair_aware_config["use_retrieval"] is True



def test_split_json_row_metrics_keeps_readout_diagnostics_out_of_selection() -> None:
    metrics = {
        "validation_generation_exact_match_rate": 0.25,
        "validation_generation_mean_sequence_accuracy": 0.5,
        "test_generation_exact_match_rate": 0.75,
        "validation_teacher_forced_exact_match_rate": 1.0,
        "validation_slot_decoder_full_answer_accuracy": 0.9,
        "test_slot_decoder_full_answer_accuracy": 0.8,
        "validation_evidence_window_accuracy": 0.7,
        "validation_extract_then_compose_mean_sequence_accuracy": 0.6,
    }

    validation, test, diagnostic = split_json_row_metrics(metrics)

    assert validation == {
        "validation_generation_exact_match_rate": 0.25,
        "validation_generation_mean_sequence_accuracy": 0.5,
    }
    assert test == {"test_generation_exact_match_rate": 0.75}
    assert diagnostic["validation_teacher_forced_exact_match_rate"] == 1.0
    assert diagnostic["validation_slot_decoder_full_answer_accuracy"] == 0.9
    assert diagnostic["test_slot_decoder_full_answer_accuracy"] == 0.8
    assert diagnostic["validation_evidence_window_accuracy"] == 0.7
    assert diagnostic["validation_extract_then_compose_mean_sequence_accuracy"] == 0.6
    assert "validation_slot_decoder_full_answer_accuracy" not in validation
    assert "test_slot_decoder_full_answer_accuracy" not in test


def test_quality_ablation_json_row_passes_slot_readout_parameters(monkeypatch) -> None:
    calls = []

    def fake_run_json_retrieval_generalization_test(**kwargs):
        calls.append(kwargs)
        return {
            "validation_pool_evaluation": {
                "generation_exact_match_rate": 0.0,
                "generation_mean_sequence_accuracy": 0.42,
                "teacher_forced_exact_match_rate": 0.0,
                "teacher_forced_mean_sequence_accuracy": 0.51,
                "slot_decoder_full_answer_accuracy": 0.25,
                "slot_decoder_museum_accuracy": 0.50,
                "slot_decoder_artifact_accuracy": 0.25,
                "slot_decoder_artist_accuracy": 0.75,
                "slot_decoder_dynasty_accuracy": 0.00,
                "evidence_window_accuracy": 0.60,
                "evidence_window_mean_distance": 1.25,
                "extract_then_compose_exact_match_rate": 0.0,
                "extract_then_compose_mean_sequence_accuracy": 0.52,
            },
            "test_pool_evaluation": {
                "generation_exact_match_rate": 0.0,
                "generation_mean_sequence_accuracy": 0.33,
                "teacher_forced_exact_match_rate": 0.0,
                "teacher_forced_mean_sequence_accuracy": 0.44,
                "slot_decoder_full_answer_accuracy": 0.20,
                "slot_decoder_museum_accuracy": 0.40,
                "slot_decoder_artifact_accuracy": 0.20,
                "slot_decoder_artist_accuracy": 0.60,
                "slot_decoder_dynasty_accuracy": 0.00,
                "evidence_window_accuracy": 0.50,
                "evidence_window_mean_distance": 2.00,
                "extract_then_compose_exact_match_rate": 0.0,
                "extract_then_compose_mean_sequence_accuracy": 0.43,
            },
        }

    monkeypatch.setattr(
        ablation,
        "run_json_retrieval_generalization_test",
        fake_run_json_retrieval_generalization_test,
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks",
            "json",
            "--groups",
            "evidence_slot_readout",
            "--json-task-seed-roots",
            "7",
            "--json-epochs",
            "3",
            "--json-train-dataset-size",
            "2",
            "--json-validation-dataset-size",
            "1",
            "--json-test-dataset-size",
            "1",
            "--json-distractor-records-per-case",
            "2",
            "--json-answer-template-mode",
            "mixed",
            "--device",
            "cpu",
        ]
    )
    planned_row = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)[0]

    row = run_json_row(args, planned_row, torch.device("cpu"))

    assert calls
    call = calls[0]
    assert call["evidence_loss_weight"] == planned_row["config"]["evidence_loss_weight"]
    assert call["slot_decoder_loss_weight"] == planned_row["config"][
        "slot_decoder_loss_weight"
    ]
    assert call["slot_decoder_logit_bias"] == planned_row["config"][
        "slot_decoder_logit_bias"
    ]
    assert call["evidence_hint_weight"] == planned_row["config"]["evidence_hint_weight"]
    assert call["generation_readout_mode"] == planned_row["config"][
        "generation_readout_mode"
    ]
    assert call["distractor_records_per_case"] == 2
    assert call["answer_template_mode"] == "mixed"
    assert planned_row["config"]["distractor_records_per_case"] == 2
    assert planned_row["config"]["answer_template_mode"] == "mixed"
    assert call["device"] == torch.device("cpu")
    assert row["status"] == "completed"
    assert row["validation_metrics"] == {
        "validation_generation_exact_match_rate": 0.0,
        "validation_generation_mean_sequence_accuracy": 0.42,
    }
    assert row["test_metrics"] == {
        "test_generation_exact_match_rate": 0.0,
        "test_generation_mean_sequence_accuracy": 0.33,
    }
    assert row["diagnostic_metrics"]["validation_slot_decoder_full_answer_accuracy"] == 0.25
    assert row["diagnostic_metrics"]["test_evidence_window_accuracy"] == 0.50


def test_quality_ablation_niah_row_passes_copy_readout_parameters(monkeypatch) -> None:
    calls = []

    def fake_run_niah_verification_case(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "final_accuracy": 0.5,
            "final_min_depth_accuracy": 0.25,
            "best_accuracy": 0.75,
            "final_eval_loss": 0.4,
            "final_train_loss": 0.6,
            "peak_memory_allocated_mb": 12.0,
            "peak_memory_reserved_mb": 16.0,
            "final_readout_mode": "needle_copy",
            "final_readout_available_rate": 0.8,
            "final_target_candidate_hit_rate": 0.25,
            "final_mean_target_candidate_rank": 3.0,
            "final_retrieval_evidence_metrics": {"available": False},
            "final_aux_diagnostics": {},
        }

    monkeypatch.setattr(
        ablation,
        "run_niah_verification_case",
        fake_run_niah_verification_case,
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks",
            "niah",
            "--groups",
            "needle_copy_readout",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
            "--device",
            "cpu",
        ]
    )
    planned_row = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)[0]

    row = ablation.run_niah_row(args, torch.device("cpu"), planned_row)

    assert calls
    call = calls[0]
    assert call["use_retrieval"] is True
    assert call["niah_readout_mode"] == "needle_copy"
    assert call["retrieval_evidence_loss_alpha"] == 0.0
    assert row["status"] == "completed"
    assert row["validation_metrics"]["final_eval_mean_accuracy"] == 0.5
    assert row["validation_metrics"]["final_readout_available_rate"] == 0.8
    assert row["validation_metrics"]["final_target_candidate_hit_rate"] == 0.25
    assert row["diagnostic_metrics"]["final_readout_mode"] == "needle_copy"
    assert row["diagnostic_metrics"]["final_mean_target_candidate_rank"] == 3.0


def test_quality_ablation_niah_row_passes_pair_copy_readout_parameters(monkeypatch) -> None:
    calls = []

    def fake_run_niah_verification_case(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "final_accuracy": 1.0,
            "final_min_depth_accuracy": 1.0,
            "best_accuracy": 1.0,
            "final_eval_loss": 0.0,
            "final_train_loss": 0.1,
            "peak_memory_allocated_mb": 12.0,
            "peak_memory_reserved_mb": 16.0,
            "final_readout_mode": "needle_pair_copy",
            "final_readout_available_rate": 1.0,
            "final_target_candidate_hit_rate": 1.0,
            "final_mean_target_candidate_rank": 2.0,
            "final_retrieval_evidence_metrics": {"available": False},
            "final_aux_diagnostics": {},
        }

    monkeypatch.setattr(
        ablation,
        "run_niah_verification_case",
        fake_run_niah_verification_case,
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks",
            "niah",
            "--groups",
            "retrieval_neighbor_span_pair_copy",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
            "--device",
            "cpu",
        ]
    )
    planned_row = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)[0]

    row = ablation.run_niah_row(args, torch.device("cpu"), planned_row)

    assert calls
    call = calls[0]
    assert call["use_retrieval"] is True
    assert call["mhdsra2_config_override"] == {"retrieval_neighbor_span": 1}
    assert call["niah_readout_mode"] == "needle_pair_copy"
    assert call["retrieval_evidence_loss_alpha"] == 0.0
    assert row["status"] == "completed"
    assert row["validation_metrics"]["final_eval_mean_accuracy"] == 1.0
    assert row["diagnostic_metrics"]["final_readout_mode"] == "needle_pair_copy"
    assert row["diagnostic_metrics"]["final_mean_target_candidate_rank"] == 2.0


def test_quality_ablation_niah_row_passes_key_evidence_target_offset(monkeypatch) -> None:
    calls = []

    def fake_run_niah_verification_case(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "final_accuracy": 0.0,
            "final_min_depth_accuracy": 0.0,
            "best_accuracy": 0.0,
            "final_eval_loss": 1.0,
            "final_train_loss": 1.0,
            "peak_memory_allocated_mb": 12.0,
            "peak_memory_reserved_mb": 16.0,
            "final_readout_mode": "needle_pair_copy",
            "final_readout_available_rate": 1.0,
            "final_target_candidate_hit_rate": 1.0,
            "final_mean_target_candidate_rank": 4.0,
            "final_retrieval_evidence_metrics": {"available": True},
            "final_aux_diagnostics": {},
        }

    monkeypatch.setattr(
        ablation,
        "run_niah_verification_case",
        fake_run_niah_verification_case,
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks",
            "niah",
            "--groups",
            "evidence_key_score_margin_neighbor_span_pair_copy",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
            "--device",
            "cpu",
        ]
    )
    planned_row = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)[0]

    row = ablation.run_niah_row(args, torch.device("cpu"), planned_row)

    assert calls
    call = calls[0]
    assert call["use_retrieval"] is True
    assert call["mhdsra2_config_override"] == {"retrieval_neighbor_span": 1}
    assert call["niah_readout_mode"] == "needle_pair_copy"
    assert call["retrieval_evidence_loss_alpha"] > 0.0
    assert call["retrieval_evidence_score_margin"] > 0.0
    assert call["retrieval_evidence_target_offset"] == 0
    assert row["config"]["retrieval_evidence_target_offset"] == 0
    assert row["diagnostic_metrics"]["final_readout_mode"] == "needle_pair_copy"


def test_quality_ablation_niah_row_passes_query_alignment_alpha(monkeypatch) -> None:
    calls = []

    def fake_run_niah_verification_case(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "final_accuracy": 0.0,
            "final_min_depth_accuracy": 0.0,
            "best_accuracy": 0.0,
            "final_eval_loss": 1.0,
            "final_train_loss": 1.0,
            "peak_memory_allocated_mb": 12.0,
            "peak_memory_reserved_mb": 16.0,
            "final_readout_mode": "needle_pair_copy",
            "final_readout_available_rate": 1.0,
            "final_target_candidate_hit_rate": 1.0,
            "final_mean_target_candidate_rank": 5.0,
            "final_retrieval_evidence_metrics": {"available": False},
            "train_query_evidence_alignment_summary": {
                "available_steps": 2,
                "mean_loss": 0.4,
                "mean_cosine": 0.6,
                "mean_mse": 0.2,
            },
            "final_aux_diagnostics": {},
        }

    monkeypatch.setattr(
        ablation,
        "run_niah_verification_case",
        fake_run_niah_verification_case,
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks",
            "niah",
            "--groups",
            "query_evidence_alignment_pair_copy",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
            "--device",
            "cpu",
        ]
    )
    planned_row = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)[0]

    row = ablation.run_niah_row(args, torch.device("cpu"), planned_row)

    assert calls
    call = calls[0]
    assert call["use_retrieval"] is True
    assert call["mhdsra2_config_override"] == {"retrieval_neighbor_span": 1}
    assert call["niah_readout_mode"] == "needle_pair_copy"
    assert call["retrieval_evidence_loss_alpha"] == 0.0
    assert call["query_evidence_alignment_alpha"] == planned_row["config"][
        "query_evidence_alignment_alpha"
    ]
    assert call["query_evidence_alignment_alpha"] > 0.0
    assert row["diagnostic_metrics"]["query_evidence_alignment_mean_loss"] == 0.4
    assert row["diagnostic_metrics"]["query_evidence_alignment_mean_cosine"] == 0.6
    assert row["diagnostic_metrics"]["query_evidence_alignment_mean_mse"] == 0.2


def test_quality_ablation_niah_row_passes_retrieval_projection_alpha(monkeypatch) -> None:
    calls = []

    def fake_run_niah_verification_case(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "final_accuracy": 0.0,
            "final_min_depth_accuracy": 0.0,
            "best_accuracy": 0.0,
            "final_eval_loss": 1.0,
            "final_train_loss": 1.0,
            "peak_memory_allocated_mb": 12.0,
            "peak_memory_reserved_mb": 16.0,
            "final_readout_mode": "needle_pair_copy",
            "final_readout_available_rate": 1.0,
            "final_target_candidate_hit_rate": 1.0,
            "final_mean_target_candidate_rank": 4.0,
            "final_retrieval_evidence_metrics": {"available": False},
            "final_retrieval_projection_metrics": {
                "available": True,
                "hit_rate": 1.0,
                "target_rank_mean": 2.0,
                "top1_rate": 0.0,
            },
            "train_retrieval_projection_summary": {
                "available_steps": 3,
                "mean_loss": 0.7,
                "mean_target_rank": 2.0,
                "mean_top1_rate": 0.25,
            },
            "final_aux_diagnostics": {},
        }

    monkeypatch.setattr(
        ablation,
        "run_niah_verification_case",
        fake_run_niah_verification_case,
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks",
            "niah",
            "--groups",
            "retrieval_projection_pair_copy",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
            "--device",
            "cpu",
        ]
    )
    planned_row = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)[0]

    row = ablation.run_niah_row(args, torch.device("cpu"), planned_row)

    assert calls
    call = calls[0]
    assert call["use_retrieval"] is True
    assert call["mhdsra2_config_override"] == {"retrieval_neighbor_span": 1}
    assert call["niah_readout_mode"] == "needle_pair_copy"
    assert call["retrieval_evidence_loss_alpha"] == 0.0
    assert call["retrieval_projection_contrastive_alpha"] == planned_row["config"][
        "retrieval_projection_contrastive_alpha"
    ]
    assert call["retrieval_projection_temperature"] == planned_row["config"][
        "retrieval_projection_temperature"
    ]
    assert row["diagnostic_metrics"]["retrieval_projection_mean_loss"] == 0.7
    assert row["diagnostic_metrics"]["retrieval_projection_mean_target_rank"] == 2.0
    assert row["diagnostic_metrics"]["retrieval_projection_mean_top1_rate"] == 0.25
    assert row["metrics"]["final_retrieval_projection_metrics"]["available"] is True


def test_quality_ablation_niah_row_passes_span_predictor_alpha(monkeypatch) -> None:
    calls = []

    def fake_run_niah_verification_case(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "final_accuracy": 0.25,
            "final_min_depth_accuracy": 0.0,
            "best_accuracy": 0.25,
            "final_eval_loss": 0.75,
            "final_train_loss": 0.9,
            "peak_memory_allocated_mb": 12.0,
            "peak_memory_reserved_mb": 16.0,
            "final_readout_mode": "span_predictor",
            "final_readout_available_rate": 1.0,
            "final_target_candidate_hit_rate": 0.5,
            "final_mean_target_candidate_rank": 3.0,
            "final_retrieval_evidence_metrics": {"available": False},
            "final_retrieval_span_predictor_metrics": {
                "available": True,
                "hit_rate": 1.0,
                "target_rank_mean": 1.5,
                "top1_rate": 0.5,
            },
            "train_retrieval_span_predictor_summary": {
                "available_steps": 3,
                "mean_loss": 0.6,
                "mean_target_rank": 1.5,
                "mean_top1_rate": 0.5,
                "mean_logit_margin": 0.25,
            },
            "final_aux_diagnostics": {},
        }

    monkeypatch.setattr(
        ablation,
        "run_niah_verification_case",
        fake_run_niah_verification_case,
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks",
            "niah",
            "--groups",
            "retrieval_span_predictor",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
            "--device",
            "cpu",
        ]
    )
    planned_row = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)[0]

    row = ablation.run_niah_row(args, torch.device("cpu"), planned_row)

    assert calls
    call = calls[0]
    assert call["use_retrieval"] is True
    assert call["mhdsra2_config_override"] == {"retrieval_neighbor_span": 1}
    assert call["niah_readout_mode"] == "span_predictor"
    assert call["retrieval_span_predictor_alpha"] == planned_row["config"][
        "retrieval_span_predictor_alpha"
    ]
    assert call["retrieval_span_predictor_alpha"] > 0.0
    assert call["retrieval_evidence_loss_alpha"] == 0.0
    assert row["diagnostic_metrics"]["retrieval_span_predictor_mean_loss"] == 0.6
    assert row["diagnostic_metrics"]["retrieval_span_predictor_mean_target_rank"] == 1.5
    assert row["diagnostic_metrics"]["retrieval_span_predictor_mean_top1_rate"] == 0.5
    assert row["diagnostic_metrics"]["retrieval_span_predictor_mean_logit_margin"] == 0.25
    assert row["metrics"]["final_retrieval_span_predictor_metrics"]["available"] is True


def test_quality_ablation_niah_row_passes_span_structure_features(monkeypatch) -> None:
    calls = []

    def fake_run_niah_verification_case(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "final_accuracy": 0.3333333333333333,
            "final_min_depth_accuracy": 0.0,
            "best_accuracy": 0.3333333333333333,
            "final_eval_loss": 0.6666666666666666,
            "final_train_loss": 1.0,
            "peak_memory_allocated_mb": 12.0,
            "peak_memory_reserved_mb": 16.0,
            "final_readout_mode": "span_predictor",
            "final_readout_available_rate": 1.0,
            "final_target_candidate_hit_rate": 1.0,
            "final_mean_target_candidate_rank": 2.0,
            "final_retrieval_evidence_metrics": {"available": False},
            "final_retrieval_span_predictor_metrics": {
                "available": True,
                "hit_rate": 1.0,
                "target_rank_mean": 1.0,
                "top1_rate": 1.0,
            },
            "train_retrieval_span_predictor_summary": {
                "available_steps": 3,
                "mean_loss": 0.5,
                "mean_target_rank": 1.0,
                "mean_top1_rate": 1.0,
                "mean_logit_margin": 0.3,
            },
            "final_aux_diagnostics": {},
            "test_accuracy": 0.25,
            "test_min_depth_accuracy": 0.0,
            "test_eval_loss": 0.8,
            "test_readout_available_rate": 0.75,
            "test_target_candidate_hit_rate": 0.5,
            "test_mean_target_candidate_rank": 4.0,
            "test_span_candidate_diagnostics": {
                "mean_span_valid_candidate_count": 9.0,
                "span_pair_available_rate": 0.25,
            },
        }

    monkeypatch.setattr(
        ablation,
        "run_niah_verification_case",
        fake_run_niah_verification_case,
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks",
            "niah",
            "--groups",
            "retrieval_structured_feature_span_predictor",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
            "--device",
            "cpu",
            "--niah-test-batches-per-depth",
            "2",
        ]
    )
    planned_row = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)[0]

    row = ablation.run_niah_row(args, torch.device("cpu"), planned_row)

    assert calls
    call = calls[0]
    assert call["use_retrieval"] is True
    assert call["retrieval_span_predictor_alpha"] == planned_row["config"][
        "retrieval_span_predictor_alpha"
    ]
    assert call["retrieval_span_structure_features"] is True
    assert call["niah_readout_mode"] == "span_predictor"
    assert call["niah_span_candidate_filter"] == "prefer_key_value_pair"
    assert call["niah_span_loss_mode"] == "multi_positive"
    assert call["niah_test_batches_per_depth"] == 2
    assert call["mhdsra2_config_override"] == {
        "retrieval_neighbor_span": 1,
        "retrieval_neighbor_direction": "both",
        "retrieval_neighbor_seed_multiplier": 4,
    }
    assert row["config"]["retrieval_span_structure_features"] is True
    assert row["config"]["test_batches_per_depth"] == 2
    assert row["test_metrics"] == {
        "test_eval_mean_accuracy": 0.25,
        "test_eval_min_depth_accuracy": 0.0,
        "test_eval_loss": 0.8,
        "test_readout_available_rate": 0.75,
        "test_target_candidate_hit_rate": 0.5,
    }
    assert "test_eval_mean_accuracy" not in row["validation_metrics"]
    assert "test_mean_span_valid_candidate_count" not in row["validation_metrics"]
    assert row["diagnostic_metrics"]["retrieval_span_predictor_mean_top1_rate"] == 1.0
    assert row["diagnostic_metrics"]["test_mean_target_candidate_rank"] == 4.0
    assert row["diagnostic_metrics"]["test_mean_span_valid_candidate_count"] == 9.0
    assert row["diagnostic_metrics"]["test_span_pair_available_rate"] == 0.25
    assert row["metrics"]["test_accuracy"] == 0.25
    assert row["metrics"]["test_span_candidate_diagnostics"][
        "mean_span_valid_candidate_count"
    ] == 9.0


def test_quality_ablation_niah_row_passes_structured_span_candidate_filter(
    monkeypatch,
) -> None:
    calls = []

    def fake_run_niah_verification_case(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "final_accuracy": 0.25,
            "final_min_depth_accuracy": 0.0,
            "best_accuracy": 0.25,
            "final_eval_loss": 0.75,
            "final_train_loss": 0.9,
            "peak_memory_allocated_mb": 12.0,
            "peak_memory_reserved_mb": 16.0,
            "final_readout_mode": "span_predictor",
            "final_readout_available_rate": 1.0,
            "final_target_candidate_hit_rate": 0.5,
            "final_mean_target_candidate_rank": 3.0,
            "final_retrieval_evidence_metrics": {"available": False},
            "final_retrieval_span_predictor_metrics": {
                "available": True,
                "candidate_filter": "key_value_pair",
            },
            "train_retrieval_span_predictor_summary": {
                "available_steps": 1,
                "mean_loss": 0.4,
            },
            "final_aux_diagnostics": {},
        }

    monkeypatch.setattr(
        ablation,
        "run_niah_verification_case",
        fake_run_niah_verification_case,
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks",
            "niah",
            "--groups",
            "retrieval_structured_span_predictor",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
            "--device",
            "cpu",
        ]
    )
    planned_row = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)[0]

    row = ablation.run_niah_row(args, torch.device("cpu"), planned_row)

    assert calls
    call = calls[0]
    assert call["use_retrieval"] is True
    assert call["niah_readout_mode"] == "span_predictor"
    assert call["niah_span_candidate_filter"] == "key_value_pair"
    assert planned_row["config"]["niah_span_candidate_filter"] == "key_value_pair"
    assert row["metrics"]["final_retrieval_span_predictor_metrics"][
        "candidate_filter"
    ] == "key_value_pair"


def test_quality_ablation_niah_row_passes_multi_positive_span_loss_mode(
    monkeypatch,
) -> None:
    calls = []

    def fake_run_niah_verification_case(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "final_accuracy": 0.25,
            "final_min_depth_accuracy": 0.0,
            "best_accuracy": 0.25,
            "final_eval_loss": 0.75,
            "final_train_loss": 0.9,
            "peak_memory_allocated_mb": 12.0,
            "peak_memory_reserved_mb": 16.0,
            "final_readout_mode": "span_predictor",
            "final_readout_available_rate": 1.0,
            "final_target_candidate_hit_rate": 0.5,
            "final_mean_target_candidate_rank": 3.0,
            "final_retrieval_evidence_metrics": {"available": False},
            "final_retrieval_span_predictor_metrics": {
                "available": True,
                "candidate_filter": "prefer_key_value_pair",
                "loss_mode": "multi_positive",
            },
            "train_retrieval_span_predictor_summary": {
                "available_steps": 1,
                "mean_loss": 0.4,
            },
            "final_aux_diagnostics": {},
        }

    monkeypatch.setattr(
        ablation,
        "run_niah_verification_case",
        fake_run_niah_verification_case,
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks",
            "niah",
            "--groups",
            "retrieval_multi_positive_span_predictor",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
            "--device",
            "cpu",
        ]
    )
    planned_row = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)[0]

    row = ablation.run_niah_row(args, torch.device("cpu"), planned_row)

    assert calls
    call = calls[0]
    assert call["use_retrieval"] is True
    assert call["niah_readout_mode"] == "span_predictor"
    assert call["niah_span_candidate_filter"] == "prefer_key_value_pair"
    assert call["niah_span_loss_mode"] == "multi_positive"
    assert planned_row["config"]["niah_span_loss_mode"] == "multi_positive"
    assert row["metrics"]["final_retrieval_span_predictor_metrics"][
        "loss_mode"
    ] == "multi_positive"


def test_quality_ablation_niah_row_preserves_span_candidate_diagnostics(
    monkeypatch,
) -> None:
    diagnostics = {
        "mean_span_valid_candidate_count": 8.0,
        "mean_span_raw_candidate_count": 16.0,
        "mean_span_pair_candidate_count": 2.0,
        "span_pair_available_rate": 1.0,
        "span_filter_fallback_rate": 0.25,
        "span_selected_pair_rate": 0.5,
        "span_target_pair_candidate_rate": 0.75,
        "span_target_value_page_candidate_rate": 0.9,
        "span_target_pair_page_candidate_rate": 0.8,
        "span_target_value_top_token_rate": 0.7,
        "span_target_pair_top_token_rate": 0.6,
        "span_target_value_seed_token_rate": 0.55,
        "span_target_pair_seed_token_rate": 0.45,
    }

    def fake_run_niah_verification_case(**kwargs):
        return {
            "status": "completed",
            "final_accuracy": 0.25,
            "final_min_depth_accuracy": 0.0,
            "best_accuracy": 0.25,
            "final_eval_loss": 0.75,
            "final_train_loss": 0.9,
            "peak_memory_allocated_mb": 12.0,
            "peak_memory_reserved_mb": 16.0,
            "final_readout_mode": "span_predictor",
            "final_readout_available_rate": 1.0,
            "final_target_candidate_hit_rate": 0.5,
            "final_mean_target_candidate_rank": 3.0,
            "final_span_candidate_diagnostics": diagnostics,
            "final_retrieval_evidence_metrics": {"available": False},
            "final_retrieval_span_predictor_metrics": {
                "available": True,
                "candidate_filter": "prefer_key_value_pair",
            },
            "train_retrieval_span_predictor_summary": {
                "available_steps": 1,
                "mean_loss": 0.4,
            },
            "final_aux_diagnostics": {},
        }

    monkeypatch.setattr(
        ablation,
        "run_niah_verification_case",
        fake_run_niah_verification_case,
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks",
            "niah",
            "--groups",
            "retrieval_multi_positive_span_predictor",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
            "--device",
            "cpu",
        ]
    )
    planned_row = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)[0]

    row = ablation.run_niah_row(args, torch.device("cpu"), planned_row)

    assert "final_mean_span_valid_candidate_count" not in row["validation_metrics"]
    assert "final_span_candidate_diagnostics" not in row["validation_metrics"]
    assert row["diagnostic_metrics"]["final_span_candidate_diagnostics"] == diagnostics
    assert row["diagnostic_metrics"][
        "final_mean_span_valid_candidate_count"
    ] == pytest.approx(8.0)
    assert row["diagnostic_metrics"][
        "final_mean_span_raw_candidate_count"
    ] == pytest.approx(16.0)
    assert row["diagnostic_metrics"][
        "final_mean_span_pair_candidate_count"
    ] == pytest.approx(2.0)
    assert row["diagnostic_metrics"]["final_span_pair_available_rate"] == pytest.approx(
        1.0
    )
    assert row["diagnostic_metrics"]["final_span_filter_fallback_rate"] == pytest.approx(
        0.25
    )
    assert row["diagnostic_metrics"]["final_span_selected_pair_rate"] == pytest.approx(
        0.5
    )
    assert row["diagnostic_metrics"][
        "final_span_target_pair_candidate_rate"
    ] == pytest.approx(0.75)
    assert row["diagnostic_metrics"][
        "final_span_target_value_page_candidate_rate"
    ] == pytest.approx(0.9)
    assert row["diagnostic_metrics"][
        "final_span_target_pair_page_candidate_rate"
    ] == pytest.approx(0.8)
    assert row["diagnostic_metrics"][
        "final_span_target_value_top_token_rate"
    ] == pytest.approx(0.7)
    assert row["diagnostic_metrics"][
        "final_span_target_pair_top_token_rate"
    ] == pytest.approx(0.6)
    assert row["diagnostic_metrics"][
        "final_span_target_value_seed_token_rate"
    ] == pytest.approx(0.55)
    assert row["diagnostic_metrics"][
        "final_span_target_pair_seed_token_rate"
    ] == pytest.approx(0.45)
    assert row["metrics"]["final_span_candidate_diagnostics"] == diagnostics


def test_quality_ablation_niah_row_passes_bidirectional_neighbor_direction(
    monkeypatch,
) -> None:
    calls = []

    def fake_run_niah_verification_case(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "final_accuracy": 0.25,
            "final_min_depth_accuracy": 0.0,
            "best_accuracy": 0.25,
            "final_eval_loss": 0.75,
            "final_train_loss": 0.9,
            "peak_memory_allocated_mb": 12.0,
            "peak_memory_reserved_mb": 16.0,
            "final_readout_mode": "span_predictor",
            "final_readout_available_rate": 1.0,
            "final_target_candidate_hit_rate": 0.5,
            "final_mean_target_candidate_rank": 3.0,
            "final_span_candidate_diagnostics": {
                "mean_span_pair_candidate_count": 2.0,
            },
            "final_retrieval_evidence_metrics": {"available": False},
            "final_retrieval_span_predictor_metrics": {
                "available": True,
                "candidate_filter": "prefer_key_value_pair",
                "loss_mode": "multi_positive",
            },
            "train_retrieval_span_predictor_summary": {
                "available_steps": 1,
                "mean_loss": 0.4,
            },
            "final_aux_diagnostics": {},
        }

    monkeypatch.setattr(
        ablation,
        "run_niah_verification_case",
        fake_run_niah_verification_case,
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks",
            "niah",
            "--groups",
            "retrieval_bidirectional_structured_span_predictor",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
            "--device",
            "cpu",
        ]
    )
    planned_row = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)[0]

    row = ablation.run_niah_row(args, torch.device("cpu"), planned_row)

    assert calls
    call = calls[0]
    assert call["use_retrieval"] is True
    assert call["niah_readout_mode"] == "span_predictor"
    assert call["niah_span_candidate_filter"] == "prefer_key_value_pair"
    assert call["niah_span_loss_mode"] == "multi_positive"
    assert call["mhdsra2_config_override"] == {
        "retrieval_neighbor_span": 1,
        "retrieval_neighbor_direction": "both",
    }
    assert planned_row["config"]["retrieval_neighbor_direction"] == "both"
    assert row["config"]["retrieval_neighbor_direction"] == "both"


def test_quality_ablation_niah_row_passes_neighbor_seed_multiplier(
    monkeypatch,
) -> None:
    calls = []

    def fake_run_niah_verification_case(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "final_accuracy": 0.25,
            "final_min_depth_accuracy": 0.0,
            "best_accuracy": 0.25,
            "final_eval_loss": 0.75,
            "final_train_loss": 0.9,
            "peak_memory_allocated_mb": 12.0,
            "peak_memory_reserved_mb": 16.0,
            "final_readout_mode": "span_predictor",
            "final_readout_available_rate": 1.0,
            "final_target_candidate_hit_rate": 0.5,
            "final_mean_target_candidate_rank": 3.0,
            "final_span_candidate_diagnostics": {
                "span_target_pair_seed_token_rate": 1.0,
            },
            "final_retrieval_evidence_metrics": {"available": False},
            "final_retrieval_span_predictor_metrics": {
                "available": True,
                "candidate_filter": "prefer_key_value_pair",
                "loss_mode": "multi_positive",
            },
            "train_retrieval_span_predictor_summary": {
                "available_steps": 1,
                "mean_loss": 0.4,
            },
            "final_aux_diagnostics": {},
        }

    monkeypatch.setattr(
        ablation,
        "run_niah_verification_case",
        fake_run_niah_verification_case,
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks",
            "niah",
            "--groups",
            "retrieval_page_local_neighbor_span_predictor",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
            "--device",
            "cpu",
        ]
    )
    planned_row = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)[0]

    row = ablation.run_niah_row(args, torch.device("cpu"), planned_row)

    assert calls
    call = calls[0]
    assert call["use_retrieval"] is True
    assert call["mhdsra2_config_override"] == {
        "retrieval_neighbor_span": 1,
        "retrieval_neighbor_direction": "both",
        "retrieval_neighbor_seed_multiplier": 4,
    }
    assert planned_row["config"]["retrieval_neighbor_seed_multiplier"] == 4
    assert row["config"]["retrieval_neighbor_seed_multiplier"] == 4
    assert row["diagnostic_metrics"][
        "final_span_target_pair_seed_token_rate"
    ] == pytest.approx(1.0)


def test_quality_ablation_niah_row_passes_compact_retrieval_budget(
    monkeypatch,
) -> None:
    calls = []

    def fake_run_niah_verification_case(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "final_accuracy": 0.25,
            "final_min_depth_accuracy": 0.0,
            "best_accuracy": 0.25,
            "final_eval_loss": 0.75,
            "final_train_loss": 0.9,
            "peak_memory_allocated_mb": 12.0,
            "peak_memory_reserved_mb": 16.0,
            "final_readout_mode": "span_predictor",
            "final_readout_available_rate": 1.0,
            "final_target_candidate_hit_rate": 0.5,
            "final_mean_target_candidate_rank": 3.0,
            "final_span_candidate_diagnostics": {
                "mean_span_valid_candidate_count": 32.0,
            },
            "final_retrieval_evidence_metrics": {"available": False},
            "final_retrieval_span_predictor_metrics": {
                "available": True,
                "candidate_filter": "prefer_key_value_pair",
                "loss_mode": "multi_positive",
            },
            "train_retrieval_span_predictor_summary": {
                "available_steps": 1,
                "mean_loss": 0.4,
            },
            "final_aux_diagnostics": {},
        }

    monkeypatch.setattr(
        ablation,
        "run_niah_verification_case",
        fake_run_niah_verification_case,
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks",
            "niah",
            "--groups",
            "retrieval_compact_page_local_span_predictor",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
            "--device",
            "cpu",
        ]
    )
    planned_row = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)[0]

    row = ablation.run_niah_row(args, torch.device("cpu"), planned_row)

    assert calls
    call = calls[0]
    assert call["use_retrieval"] is True
    assert call["mhdsra2_config_override"] == {
        "retrieval_neighbor_span": 1,
        "retrieval_neighbor_direction": "both",
        "retrieval_neighbor_seed_multiplier": 4,
        "retrieval_max_tokens": 32,
    }
    assert planned_row["config"]["retrieval_max_tokens"] == 32
    assert row["config"]["retrieval_max_tokens"] == 32
    assert row["diagnostic_metrics"][
        "final_mean_span_valid_candidate_count"
    ] == pytest.approx(32.0)


def test_mhdsra2_config_rejects_non_positive_neighbor_seed_multiplier() -> None:
    with pytest.raises(ValueError, match="retrieval_neighbor_seed_multiplier"):
        MHDSRA2Config(
            dim=16,
            heads=2,
            retrieval_neighbor_span=1,
            retrieval_neighbor_seed_multiplier=0,
        )


def test_mhdsra2_config_rejects_non_positive_retrieval_max_tokens() -> None:
    with pytest.raises(ValueError, match="retrieval_max_tokens"):
        MHDSRA2Config(
            dim=16,
            heads=2,
            retrieval_max_tokens=0,
        )


def test_mhdsra2_config_rejects_unknown_neighbor_budget_mode() -> None:
    with pytest.raises(ValueError, match="retrieval_neighbor_budget_mode"):
        MHDSRA2Config(
            dim=16,
            heads=2,
            retrieval_neighbor_budget_mode="bad",
        )


def test_multilayer_repository_uses_retrieval_max_tokens_override() -> None:
    default_model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=4,
        kr=2,
        use_retrieval=True,
    )
    default_repositories = default_model._new_retrieval_repositories()

    assert default_repositories[0].max_tokens == 128

    compact_model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=4,
        kr=2,
        use_retrieval=True,
        mhdsra2_config_override={"retrieval_max_tokens": 32},
    )
    compact_repositories = compact_model._new_retrieval_repositories()

    assert compact_repositories[0].max_tokens == 32


def test_multilayer_repository_uses_neighbor_budget_mode_override() -> None:
    default_model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=4,
        kr=2,
        use_retrieval=True,
    )
    default_repositories = default_model._new_retrieval_repositories()

    assert default_repositories[0].neighbor_budget_mode == "unbounded"

    pair_aware_model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=4,
        kr=2,
        use_retrieval=True,
        mhdsra2_config_override={
            "retrieval_neighbor_budget_mode": "pair_aware",
        },
    )
    pair_aware_repositories = pair_aware_model._new_retrieval_repositories()

    assert pair_aware_repositories[0].neighbor_budget_mode == "pair_aware"

def test_quality_ablation_niah_row_passes_rank_margin_parameter(monkeypatch) -> None:
    calls = []

    def fake_run_niah_verification_case(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "final_accuracy": 0.0,
            "final_min_depth_accuracy": 0.0,
            "best_accuracy": 0.0,
            "final_eval_loss": 1.0,
            "final_train_loss": 1.2,
            "peak_memory_allocated_mb": 0.0,
            "peak_memory_reserved_mb": 0.0,
            "final_readout_mode": "model",
            "final_readout_available_rate": None,
            "final_target_candidate_hit_rate": None,
            "final_mean_target_candidate_rank": None,
            "final_retrieval_evidence_metrics": {
                "available": True,
                "hit_rate": 1.0,
                "evidence_weight_mean": 0.25,
                "best_negative_weight_mean": 0.50,
                "evidence_margin_mean": -0.25,
                "target_rank_mean": 2.0,
                "top1_rate": 0.0,
            },
            "final_aux_diagnostics": {},
        }

    monkeypatch.setattr(
        ablation,
        "run_niah_verification_case",
        fake_run_niah_verification_case,
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks",
            "niah",
            "--groups",
            "evidence_rank_margin",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
            "--device",
            "cpu",
        ]
    )
    planned_row = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)[0]

    row = ablation.run_niah_row(args, torch.device("cpu"), planned_row)

    assert calls
    call = calls[0]
    assert call["use_retrieval"] is True
    assert call["retrieval_evidence_loss_alpha"] > 0.0
    assert call["retrieval_evidence_rank_margin"] == planned_row["config"][
        "retrieval_evidence_rank_margin"
    ]
    assert row["status"] == "completed"
    assert row["diagnostic_metrics"]["retrieval_evidence_margin_mean"] == -0.25
    assert row["diagnostic_metrics"]["retrieval_evidence_target_rank_mean"] == 2.0
    assert row["diagnostic_metrics"]["retrieval_evidence_top1_rate"] == 0.0


def test_quality_ablation_niah_row_passes_score_margin_parameter(monkeypatch) -> None:
    calls = []

    def fake_run_niah_verification_case(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "final_accuracy": 0.0,
            "final_min_depth_accuracy": 0.0,
            "best_accuracy": 0.0,
            "final_eval_loss": 1.0,
            "final_train_loss": 1.2,
            "peak_memory_allocated_mb": 0.0,
            "peak_memory_reserved_mb": 0.0,
            "final_readout_mode": "model",
            "final_readout_available_rate": None,
            "final_target_candidate_hit_rate": None,
            "final_mean_target_candidate_rank": None,
            "final_retrieval_evidence_metrics": {
                "available": True,
                "hit_rate": 1.0,
                "evidence_weight_mean": 0.25,
                "best_negative_weight_mean": 0.50,
                "evidence_margin_mean": -0.25,
                "target_rank_mean": 2.0,
                "top1_rate": 0.0,
                "score_margin_loss": 0.70,
                "evidence_score_mean": 0.10,
                "best_negative_score_mean": 0.30,
                "evidence_score_margin_mean": -0.20,
                "score_target_rank_mean": 2.0,
                "score_top1_rate": 0.0,
            },
            "final_aux_diagnostics": {},
        }

    monkeypatch.setattr(
        ablation,
        "run_niah_verification_case",
        fake_run_niah_verification_case,
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks",
            "niah",
            "--groups",
            "evidence_score_margin",
            "--niah-seq-lengths",
            "256",
            "--niah-seeds",
            "101",
            "--device",
            "cpu",
        ]
    )
    planned_row = build_run_rows(groups=args.groups, tasks=args.tasks, args=args)[0]

    row = ablation.run_niah_row(args, torch.device("cpu"), planned_row)

    assert calls
    call = calls[0]
    assert call["use_retrieval"] is True
    assert call["retrieval_evidence_loss_alpha"] > 0.0
    assert call["retrieval_evidence_rank_margin"] == 0.0
    assert call["retrieval_evidence_score_margin"] == planned_row["config"][
        "retrieval_evidence_score_margin"
    ]
    assert row["status"] == "completed"
    assert row["diagnostic_metrics"]["retrieval_evidence_score_margin_loss"] == 0.70
    assert row["diagnostic_metrics"]["retrieval_evidence_score_mean"] == 0.10
    assert row["diagnostic_metrics"]["retrieval_evidence_best_negative_score_mean"] == 0.30
    assert row["diagnostic_metrics"]["retrieval_evidence_score_margin_mean"] == -0.20
    assert row["diagnostic_metrics"]["retrieval_evidence_score_target_rank_mean"] == 2.0
    assert row["diagnostic_metrics"]["retrieval_evidence_score_top1_rate"] == 0.0


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


def test_json_extract_compose_readout_mode_reuses_existing_readout(monkeypatch) -> None:
    case = {
        "sample_bytes": b"context",
        "question_bytes": b"question",
        "expected_answer_bytes": b"ANSWER",
        "metadata": {
            "museum": "Palace Museum",
            "artifact": "Autumn Lantern Procession",
            "artist": "Lin Qiao",
            "dynasty": "Tang",
            "expected_answer_text": "ANSWER",
        },
    }
    hidden_states = torch.zeros(1, 1, 4)

    monkeypatch.setattr(
        json_task,
        "evaluate_extract_then_compose",
        lambda *_args, **_kwargs: {
            "available": True,
            "predicted_text": "ANSWER",
            "exact_byte_match": True,
        },
    )

    metrics = json_task.evaluate_generation(
        model=object(),
        case=case,
        device=torch.device("cpu"),
        hidden_states=hidden_states,
        generation_readout_mode="extract_then_compose",
    )

    assert metrics["readout_mode"] == "extract_then_compose"
    assert metrics["readout_available"] is True
    assert metrics["exact_byte_match"] is True
    assert metrics["sequence_accuracy"] == pytest.approx(1.0)


def test_json_extract_compose_prefers_complete_target_record() -> None:
    target_record = (
        b"The most valuable exhibit in the Palace Museum is Autumn Lantern Procession "
        b"painted by Lin Qiao of the Tang dynasty."
    )
    distractor_record = (
        b"The most valuable exhibit in the Grand Archive Museum is "
        b"Along the River During the Qingming Festival painted by Zhang Zeduan "
        b"of the Northern Song dynasty."
    )
    window_bytes = distractor_record + b" " + target_record

    extracted = json_task.extract_slot_labels_from_window_bytes(
        window_bytes,
        target_museum="Palace Museum",
    )

    assert extracted["museum"] == "Palace Museum"
    assert extracted["artifact"] == "Autumn Lantern Procession"
    assert extracted["artist"] == "Lin Qiao"
    assert extracted["dynasty"] == "Tang"
    assert extracted["answer_template_id"] == 0


def test_json_extract_compose_falls_back_to_window_label_lookup() -> None:
    window_bytes = (
        b"catalog card Palace Museum shelf note Autumn Lantern Procession "
        b"artist Lin Qiao dynasty Tang"
    )

    extracted = json_task.extract_slot_labels_from_window_bytes(
        window_bytes,
        target_museum="Palace Museum",
    )

    assert extracted["museum"] == "Palace Museum"
    assert extracted["artifact"] == "Autumn Lantern Procession"
    assert extracted["artist"] == "Lin Qiao"
    assert extracted["dynasty"] == "Tang"


def test_json_extract_compose_parses_answer_template_variants() -> None:
    metadata = {
        "museum": "Palace Museum",
        "artifact": "Autumn Lantern Procession",
        "artist": "Lin Qiao",
        "dynasty": "Tang",
    }

    for template_id, template in enumerate(json_task.ANSWER_TEMPLATE_VARIANTS):
        answer_text = template.format(**metadata)
        extracted = json_task.extract_slot_labels_from_window_bytes(
            answer_text.encode("ascii"),
            target_museum="Palace Museum",
        )
        predicted = json_task.build_extract_compose_prediction_bytes(extracted)

        assert extracted["museum"] == metadata["museum"]
        assert extracted["artifact"] == metadata["artifact"]
        assert extracted["artist"] == metadata["artist"]
        assert extracted["dynasty"] == metadata["dynasty"]
        assert extracted["answer_template_id"] == template_id
        assert predicted == answer_text.encode("ascii")


def test_json_extract_compose_template_parser_does_not_cross_sentence_boundary() -> None:
    metadata = {
        "museum": "Palace Museum",
        "artifact": "Autumn Lantern Procession",
        "artist": "Lin Qiao",
        "dynasty": "Tang",
    }
    answer_text = json_task.ANSWER_TEMPLATE_VARIANTS[2].format(**metadata)
    window_text = (
        "A noisy card, painted by Someone Else of the Han dynasty, is irrelevant. "
        + answer_text
    )

    extracted = json_task.extract_slot_labels_from_window_bytes(
        window_text.encode("ascii"),
        target_museum="Palace Museum",
    )

    assert extracted["artifact"] == metadata["artifact"]
    assert extracted["artist"] == metadata["artist"]
    assert extracted["dynasty"] == metadata["dynasty"]
    assert extracted["museum"] == metadata["museum"]
    assert not extracted["artifact"].startswith("A noisy card")


def test_json_extract_compose_expands_cut_window_to_sentence_boundaries() -> None:
    metadata = {
        "museum": "Palace Museum",
        "artifact": "Autumn Lantern Procession",
        "artist": "Lin Qiao",
        "dynasty": "Tang",
    }
    answer_text = json_task.ANSWER_TEMPLATE_VARIANTS[2].format(**metadata)
    sample_bytes = (
        b"catalog note. "
        + answer_text.encode("ascii")
        + b" Archive footer says unrelated field names."
    )
    raw_window_start = sample_bytes.index(b"painted by")
    raw_window_end = sample_bytes.index(b"is the most valuable") + len(b"is the most valuable")

    raw_extracted = json_task.extract_slot_labels_from_window_bytes(
        sample_bytes[raw_window_start:raw_window_end],
        target_museum="Palace Museum",
    )
    expanded_start, expanded_end = json_task.expand_window_to_sentence_boundaries(
        sample_bytes,
        raw_window_start,
        raw_window_end,
        max_expand_bytes=256,
    )
    expanded_extracted = json_task.extract_slot_labels_from_window_bytes(
        sample_bytes[expanded_start:expanded_end],
        target_museum="Palace Museum",
    )

    assert "museum" not in raw_extracted
    assert expanded_start == sample_bytes.index(answer_text.encode("ascii"))
    assert expanded_end == expanded_start + len(answer_text.encode("ascii"))
    assert expanded_extracted["museum"] == metadata["museum"]
    assert expanded_extracted["artifact"] == metadata["artifact"]
    assert expanded_extracted["artist"] == metadata["artist"]
    assert expanded_extracted["dynasty"] == metadata["dynasty"]
    assert expanded_extracted["answer_template_id"] == 2


def test_json_generator_inserts_distractor_answer_records() -> None:
    reference_case = json_task.load_json_retrieval_case()
    allowed_pairs = (
        ("Palace Museum", "Autumn Lantern Procession"),
        ("Grand Archive Museum", "Golden Crane Panorama"),
        ("Riverfront Gallery", "Jade Mountain Chronicle"),
    )
    generated = json_task.generate_random_json_retrieval_case(
        reference_case,
        rng=json_task.random.Random(20260611),
        target_total_bytes=2048,
        allowed_museum_artifact_pairs=allowed_pairs,
        forced_museum_artifact_pair=("Palace Museum", "Autumn Lantern Procession"),
        distractor_records_per_case=2,
    )

    metadata = generated["metadata"]
    distractor_records = metadata["distractor_records"]
    sample_bytes = generated["sample_bytes"]

    assert metadata["distractor_records_per_case"] == 2
    assert metadata["requested_distractor_records_per_case"] == 2
    assert len(distractor_records) == 2
    assert generated["expected_answer_bytes"] in sample_bytes
    assert all(record.encode("ascii") in sample_bytes for record in distractor_records)
    assert all("Palace Museum" not in record for record in distractor_records)
    assert all(
        any(
            museum in record and artifact in record
            for museum, artifact in allowed_pairs
            if museum != "Palace Museum"
        )
        for record in distractor_records
    )

    distractor_bytes = (" ".join(distractor_records) + " ").encode("ascii")
    assert metadata["answer_bytes"] == len(generated["expected_answer_bytes"])
    assert metadata["needle_bytes"] == metadata["answer_bytes"] + len(distractor_bytes)
    assert metadata["actual_total_bytes"] == len(sample_bytes)


def test_json_generator_mixed_templates_are_recorded_and_parseable() -> None:
    reference_case = json_task.load_json_retrieval_case()
    allowed_pairs = (
        ("Palace Museum", "Autumn Lantern Procession"),
        ("Grand Archive Museum", "Golden Crane Panorama"),
        ("Riverfront Gallery", "Jade Mountain Chronicle"),
    )
    generated = None
    for seed in range(20):
        candidate = json_task.generate_random_json_retrieval_case(
            reference_case,
            rng=json_task.random.Random(seed),
            target_total_bytes=2048,
            allowed_museum_artifact_pairs=allowed_pairs,
            forced_museum_artifact_pair=("Palace Museum", "Autumn Lantern Procession"),
            distractor_records_per_case=2,
            answer_template_mode="mixed",
        )
        if candidate["metadata"]["answer_template_id"] != 0:
            generated = candidate
            break

    assert generated is not None

    metadata = generated["metadata"]
    extracted = json_task.extract_slot_labels_from_window_bytes(
        generated["expected_answer_bytes"],
        target_museum="Palace Museum",
    )

    assert metadata["answer_template_mode"] == "mixed"
    assert metadata["answer_template_id"] != 0
    assert extracted["answer_template_id"] == metadata["answer_template_id"]
    assert json_task.build_extract_compose_prediction_bytes(extracted) == generated[
        "expected_answer_bytes"
    ]


def test_json_evidence_target_uses_answer_record_not_distractor_center() -> None:
    reference_case = json_task.load_json_retrieval_case()
    allowed_pairs = (
        ("Palace Museum", "Autumn Lantern Procession"),
        ("Grand Archive Museum", "Golden Crane Panorama"),
        ("Riverfront Gallery", "Jade Mountain Chronicle"),
    )
    generated = json_task.generate_random_json_retrieval_case(
        reference_case,
        rng=json_task.random.Random(20260613),
        target_total_bytes=2048,
        allowed_museum_artifact_pairs=allowed_pairs,
        forced_museum_artifact_pair=("Palace Museum", "Autumn Lantern Procession"),
        distractor_records_per_case=2,
        answer_template_mode="mixed",
    )
    context_bytes = 2048
    window_count = 16
    answer_start, answer_end = json_task.get_relative_answer_record_bounds(
        generated,
        context_bytes=context_bytes,
    )
    needle_start, needle_end = json_task.get_relative_needle_bounds(
        generated,
        context_bytes=context_bytes,
    )
    sample_length = len(json_task.build_curriculum_context(generated, context_bytes))
    answer_center = (answer_start + max(answer_start, answer_end - 1)) / 2.0
    needle_center = (needle_start + max(needle_start, needle_end - 1)) / 2.0
    expected_answer_window = int((answer_center / sample_length) * window_count)
    old_needle_window = int((needle_center / sample_length) * window_count)

    assert generated["metadata"]["distractor_records_per_case"] == 2
    assert answer_start > needle_start
    assert json_task.get_evidence_window_target(
        generated,
        context_bytes=context_bytes,
        window_count=window_count,
    ) == expected_answer_window
    assert expected_answer_window != old_needle_window


def test_json_evidence_target_without_distractors_matches_needle_bounds() -> None:
    reference_case = json_task.load_json_retrieval_case()
    generated = json_task.generate_random_json_retrieval_case(
        reference_case,
        rng=json_task.random.Random(20260614),
        target_total_bytes=2048,
        forced_museum_artifact_pair=("Palace Museum", "Autumn Lantern Procession"),
        distractor_records_per_case=0,
    )

    assert json_task.get_relative_answer_record_bounds(
        generated,
        context_bytes=2048,
    ) == json_task.get_relative_needle_bounds(
        generated,
        context_bytes=2048,
    )


def test_json_generator_keeps_distractors_inside_allowed_split() -> None:
    reference_case = json_task.load_json_retrieval_case()
    allowed_pairs = (
        ("Palace Museum", "Autumn Lantern Procession"),
        ("Grand Archive Museum", "Golden Crane Panorama"),
    )
    generated = json_task.generate_random_json_retrieval_case(
        reference_case,
        rng=json_task.random.Random(20260612),
        target_total_bytes=2048,
        allowed_museum_artifact_pairs=allowed_pairs,
        forced_museum_artifact_pair=("Palace Museum", "Autumn Lantern Procession"),
        distractor_records_per_case=5,
    )

    metadata = generated["metadata"]
    distractor_records = metadata["distractor_records"]
    assert metadata["requested_distractor_records_per_case"] == 5
    assert metadata["distractor_records_per_case"] == 1
    assert len(distractor_records) == 1
    assert "Grand Archive Museum" in distractor_records[0]
    assert "Golden Crane Panorama" in distractor_records[0]
    assert "Riverfront Gallery" not in generated["sample_bytes"].decode("ascii")


def test_niah_needle_copy_readout_copies_from_selected_retrieval_only() -> None:
    X = torch.tensor([[9, 2, 17, 1]], dtype=torch.long)
    targets = torch.tensor([17], dtype=torch.long)
    query_positions = torch.tensor([3], dtype=torch.long)
    aux = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[1, 0]], dtype=torch.long),
                "mask": torch.tensor([[True, True]]),
            },
            "selected_retrieval_token_weight_by_sample": torch.tensor(
                [[0.9, 0.1]],
                dtype=torch.float32,
            ),
        }
    }

    rows = niah_task.compute_niah_needle_copy_readout_sample_metrics(
        X=X,
        targets=targets,
        query_positions=query_positions,
        aux=aux,
        seq_len=4,
        depth=0.5,
    )

    assert rows[0]["readout_available"] is True
    assert rows[0]["copied_candidate_position"] == 1
    assert rows[0]["copied_from_position"] == 2
    assert rows[0]["pred_token"] == 17
    assert rows[0]["correct"] is True
    assert rows[0]["target_candidate_present"] is True
    assert rows[0]["target_candidate_rank"] == 1


def test_niah_pair_copy_readout_prefers_key_value_pair_without_target_lookup() -> None:
    X = torch.tensor([[9, 2, 17, 6, 1]], dtype=torch.long)
    targets = torch.tensor([17], dtype=torch.long)
    query_positions = torch.tensor([4], dtype=torch.long)
    aux = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[3, 1, 2]], dtype=torch.long),
                "mask": torch.tensor([[True, True, True]]),
            },
            "selected_retrieval_token_weight_by_sample": torch.tensor(
                [[0.95, 0.20, 0.10]],
                dtype=torch.float32,
            ),
        }
    }

    top1_rows = niah_task.compute_niah_needle_copy_readout_sample_metrics(
        X=X,
        targets=targets,
        query_positions=query_positions,
        aux=aux,
        seq_len=5,
        depth=0.5,
        readout_mode="needle_copy",
    )
    pair_rows = niah_task.compute_niah_needle_copy_readout_sample_metrics(
        X=X,
        targets=targets,
        query_positions=query_positions,
        aux=aux,
        seq_len=5,
        depth=0.5,
        readout_mode="needle_pair_copy",
    )

    assert top1_rows[0]["readout_available"] is True
    assert top1_rows[0]["copied_candidate_position"] == 3
    assert top1_rows[0]["pred_token"] == 6
    assert top1_rows[0]["correct"] is False

    assert pair_rows[0]["readout_available"] is True
    assert pair_rows[0]["readout_mode"] == "needle_pair_copy"
    assert pair_rows[0]["copied_candidate_position"] == 1
    assert pair_rows[0]["copied_from_position"] == 2
    assert pair_rows[0]["pair_candidate_position"] == 1
    assert pair_rows[0]["pair_neighbor_position"] == 2
    assert pair_rows[0]["pair_neighbor_present"] is True
    assert pair_rows[0]["pred_token"] == 17
    assert pair_rows[0]["correct"] is True
    assert pair_rows[0]["target_candidate_present"] is True
    assert pair_rows[0]["target_candidate_rank"] == 2


def test_niah_pair_copy_readout_accepts_value_candidate_with_left_key() -> None:
    X = torch.tensor([[9, 2, 17, 6, 1]], dtype=torch.long)
    targets = torch.tensor([17], dtype=torch.long)
    query_positions = torch.tensor([4], dtype=torch.long)
    aux = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[3, 2]], dtype=torch.long),
                "mask": torch.tensor([[True, True]]),
            },
            "selected_retrieval_token_weight_by_sample": torch.tensor(
                [[0.95, 0.20]],
                dtype=torch.float32,
            ),
        }
    }

    pair_rows = niah_task.compute_niah_needle_copy_readout_sample_metrics(
        X=X,
        targets=targets,
        query_positions=query_positions,
        aux=aux,
        seq_len=5,
        depth=0.5,
        readout_mode="needle_pair_copy",
    )

    assert pair_rows[0]["readout_available"] is True
    assert pair_rows[0]["copied_candidate_position"] == 2
    assert pair_rows[0]["copied_from_position"] == 2
    assert pair_rows[0]["pair_candidate_position"] == 2
    assert pair_rows[0]["pair_neighbor_position"] == 1
    assert pair_rows[0]["pair_neighbor_present"] is False
    assert pair_rows[0]["pred_token"] == 17
    assert pair_rows[0]["correct"] is True
    assert pair_rows[0]["target_candidate_present"] is True
    assert pair_rows[0]["target_candidate_rank"] == 2


def test_niah_needle_copy_readout_reports_unavailable_without_metadata() -> None:
    X = torch.tensor([[9, 2, 17, 1]], dtype=torch.long)
    targets = torch.tensor([17], dtype=torch.long)
    query_positions = torch.tensor([3], dtype=torch.long)

    rows = niah_task.compute_niah_needle_copy_readout_sample_metrics(
        X=X,
        targets=targets,
        query_positions=query_positions,
        aux={},
        seq_len=4,
        depth=0.5,
    )

    assert rows[0]["readout_available"] is False
    assert rows[0]["readout_unavailable_reason"] == "missing_selected_retrieval_metadata"
    assert rows[0]["pred_token"] == 0


def test_niah_span_predictor_reports_candidate_pair_diagnostics() -> None:
    class FixedSpanModel:
        def score_retrieval_span_candidates(self, *args, **kwargs):
            return torch.tensor([[5.0, 1.0, 2.0]], dtype=torch.float32)

    X = torch.tensor([[9, 2, 17, 8, 1]], dtype=torch.long)
    targets = torch.tensor([17], dtype=torch.long)
    query_positions = torch.tensor([4], dtype=torch.long)
    aux = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[3, 1, 2]], dtype=torch.long),
                "mask": torch.tensor([[True, True, True]]),
            },
            "selected_retrieval_token_weight_by_sample": torch.tensor(
                [[0.80, 0.10, 0.10]],
                dtype=torch.float32,
            ),
        }
    }

    rows = niah_task.compute_niah_span_predictor_sample_metrics(
        model=FixedSpanModel(),
        hidden_query=torch.zeros(1, 16),
        X=X,
        targets=targets,
        query_positions=query_positions,
        aux=aux,
        seq_len=5,
        depth=0.5,
        candidate_filter="prefer_key_value_pair",
    )

    assert rows[0]["readout_available"] is True
    assert rows[0]["correct"] is True
    assert rows[0]["span_raw_candidate_count"] == 3
    assert rows[0]["span_valid_candidate_count"] == 2
    assert rows[0]["span_pair_candidate_count"] == 2
    assert rows[0]["span_pair_available"] is True
    assert rows[0]["span_filter_fallback"] is False
    assert rows[0]["span_selected_is_pair"] is True
    assert rows[0]["span_target_pair_candidate_present"] is True

    summary = niah_task.summarize_niah_sample_metrics(rows)

    assert summary["mean_span_raw_candidate_count"] == pytest.approx(3.0)
    assert summary["mean_span_valid_candidate_count"] == pytest.approx(2.0)
    assert summary["mean_span_pair_candidate_count"] == pytest.approx(2.0)
    assert summary["span_pair_available_rate"] == pytest.approx(1.0)
    assert summary["span_filter_fallback_rate"] == pytest.approx(0.0)
    assert summary["span_selected_pair_rate"] == pytest.approx(1.0)
    assert summary["span_target_pair_candidate_rate"] == pytest.approx(1.0)


def test_niah_span_predictor_reports_page_and_top_token_locality_diagnostics() -> None:
    class FixedSpanModel:
        def score_retrieval_span_candidates(self, *args, **kwargs):
            return torch.tensor([[0.5, 2.0]], dtype=torch.float32)

    X = torch.tensor([[9, 2, 17, 8, 1]], dtype=torch.long)
    targets = torch.tensor([17], dtype=torch.long)
    query_positions = torch.tensor([4], dtype=torch.long)
    aux = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[2, 3]], dtype=torch.long),
                "mask": torch.tensor([[True, True]]),
                "page_candidate_positions": torch.tensor([1, 2, 3], dtype=torch.long),
                "top_token_positions": torch.tensor([2], dtype=torch.long),
                "seed_token_positions": torch.tensor([1, 2], dtype=torch.long),
            },
        }
    }

    rows = niah_task.compute_niah_span_predictor_sample_metrics(
        model=FixedSpanModel(),
        hidden_query=torch.zeros(1, 16),
        X=X,
        targets=targets,
        query_positions=query_positions,
        aux=aux,
        seq_len=5,
        depth=0.5,
        candidate_filter="prefer_key_value_pair",
    )

    assert rows[0]["span_target_value_in_page_candidates"] is True
    assert rows[0]["span_target_pair_in_page_candidates"] is True
    assert rows[0]["span_target_value_in_top_tokens"] is True
    assert rows[0]["span_target_pair_in_top_tokens"] is False
    assert rows[0]["span_target_value_in_seed_tokens"] is True
    assert rows[0]["span_target_pair_in_seed_tokens"] is True

    summary = niah_task.summarize_niah_sample_metrics(rows)

    assert summary["span_target_value_page_candidate_rate"] == pytest.approx(1.0)
    assert summary["span_target_pair_page_candidate_rate"] == pytest.approx(1.0)
    assert summary["span_target_value_top_token_rate"] == pytest.approx(1.0)
    assert summary["span_target_pair_top_token_rate"] == pytest.approx(0.0)
    assert summary["span_target_value_seed_token_rate"] == pytest.approx(1.0)
    assert summary["span_target_pair_seed_token_rate"] == pytest.approx(1.0)


def test_niah_eval_returns_seed_token_locality_diagnostics() -> None:
    class FixedSpanEvalModel:
        def __init__(self) -> None:
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self):
            self.training = True
            return self

        def forward_selected_logits(
            self,
            X,
            query_positions,
            return_hidden=False,
            return_aux=False,
        ):
            batch_size, _ = X.shape
            position_rows = []
            mask_rows = []
            page_rows = []
            top_rows = []
            seed_rows = []
            for sample_idx in range(batch_size):
                query_position = int(query_positions[sample_idx].item())
                key_positions = (
                    X[sample_idx, :query_position]
                    .eq(niah_task.NEEDLE_KEY_TOKEN_ID)
                    .nonzero(as_tuple=True)[0]
                )
                needle_key_position = int(key_positions[0].item())
                needle_value_position = needle_key_position + 1
                positions = torch.tensor(
                    [needle_key_position, needle_value_position],
                    dtype=torch.long,
                )
                position_rows.append(positions)
                mask_rows.append(torch.ones_like(positions, dtype=torch.bool))
                page_rows.append(
                    torch.tensor(
                        [needle_key_position, needle_value_position, query_position - 1],
                        dtype=torch.long,
                    )
                )
                top_rows.append(torch.tensor([needle_value_position], dtype=torch.long))
                seed_rows.append(positions)
            aux = {
                "last_layer": {
                    "selected_retrieval_metadata": {
                        "positions": torch.stack(position_rows),
                        "mask": torch.stack(mask_rows),
                        "page_candidate_positions_by_sample": page_rows,
                        "top_token_positions_by_sample": top_rows,
                        "seed_token_positions_by_sample": seed_rows,
                    }
                }
            }
            logits = torch.zeros(batch_size, int(X.max().item()) + 1)
            hidden = torch.zeros(batch_size, 4)
            if return_hidden and return_aux:
                return logits, hidden, aux
            if return_aux:
                return logits, aux
            if return_hidden:
                return logits, hidden
            return logits

        def score_retrieval_span_candidates(self, hidden_query, *args, **kwargs):
            candidate_token_ids = args[0]
            logits = torch.zeros_like(candidate_token_ids, dtype=torch.float32)
            logits[:, 0] = 2.0
            if logits.shape[1] > 1:
                logits[:, 1:] = 1.0
            return logits

    result = niah_task.evaluate_niah_depths(
        model=FixedSpanEvalModel(),
        seq_len=16,
        device=torch.device("cpu"),
        vocab_size=32,
        batch_size=2,
        criterion=torch.nn.CrossEntropyLoss(),
        depths=(0.5,),
        batches_per_depth=1,
        niah_readout_mode="span_predictor",
        niah_span_candidate_filter="prefer_key_value_pair",
    )

    assert result["span_target_value_seed_token_rate"] == pytest.approx(1.0)
    assert result["span_target_pair_seed_token_rate"] == pytest.approx(1.0)
    assert result["depth_rows"][0]["span_target_value_seed_token_rate"] == pytest.approx(
        1.0
    )
    assert result["depth_rows"][0]["span_target_pair_seed_token_rate"] == pytest.approx(
        1.0
    )


def test_niah_span_predictor_reports_prefer_pair_fallback() -> None:
    class FixedSpanModel:
        def score_retrieval_span_candidates(self, *args, **kwargs):
            return torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    X = torch.tensor([[9, 8, 7, 1]], dtype=torch.long)
    targets = torch.tensor([7], dtype=torch.long)
    query_positions = torch.tensor([3], dtype=torch.long)
    aux = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[0, 2]], dtype=torch.long),
                "mask": torch.tensor([[True, True]]),
            },
        }
    }

    rows = niah_task.compute_niah_span_predictor_sample_metrics(
        model=FixedSpanModel(),
        hidden_query=torch.zeros(1, 16),
        X=X,
        targets=targets,
        query_positions=query_positions,
        aux=aux,
        seq_len=4,
        depth=0.5,
        candidate_filter="prefer_key_value_pair",
    )

    assert rows[0]["readout_available"] is True
    assert rows[0]["correct"] is True
    assert rows[0]["span_raw_candidate_count"] == 2
    assert rows[0]["span_valid_candidate_count"] == 2
    assert rows[0]["span_pair_candidate_count"] == 0
    assert rows[0]["span_pair_available"] is False
    assert rows[0]["span_filter_fallback"] is True
    assert rows[0]["span_selected_is_pair"] is False
    assert rows[0]["span_target_pair_candidate_present"] is False


def test_niah_verification_rejects_invalid_retrieval_evidence_target_offset() -> None:
    with pytest.raises(ValueError, match="retrieval_evidence_target_offset"):
        niah_task.run_niah_verification_case(
            seq_len=16,
            device=torch.device("cpu"),
            vocab_size=16,
            dim=8,
            num_layers=1,
            K=4,
            kr=1,
            chunk_size=8,
            batch_size=1,
            epochs=1,
            retrieval_evidence_target_offset=2,
        )


def test_json_generalization_entry_accepts_mhdsra2_override() -> None:
    signature = inspect.signature(run_json_retrieval_generalization_test)

    assert "mhdsra2_config_override" in signature.parameters
    assert signature.parameters["mhdsra2_config_override"].default is None
    assert "distractor_records_per_case" in signature.parameters
    assert signature.parameters["distractor_records_per_case"].default == 0


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
    if "selected_retrieval_token_score_by_sample" in aux["last_layer"]:
        selected_scores = aux["last_layer"]["selected_retrieval_token_score_by_sample"]
        assert selected_scores.dim() == 2
        assert selected_scores.shape[0] == 1
    if "retrieval_metadata" in aux["last_layer"]:
        assert "selected_retrieval_metadata" in aux["last_layer"]
    assert "selected_retrieval_query_projection" not in aux["last_layer"]
    assert "selected_retrieval_key_projection" not in aux["last_layer"]


def test_multilayer_selected_retrieval_projection_aux_is_explicit() -> None:
    import torch

    torch.manual_seed(788)
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
    query_positions = torch.tensor([7], dtype=torch.long)

    with torch.no_grad():
        logits, aux = model.forward_selected_logits(
            tokens,
            query_positions,
            return_aux=True,
            return_retrieval_projection_aux=True,
            train_retrieval_evidence_positions=torch.tensor([2], dtype=torch.long),
        )

    assert logits.shape == (1, 32)
    last_layer = aux["last_layer"]
    assert "selected_retrieval_query_projection" in last_layer
    assert "selected_retrieval_key_projection" in last_layer
    query_projection = last_layer["selected_retrieval_query_projection"]
    key_projection = last_layer["selected_retrieval_key_projection"]
    heads = model.layers[-1].heads
    d_head = model.layers[-1].d_head
    assert query_projection.shape == (1, heads, d_head)
    assert key_projection.dim() == 4
    assert key_projection.shape[0] == 1
    assert key_projection.shape[1] == heads
    assert key_projection.shape[-1] == d_head


def test_multilayer_train_evidence_injection_is_explicit_and_selected() -> None:
    import torch

    torch.manual_seed(789)
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
    query_positions = torch.tensor([7], dtype=torch.long)
    evidence_positions = torch.tensor([2], dtype=torch.long)

    with torch.no_grad():
        _, aux_without_evidence = model.forward_selected_logits(
            tokens,
            query_positions,
            return_aux=True,
        )
        _, aux_with_evidence = model.forward_selected_logits(
            tokens,
            query_positions,
            return_aux=True,
            train_retrieval_evidence_positions=evidence_positions,
        )

    metadata_without = aux_without_evidence["last_layer"].get("selected_retrieval_metadata", {})
    assert metadata_without.get("train_evidence_injected") is not True
    metadata_with = aux_with_evidence["last_layer"]["selected_retrieval_metadata"]
    assert metadata_with["train_evidence_injected"] is True
    positions = metadata_with["positions"]
    mask = metadata_with["mask"]
    assert bool(((positions == 2) & mask).any().item())


def test_multilayer_train_evidence_injection_makes_evidence_loss_available() -> None:
    import torch

    torch.manual_seed(790)
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
    logits, aux = model.forward_selected_logits(
        tokens,
        torch.tensor([7], dtype=torch.long),
        return_aux=True,
        train_retrieval_evidence_positions=torch.tensor([2], dtype=torch.long),
    )

    loss, metrics = compute_retrieval_evidence_gate_loss(
        aux,
        torch.tensor([2], dtype=torch.long),
        device=logits.device,
    )
    loss.backward()

    assert metrics["available"] is True
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["positive_count"] == 1
    assert loss.item() > 0.0


def test_multilayer_train_evidence_injection_accepts_multiple_positions() -> None:
    torch.manual_seed(810)
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

    with torch.no_grad():
        _, aux = model.forward_selected_logits(
            tokens,
            torch.tensor([7], dtype=torch.long),
            return_aux=True,
            train_retrieval_evidence_positions=torch.tensor([[1, 2]], dtype=torch.long),
        )

    metadata = aux["last_layer"]["selected_retrieval_metadata"]
    positions = metadata["positions"]
    mask = metadata["mask"]
    assert bool(((positions == 1) & mask).any().item())
    assert bool(((positions == 2) & mask).any().item())


def test_multilayer_retrieval_span_predictor_is_default_off() -> None:
    model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=4,
        kr=1,
        chunk_size=4,
        use_retrieval=True,
    )

    assert model.use_retrieval_span_predictor is False
    assert model.retrieval_span_predictor is None


def test_retrieval_span_predictor_structure_features_are_opt_in() -> None:
    base_model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=4,
        kr=1,
        chunk_size=4,
        use_retrieval=True,
        use_retrieval_span_predictor=True,
    )
    structured_model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=4,
        kr=1,
        chunk_size=4,
        use_retrieval=True,
        use_retrieval_span_predictor=True,
        retrieval_span_structure_features=True,
    )

    assert base_model.retrieval_span_structure_features is False
    assert structured_model.retrieval_span_structure_features is True
    assert base_model.retrieval_span_predictor[0].in_features == 16 * 4 + 4
    assert structured_model.retrieval_span_predictor[0].in_features == 16 * 4 + 8


def test_retrieval_span_predictor_loss_backpropagates_to_predictor() -> None:
    torch.manual_seed(811)
    model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=4,
        kr=1,
        chunk_size=4,
        use_retrieval=True,
        use_retrieval_span_predictor=True,
    )
    tokens = torch.tensor([[9, 2, 17, 5, 1]], dtype=torch.long)
    targets = torch.tensor([17], dtype=torch.long)
    query_positions = torch.tensor([4], dtype=torch.long)
    hidden_query = torch.randn(1, 16, requires_grad=True)
    aux = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "mask": torch.tensor([[True, True, True]]),
            },
            "selected_retrieval_token_weight_by_sample_for_loss": torch.tensor(
                [[0.4, 0.2, 0.6]],
                dtype=torch.float32,
            ),
        }
    }

    loss, metrics = compute_retrieval_span_predictor_loss(
        model,
        hidden_query,
        tokens,
        targets,
        query_positions,
        aux,
        device=torch.device("cpu"),
    )
    loss.backward()

    assert metrics["available"] is True
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["positive_count"] == 1
    assert metrics["target_rank_mean"] is not None
    assert hidden_query.grad is not None
    assert float(hidden_query.grad.abs().sum().item()) > 0.0
    first_weight = model.retrieval_span_predictor[0].weight
    assert first_weight.grad is not None
    assert float(first_weight.grad.abs().sum().item()) > 0.0


def test_retrieval_span_predictor_loss_passes_structure_features() -> None:
    class CaptureSpanModel:
        def __init__(self) -> None:
            self.kwargs = None

        def score_retrieval_span_candidates(self, *args, **kwargs):
            self.kwargs = kwargs
            return torch.tensor([[0.1, 2.0]], dtype=torch.float32, requires_grad=True)

    model = CaptureSpanModel()
    tokens = torch.tensor([[9, 2, 17, 5, 1]], dtype=torch.long)
    targets = torch.tensor([17], dtype=torch.long)
    aux = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[1, 2]], dtype=torch.long),
                "mask": torch.tensor([[True, True]]),
            },
            "selected_retrieval_token_weight_by_sample_for_loss": torch.tensor(
                [[0.4, 0.6]],
                dtype=torch.float32,
            ),
        }
    }

    loss, metrics = compute_retrieval_span_predictor_loss(
        model,
        torch.randn(1, 16, requires_grad=True),
        tokens,
        targets,
        torch.tensor([4], dtype=torch.long),
        aux,
        device=torch.device("cpu"),
        candidate_filter="prefer_key_value_pair",
    )

    assert metrics["available"] is True
    assert loss.item() >= 0.0
    assert model.kwargs is not None
    assert torch.equal(
        model.kwargs["candidate_pair_mask"],
        torch.tensor([[True, True]]),
    )
    assert torch.equal(
        model.kwargs["source_positions"],
        torch.tensor([[2, 2]], dtype=torch.long),
    )


def test_retrieval_span_predictor_loss_rejects_missing_candidate_hit() -> None:
    model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=4,
        kr=1,
        chunk_size=4,
        use_retrieval=True,
        use_retrieval_span_predictor=True,
    )
    tokens = torch.tensor([[9, 2, 17, 5, 1]], dtype=torch.long)
    targets = torch.tensor([17], dtype=torch.long)
    aux = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[3]], dtype=torch.long),
                "mask": torch.tensor([[True]]),
            },
        }
    }

    loss, metrics = compute_retrieval_span_predictor_loss(
        model,
        torch.randn(1, 16, requires_grad=True),
        tokens,
        targets,
        torch.tensor([4], dtype=torch.long),
        aux,
        device=torch.device("cpu"),
    )

    assert loss.item() == pytest.approx(0.0)
    assert metrics["available"] is False
    assert metrics["unavailable_reason"] == "target_not_in_selected_candidates"


def test_retrieval_span_predictor_requires_right_neighbor_candidate_for_key_copy() -> None:
    model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=4,
        kr=1,
        chunk_size=4,
        use_retrieval=True,
        use_retrieval_span_predictor=True,
    )
    tokens = torch.tensor([[9, 2, 17, 5, 1]], dtype=torch.long)
    targets = torch.tensor([17], dtype=torch.long)
    aux = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[1, 3]], dtype=torch.long),
                "mask": torch.tensor([[True, True]]),
            },
        }
    }

    loss, metrics = compute_retrieval_span_predictor_loss(
        model,
        torch.randn(1, 16, requires_grad=True),
        tokens,
        targets,
        torch.tensor([4], dtype=torch.long),
        aux,
        device=torch.device("cpu"),
    )

    assert loss.item() == pytest.approx(0.0)
    assert metrics["available"] is False
    assert metrics["unavailable_reason"] == "target_not_in_selected_candidates"


def test_structured_span_filter_requires_recalled_key_value_pair() -> None:
    model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=4,
        kr=1,
        chunk_size=4,
        use_retrieval=True,
        use_retrieval_span_predictor=True,
    )
    tokens = torch.tensor([[9, 2, 17, 5, 1]], dtype=torch.long)
    targets = torch.tensor([17], dtype=torch.long)
    query_positions = torch.tensor([4], dtype=torch.long)
    aux_value_only = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[2, 3]], dtype=torch.long),
                "mask": torch.tensor([[True, True]]),
            },
        }
    }

    all_loss, all_metrics = compute_retrieval_span_predictor_loss(
        model,
        torch.randn(1, 16, requires_grad=True),
        tokens,
        targets,
        query_positions,
        aux_value_only,
        device=torch.device("cpu"),
        candidate_filter="all",
    )
    structured_loss, structured_metrics = compute_retrieval_span_predictor_loss(
        model,
        torch.randn(1, 16, requires_grad=True),
        tokens,
        targets,
        query_positions,
        aux_value_only,
        device=torch.device("cpu"),
        candidate_filter="key_value_pair",
    )

    assert all_loss.item() > 0.0
    assert all_metrics["available"] is True
    assert all_metrics["candidate_filter"] == "all"
    assert structured_loss.item() == pytest.approx(0.0)
    assert structured_metrics["available"] is False
    assert structured_metrics["candidate_filter"] == "key_value_pair"
    assert structured_metrics["unavailable_reason"] == "target_not_in_selected_candidates"

    aux_key_value_pair = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "mask": torch.tensor([[True, True, True]]),
            },
        }
    }
    pair_loss, pair_metrics = compute_retrieval_span_predictor_loss(
        model,
        torch.randn(1, 16, requires_grad=True),
        tokens,
        targets,
        query_positions,
        aux_key_value_pair,
        device=torch.device("cpu"),
        candidate_filter="key_value_pair",
    )

    assert pair_loss.item() > 0.0
    assert pair_metrics["available"] is True
    assert pair_metrics["candidate_filter"] == "key_value_pair"
    assert pair_metrics["hit_rate"] == pytest.approx(1.0)


def test_prefer_structured_span_filter_falls_back_when_pair_is_missing() -> None:
    model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=4,
        kr=1,
        chunk_size=4,
        use_retrieval=True,
        use_retrieval_span_predictor=True,
    )
    tokens = torch.tensor([[9, 2, 17, 5, 1]], dtype=torch.long)
    targets = torch.tensor([17], dtype=torch.long)
    query_positions = torch.tensor([4], dtype=torch.long)
    aux_value_only = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[2, 3]], dtype=torch.long),
                "mask": torch.tensor([[True, True]]),
            },
        }
    }

    fallback_loss, fallback_metrics = compute_retrieval_span_predictor_loss(
        model,
        torch.randn(1, 16, requires_grad=True),
        tokens,
        targets,
        query_positions,
        aux_value_only,
        device=torch.device("cpu"),
        candidate_filter="prefer_key_value_pair",
    )

    assert fallback_loss.item() > 0.0
    assert fallback_metrics["available"] is True
    assert fallback_metrics["candidate_filter"] == "prefer_key_value_pair"
    assert fallback_metrics["hit_rate"] == pytest.approx(1.0)

    aux_key_value_pair = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "mask": torch.tensor([[True, True, True]]),
            },
        }
    }
    pair_loss, pair_metrics = compute_retrieval_span_predictor_loss(
        model,
        torch.randn(1, 16, requires_grad=True),
        tokens,
        targets,
        query_positions,
        aux_key_value_pair,
        device=torch.device("cpu"),
        candidate_filter="prefer_key_value_pair",
    )

    assert pair_loss.item() > 0.0
    assert pair_metrics["available"] is True
    assert pair_metrics["candidate_filter"] == "prefer_key_value_pair"
    assert pair_metrics["target_rank_mean"] == pytest.approx(1.0)


def test_retrieval_span_predictor_multi_positive_loss_accepts_all_correct_candidates() -> None:
    class FixedSpanModel:
        def score_retrieval_span_candidates(self, *args, **kwargs):
            return torch.tensor([[0.0, 4.0, 4.0]], dtype=torch.float32, requires_grad=True)

    model = FixedSpanModel()
    tokens = torch.tensor([[9, 2, 17, 17, 1]], dtype=torch.long)
    targets = torch.tensor([17], dtype=torch.long)
    query_positions = torch.tensor([4], dtype=torch.long)
    aux = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[0, 2, 3]], dtype=torch.long),
                "mask": torch.tensor([[True, True, True]]),
            },
        }
    }

    single_loss, single_metrics = compute_retrieval_span_predictor_loss(
        model,
        torch.zeros(1, 16),
        tokens,
        targets,
        query_positions,
        aux,
        device=torch.device("cpu"),
        loss_mode="single_positive",
    )
    multi_loss, multi_metrics = compute_retrieval_span_predictor_loss(
        model,
        torch.zeros(1, 16),
        tokens,
        targets,
        query_positions,
        aux,
        device=torch.device("cpu"),
        loss_mode="multi_positive",
    )

    assert single_metrics["available"] is True
    assert single_metrics["loss_mode"] == "single_positive"
    assert multi_metrics["available"] is True
    assert multi_metrics["loss_mode"] == "multi_positive"
    assert multi_metrics["positive_count"] == 1
    assert multi_loss.item() < single_loss.item()
    expected_single = -torch.log(
        torch.exp(torch.tensor(4.0))
        / (torch.exp(torch.tensor(0.0)) + torch.exp(torch.tensor(4.0)) * 2.0)
    )
    expected_multi = -torch.log(
        (torch.exp(torch.tensor(4.0)) * 2.0)
        / (torch.exp(torch.tensor(0.0)) + torch.exp(torch.tensor(4.0)) * 2.0)
    )
    assert multi_loss.item() == pytest.approx(float(expected_multi.item()), abs=1e-6)
    assert single_loss.item() == pytest.approx(float(expected_single.item()), abs=1e-6)


def test_retrieval_span_predictor_step_summary_aggregates_available_rows() -> None:
    summary = summarize_retrieval_span_predictor_step_metrics(
        [
            {
                "retrieval_span_predictor_available": True,
                "retrieval_span_predictor_loss": 0.8,
                "retrieval_span_predictor_hit_rate": 1.0,
                "retrieval_span_predictor_positive_count": 1,
                "retrieval_span_predictor_target_rank_mean": 2.0,
                "retrieval_span_predictor_top1_rate": 0.0,
                "retrieval_span_predictor_logit_margin_mean": -0.5,
            },
            {
                "retrieval_span_predictor_available": True,
                "retrieval_span_predictor_loss": 0.2,
                "retrieval_span_predictor_hit_rate": 1.0,
                "retrieval_span_predictor_positive_count": 1,
                "retrieval_span_predictor_target_rank_mean": 1.0,
                "retrieval_span_predictor_top1_rate": 1.0,
                "retrieval_span_predictor_logit_margin_mean": 0.5,
            },
            {"retrieval_span_predictor_available": False},
        ]
    )

    assert summary["steps"] == 3
    assert summary["available_steps"] == 2
    assert summary["positive_steps"] == 2
    assert summary["mean_loss"] == pytest.approx(0.5)
    assert summary["mean_target_rank"] == pytest.approx(1.5)
    assert summary["mean_top1_rate"] == pytest.approx(0.5)
    assert summary["mean_logit_margin"] == pytest.approx(0.0)


def test_multilayer_train_evidence_injection_respects_future_cutoff() -> None:
    import torch

    torch.manual_seed(791)
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

    with torch.no_grad():
        _, aux = model.forward_selected_logits(
            tokens,
            torch.tensor([3], dtype=torch.long),
            return_aux=True,
            train_retrieval_evidence_positions=torch.tensor([6], dtype=torch.long),
        )

    metadata = aux["last_layer"].get("selected_retrieval_metadata")
    if isinstance(metadata, dict):
        positions = metadata.get("positions")
        mask = metadata.get("mask")
        if isinstance(positions, torch.Tensor) and isinstance(mask, torch.Tensor):
            assert not bool(((positions == 6) & mask).any().item())


def test_multilayer_retrieval_neighbor_span_reaches_selected_metadata() -> None:
    import torch

    torch.manual_seed(792)
    model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=4,
        kr=1,
        chunk_size=4,
        use_retrieval=True,
        mhdsra2_config_override={"retrieval_neighbor_span": 1},
    )
    tokens = torch.tensor([[2, 17, 5, 6, 2, 1, 7, 8]], dtype=torch.long)

    with torch.no_grad():
        _, aux = model.forward_selected_logits(
            tokens,
            torch.tensor([5], dtype=torch.long),
            return_aux=True,
        )

    metadata = aux["last_layer"].get("selected_retrieval_metadata")
    assert isinstance(metadata, dict)
    positions = metadata.get("positions")
    mask = metadata.get("mask")
    assert isinstance(positions, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    valid_positions = set(positions[mask].detach().cpu().tolist())
    assert any((position + 1) in valid_positions for position in valid_positions)


def test_multilayer_bidirectional_neighbor_span_reaches_selected_metadata() -> None:
    import torch

    torch.manual_seed(793)
    model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=4,
        kr=1,
        chunk_size=4,
        use_retrieval=True,
        mhdsra2_config_override={
            "retrieval_neighbor_span": 1,
            "retrieval_neighbor_direction": "both",
        },
    )
    tokens = torch.tensor([[9, 2, 17, 6, 5, 1, 7, 8]], dtype=torch.long)

    with torch.no_grad():
        _, aux = model.forward_selected_logits(
            tokens,
            torch.tensor([5], dtype=torch.long),
            return_aux=True,
        )

    metadata = aux["last_layer"].get("selected_retrieval_metadata")
    assert isinstance(metadata, dict)
    positions = metadata.get("positions")
    mask = metadata.get("mask")
    assert isinstance(positions, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    valid_positions = set(positions[mask].detach().cpu().tolist())
    assert any((position - 1) in valid_positions for position in valid_positions)


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
    assert "page_candidate_positions" in metadata
    assert "top_token_positions" in metadata
    assert metadata["page_candidate_positions"].dim() == 1
    assert metadata["top_token_positions"].dim() == 1


def test_multilayer_selected_metadata_carries_retrieval_locality_diagnostics() -> None:
    import torch

    torch.manual_seed(794)
    model = MultiLayerMHDSRA2Model(
        vocab_size=32,
        dim=16,
        num_layers=1,
        K=4,
        kr=1,
        chunk_size=4,
        use_retrieval=True,
    )
    tokens = torch.tensor([[2, 17, 5, 6, 2, 1, 7, 8]], dtype=torch.long)

    with torch.no_grad():
        _, aux = model.forward_selected_logits(
            tokens,
            torch.tensor([5], dtype=torch.long),
            return_aux=True,
        )

    metadata = aux["last_layer"].get("selected_retrieval_metadata")
    assert isinstance(metadata, dict)
    assert isinstance(metadata.get("page_candidate_positions_by_sample"), list)
    assert isinstance(metadata.get("top_token_positions_by_sample"), list)
    assert isinstance(metadata.get("seed_token_positions_by_sample"), list)
    assert len(metadata["page_candidate_positions_by_sample"]) == 1
    assert len(metadata["top_token_positions_by_sample"]) == 1
    assert len(metadata["seed_token_positions_by_sample"]) == 1


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


def test_retrieval_evidence_rank_margin_penalizes_stronger_negative() -> None:
    import torch

    gate = torch.tensor([0.80], requires_grad=True)
    token_weights = torch.tensor([[0.20, 0.60, 0.20]], requires_grad=True)
    aux = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([1, 2, 3]),
                "mask": torch.tensor([True, True, True]),
            },
            "selected_gate_retrieval_by_sample_for_loss": gate,
            "selected_retrieval_token_weight_by_sample_for_loss": token_weights,
        }
    }

    loss, metrics = compute_retrieval_evidence_gate_loss(
        aux,
        torch.tensor([1]),
        device=torch.device("cpu"),
        rank_margin=0.15,
    )
    loss.backward()

    assert metrics["available"] is True
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["evidence_weight_mean"] == pytest.approx(0.20)
    assert metrics["best_negative_weight_mean"] == pytest.approx(0.60)
    assert metrics["evidence_margin_mean"] == pytest.approx(-0.40)
    assert metrics["target_rank_mean"] == pytest.approx(2.0)
    assert metrics["top1_rate"] == pytest.approx(0.0)
    assert metrics["margin_loss"] == pytest.approx(0.55)
    assert token_weights.grad is not None
    assert float(token_weights.grad[0, 0].item()) < 0.0
    assert float(token_weights.grad[0, 1].item()) > 0.0


def test_retrieval_evidence_score_margin_penalizes_stronger_negative_score() -> None:
    import torch

    gate = torch.tensor([0.80], requires_grad=True)
    token_weights = torch.tensor([[0.20, 0.60, 0.20]], requires_grad=True)
    token_scores = torch.tensor([[0.10, 0.70, 0.20]], requires_grad=True)
    aux = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([1, 2, 3]),
                "mask": torch.tensor([True, True, True]),
            },
            "selected_gate_retrieval_by_sample_for_loss": gate,
            "selected_retrieval_token_weight_by_sample_for_loss": token_weights,
            "selected_retrieval_token_score_by_sample_for_loss": token_scores,
        }
    }

    loss, metrics = compute_retrieval_evidence_gate_loss(
        aux,
        torch.tensor([1]),
        device=torch.device("cpu"),
        score_margin=0.25,
    )
    loss.backward()

    assert metrics["available"] is True
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["evidence_score_mean"] == pytest.approx(0.10)
    assert metrics["best_negative_score_mean"] == pytest.approx(0.70)
    assert metrics["evidence_score_margin_mean"] == pytest.approx(-0.60)
    assert metrics["score_target_rank_mean"] == pytest.approx(3.0)
    assert metrics["score_top1_rate"] == pytest.approx(0.0)
    assert metrics["score_margin_loss"] == pytest.approx(0.85)
    assert token_scores.grad is not None
    assert float(token_scores.grad[0, 0].item()) < 0.0
    assert float(token_scores.grad[0, 1].item()) > 0.0


def test_query_evidence_alignment_loss_backpropagates_to_hidden_only() -> None:
    import torch

    embedding = torch.nn.Embedding(8, 4)
    hidden_query = torch.randn(2, 4, requires_grad=True)

    loss, metrics = compute_query_evidence_alignment_loss(
        hidden_query,
        torch.tensor([2, 3], dtype=torch.long),
        embedding,
    )
    loss.backward()

    assert metrics["available"] is True
    assert metrics["loss"] == pytest.approx(float(loss.detach().item()))
    assert metrics["mean_cosine"] is not None
    assert metrics["mean_mse"] is not None
    assert hidden_query.grad is not None
    assert embedding.weight.grad is None


def test_query_evidence_alignment_loss_rejects_batch_mismatch() -> None:
    import torch

    embedding = torch.nn.Embedding(8, 4)
    hidden_query = torch.randn(2, 4)

    with pytest.raises(ValueError, match="evidence token count"):
        compute_query_evidence_alignment_loss(
            hidden_query,
            torch.tensor([2], dtype=torch.long),
            embedding,
        )


def test_retrieval_projection_contrastive_loss_backpropagates_to_query() -> None:
    import torch

    query = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    keys = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    aux = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[4, 5]], dtype=torch.long),
                "mask": torch.tensor([[True, True]]),
            },
            "selected_retrieval_query_projection_for_loss": query,
            "selected_retrieval_key_projection_for_loss": keys,
        }
    }

    loss, metrics = compute_retrieval_projection_contrastive_loss(
        aux,
        torch.tensor([4]),
        device=torch.device("cpu"),
        temperature=0.5,
    )
    loss.backward()

    assert metrics["available"] is True
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["positive_count"] == 1
    assert metrics["target_rank_mean"] == pytest.approx(2.0)
    assert metrics["top1_rate"] == pytest.approx(0.0)
    assert metrics["score_margin_mean"] < 0.0
    assert query.grad is not None
    assert query.grad[0, 0, 1].item() < 0.0


def test_retrieval_projection_contrastive_loss_rejects_missing_projection_aux() -> None:
    aux = {
        "last_layer": {
            "selected_retrieval_metadata": {
                "positions": torch.tensor([[4]], dtype=torch.long),
                "mask": torch.tensor([[True]]),
            }
        }
    }

    loss, metrics = compute_retrieval_projection_contrastive_loss(
        aux,
        torch.tensor([4]),
        device=torch.device("cpu"),
    )

    assert loss.item() == pytest.approx(0.0)
    assert metrics["available"] is False
    assert metrics["unavailable_reason"] == "missing_selected_projection_aux"


def test_retrieval_projection_step_summary_aggregates_available_rows() -> None:
    rows = [
        {
            "retrieval_projection_available": False,
            "retrieval_projection_hit_rate": None,
        },
        {
            "retrieval_projection_available": True,
            "retrieval_projection_loss": 0.8,
            "retrieval_projection_hit_rate": 1.0,
            "retrieval_projection_positive_count": 1,
            "retrieval_projection_evidence_score_mean": 0.2,
            "retrieval_projection_best_negative_score_mean": 0.5,
            "retrieval_projection_score_margin_mean": -0.3,
            "retrieval_projection_target_rank_mean": 2.0,
            "retrieval_projection_top1_rate": 0.0,
        },
        {
            "retrieval_projection_available": True,
            "retrieval_projection_loss": 0.2,
            "retrieval_projection_hit_rate": 1.0,
            "retrieval_projection_positive_count": 1,
            "retrieval_projection_evidence_score_mean": 0.7,
            "retrieval_projection_best_negative_score_mean": 0.4,
            "retrieval_projection_score_margin_mean": 0.3,
            "retrieval_projection_target_rank_mean": 1.0,
            "retrieval_projection_top1_rate": 1.0,
        },
    ]

    summary = summarize_retrieval_projection_step_metrics(rows)

    assert summary["steps"] == 3
    assert summary["available_steps"] == 2
    assert summary["positive_steps"] == 2
    assert summary["mean_loss"] == pytest.approx(0.5)
    assert summary["mean_target_rank"] == pytest.approx(1.5)
    assert summary["mean_top1_rate"] == pytest.approx(0.5)


def test_retrieval_evidence_step_summary_aggregates_available_rows() -> None:
    rows = [
        {
            "retrieval_evidence_available": False,
            "retrieval_evidence_hit_rate": None,
        },
        {
            "retrieval_evidence_available": True,
            "retrieval_evidence_hit_rate": 0.0,
            "retrieval_evidence_weight_mean": 0.0,
            "retrieval_evidence_best_negative_weight_mean": 0.1,
            "retrieval_evidence_margin_mean": -0.1,
            "retrieval_evidence_target_rank_mean": None,
            "retrieval_evidence_top1_rate": None,
            "retrieval_evidence_ranking_loss": 0.0,
            "retrieval_evidence_margin_loss": 0.0,
            "retrieval_evidence_score_margin_loss": 0.0,
            "retrieval_evidence_score_mean": 0.0,
            "retrieval_evidence_best_negative_score_mean": 0.0,
            "retrieval_evidence_score_margin_mean": 0.0,
            "retrieval_evidence_score_target_rank_mean": None,
            "retrieval_evidence_score_top1_rate": None,
            "retrieval_evidence_gate_loss": 0.5,
            "retrieval_evidence_positive_count": 0,
        },
        {
            "retrieval_evidence_available": True,
            "retrieval_evidence_hit_rate": 1.0,
            "retrieval_evidence_weight_mean": 0.25,
            "retrieval_evidence_best_negative_weight_mean": 0.5,
            "retrieval_evidence_margin_mean": -0.25,
            "retrieval_evidence_target_rank_mean": 3.0,
            "retrieval_evidence_top1_rate": 0.0,
            "retrieval_evidence_ranking_loss": 1.4,
            "retrieval_evidence_margin_loss": 0.4,
            "retrieval_evidence_score_margin_loss": 0.8,
            "retrieval_evidence_score_mean": 0.1,
            "retrieval_evidence_best_negative_score_mean": 0.7,
            "retrieval_evidence_score_margin_mean": -0.6,
            "retrieval_evidence_score_target_rank_mean": 3.0,
            "retrieval_evidence_score_top1_rate": 0.0,
            "retrieval_evidence_gate_loss": 0.3,
            "retrieval_evidence_positive_count": 1,
        },
    ]

    summary = summarize_retrieval_evidence_step_metrics(rows)

    assert summary["steps"] == 3
    assert summary["available_steps"] == 2
    assert summary["positive_steps"] == 1
    assert summary["mean_hit_rate"] == pytest.approx(0.5)
    assert summary["mean_evidence_weight"] == pytest.approx(0.125)
    assert summary["mean_best_negative_weight"] == pytest.approx(0.3)
    assert summary["mean_evidence_margin"] == pytest.approx(-0.175)
    assert summary["mean_target_rank"] == pytest.approx(3.0)
    assert summary["mean_top1_rate"] == pytest.approx(0.0)
    assert summary["mean_margin_loss"] == pytest.approx(0.2)
    assert summary["mean_score_margin_loss"] == pytest.approx(0.4)
    assert summary["mean_evidence_score"] == pytest.approx(0.05)
    assert summary["mean_best_negative_score"] == pytest.approx(0.35)
    assert summary["mean_evidence_score_margin"] == pytest.approx(-0.3)
    assert summary["mean_score_target_rank"] == pytest.approx(3.0)
    assert summary["mean_score_top1_rate"] == pytest.approx(0.0)


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
    token_scores = aux["retrieval_token_score_by_sample"]
    assert token_scores.shape == (1, 3)
    assert token_scores[0, 2].item() == pytest.approx(0.0, abs=1e-6)
    assert "retrieval_token_score_by_sample_for_loss" in aux


def test_retrieval_attention_return_scores_preserves_legacy_weights_tuple() -> None:
    import torch

    torch.manual_seed(457)
    layer = MultiHeadDSRA2(MHDSRA2Config(dim=16, heads=2, slots=4, read_topk=2))
    q = torch.randn(1, 2, 3, 8)
    retrieved_k = torch.randn(1, 2, 4, 8)
    retrieved_v = torch.randn(1, 2, 4, 8)
    retrieved_mask = torch.tensor([[True, True, False, True]])

    legacy = layer._retrieval_attention(
        q,
        retrieved_k,
        retrieved_v,
        retrieved_mask,
        return_weights=True,
    )
    with_scores = layer._retrieval_attention(
        q,
        retrieved_k,
        retrieved_v,
        retrieved_mask,
        return_weights=True,
        return_scores=True,
    )

    assert isinstance(legacy, tuple)
    assert len(legacy) == 2
    legacy_out, legacy_weights = legacy
    assert isinstance(with_scores, tuple)
    assert len(with_scores) == 3
    score_out, score_weights, raw_scores = with_scores
    torch.testing.assert_close(score_out, legacy_out)
    torch.testing.assert_close(score_weights, legacy_weights)
    assert raw_scores.shape == legacy_weights.shape
    assert raw_scores[0, :, :, 2].max().item() < -1e20


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
