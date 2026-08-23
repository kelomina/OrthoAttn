"""Run MHDSRA2 quality-improvement ablations across NIAH, JSON, and arithmetic."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mhdsra2_batch_retrieval_quality_smoke import run_quality_smoke  # noqa: E402
from scripts.next_round_benchmark_runner import (  # noqa: E402
    aggregate_model_seed_runs,
    build_task_seed_bundle,
    seed_everything,
)
from scripts.needle_in_haystack_test import (  # noqa: E402
    DEFAULT_NIAH_SPAN_CANDIDATE_FILTER,
    DEFAULT_NIAH_SPAN_LOSS_MODE,
    DEFAULT_NIAH_TEST_BATCHES_PER_DEPTH,
    cleanup_after_oom,
    is_oom_error,
    run_niah_verification_case,
)
from scripts.json_retrieval_test import (  # noqa: E402
    DEFAULT_LOCAL_CONTEXT_MODE,
    DEFAULT_LOCAL_CONTEXT_SIZE,
    run_json_retrieval_generalization_test,
)
from src.dsra.application.arithmetic_emergence_service import (  # noqa: E402
    DEFAULT_TWO_DIGIT_REPLAY_RATIO,
    DEFAULT_TWO_DIGIT_STAGE_LOSS_WEIGHTS,
    TWO_DIGIT_ONLY,
    build_two_digit_diagnostic_grid_payload,
    run_one_two_digit_diagnostic_grid_point,
    select_two_digit_diagnostic_dataset_specs,
    serialize_two_digit_diagnostic_run,
)
from src.dsra.report_utils import ensure_reports_dir, write_json, write_markdown  # noqa: E402


DEFAULT_NIAH_RETRIEVAL_EVIDENCE_LOSS_ALPHA = 0.25
DEFAULT_NIAH_RETRIEVAL_EVIDENCE_RANK_MARGIN = 0.15
DEFAULT_NIAH_RETRIEVAL_EVIDENCE_SCORE_MARGIN = 0.50
DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN = 1
DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_DIRECTION = "right"
BIDIRECTIONAL_NIAH_RETRIEVAL_NEIGHBOR_DIRECTION = "both"
DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SEED_MULTIPLIER = 4
COMPACT_NIAH_RETRIEVAL_MAX_TOKENS = 32
PAIR_AWARE_NIAH_RETRIEVAL_NEIGHBOR_BUDGET_MODE = "pair_aware"
DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET = 1
NIAH_RETRIEVAL_EVIDENCE_KEY_OFFSET = 0
DEFAULT_NIAH_QUERY_EVIDENCE_ALIGNMENT_ALPHA = 0.20
DEFAULT_NIAH_RETRIEVAL_PROJECTION_CONTRASTIVE_ALPHA = 0.20
DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE = 0.10
DEFAULT_NIAH_RETRIEVAL_SPAN_PREDICTOR_ALPHA = 0.35
STRUCTURED_NIAH_SPAN_CANDIDATE_FILTER = "key_value_pair"
PREFERRED_STRUCTURED_NIAH_SPAN_CANDIDATE_FILTER = "prefer_key_value_pair"
MULTI_POSITIVE_NIAH_SPAN_LOSS_MODE = "multi_positive"
DEFAULT_JSON_EVIDENCE_LOSS_WEIGHT = 0.20
DEFAULT_JSON_SLOT_DECODER_LOSS_WEIGHT = 0.35
DEFAULT_JSON_SLOT_DECODER_LOGIT_BIAS = 4.0
DEFAULT_JSON_EVIDENCE_HINT_WEIGHT = 0.0
DEFAULT_REPORT_NAME = "mhdsra2_evidence_retrieval_ablation"
NIAH_SPAN_CANDIDATE_DIAGNOSTIC_FIELDS = (
    "mean_span_valid_candidate_count",
    "mean_span_raw_candidate_count",
    "mean_span_pair_candidate_count",
    "span_pair_available_rate",
    "span_filter_fallback_rate",
    "span_selected_pair_rate",
    "span_target_pair_candidate_rate",
    "span_target_value_page_candidate_rate",
    "span_target_pair_page_candidate_rate",
    "span_target_value_top_token_rate",
    "span_target_pair_top_token_rate",
    "span_target_value_seed_token_rate",
    "span_target_pair_seed_token_rate",
)


GROUP_CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": {
        "description": "Current default MHDSRA2 configuration.",
        "override": {},
        "capabilities": {
            "query_pooling": "mean",
            "retrieval_max_tokens": 128,
            "neighbor_span": 0,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "retrieval_evidence_target_offset": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
            "query_evidence_alignment_alpha": 0.0,
            "retrieval_projection_contrastive_alpha": 0.0,
            "retrieval_projection_temperature": DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
            "retrieval_span_predictor_alpha": 0.0,
            "retrieval_span_structure_features": False,
            "niah_span_candidate_filter": DEFAULT_NIAH_SPAN_CANDIDATE_FILTER,
            "niah_span_loss_mode": DEFAULT_NIAH_SPAN_LOSS_MODE,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "evidence_hit_supervision": {
        "description": "Train-only auxiliary evidence-hit supervision for NIAH/JSON.",
        "override": {},
        "capabilities": {
            "query_pooling": "mean",
            "retrieval_max_tokens": 128,
            "neighbor_span": 0,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_LOSS_ALPHA,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "retrieval_evidence_target_offset": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
            "query_evidence_alignment_alpha": 0.0,
            "retrieval_projection_contrastive_alpha": 0.0,
            "retrieval_projection_temperature": DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
            "retrieval_span_structure_features": False,
            "json_evidence_loss_weight": DEFAULT_JSON_EVIDENCE_LOSS_WEIGHT,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "learned_retrieval_gate": {
        "description": "Enable a zero-initialized learned retrieval gate adapter.",
        "override": {"retrieval_quality_gate_adapter": True},
        "capabilities": {
            "query_pooling": "mean",
            "retrieval_max_tokens": 128,
            "neighbor_span": 0,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": True,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "retrieval_evidence_target_offset": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
            "query_evidence_alignment_alpha": 0.0,
            "retrieval_projection_contrastive_alpha": 0.0,
            "retrieval_projection_temperature": DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
            "retrieval_span_structure_features": False,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "evidence_plus_gate": {
        "description": "Combine train-only evidence supervision with the learned gate adapter.",
        "override": {"retrieval_quality_gate_adapter": True},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": 0,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": True,
            "retrieval_evidence_loss_alpha": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_LOSS_ALPHA,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "retrieval_evidence_target_offset": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
            "query_evidence_alignment_alpha": 0.0,
            "retrieval_projection_contrastive_alpha": 0.0,
            "retrieval_projection_temperature": DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
            "json_evidence_loss_weight": DEFAULT_JSON_EVIDENCE_LOSS_WEIGHT,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "evidence_rank_margin": {
        "description": (
            "Train NIAH evidence retrieval with an extra margin objective so the "
            "known evidence candidate must outrank the strongest negative candidate."
        ),
        "override": {},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": 0,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_LOSS_ALPHA,
            "retrieval_evidence_rank_margin": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_RANK_MARGIN,
            "retrieval_evidence_score_margin": 0.0,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "evidence_rank_margin_needle_copy": {
        "description": (
            "Train the NIAH evidence rank-margin objective, then evaluate with "
            "retrieval-candidate needle copy readout."
        ),
        "override": {},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": 0,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_LOSS_ALPHA,
            "retrieval_evidence_rank_margin": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_RANK_MARGIN,
            "retrieval_evidence_score_margin": 0.0,
            "niah_readout_mode": "needle_copy",
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "evidence_score_margin": {
        "description": (
            "Train NIAH evidence retrieval with a margin objective on retrieval raw "
            "scores before softmax, so the known evidence key must score above "
            "the strongest valid negative candidate."
        ),
        "override": {},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": 0,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_LOSS_ALPHA,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_SCORE_MARGIN,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "evidence_score_margin_needle_copy": {
        "description": (
            "Train the NIAH raw-score evidence margin objective, then evaluate with "
            "retrieval-candidate needle copy readout."
        ),
        "override": {},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": 0,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_LOSS_ALPHA,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_SCORE_MARGIN,
            "niah_readout_mode": "needle_copy",
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "retrieval_neighbor_span": {
        "description": (
            "Expand each retrieved token with its right neighbor so key-value "
            "patterns can return the value next to a matched key."
        ),
        "override": {"retrieval_neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "retrieval_neighbor_span_needle_copy": {
        "description": (
            "Enable right-neighbor retrieval candidates and evaluate NIAH with "
            "retrieval-candidate needle copy readout."
        ),
        "override": {"retrieval_neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "niah_readout_mode": "needle_copy",
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "evidence_score_margin_neighbor_span_needle_copy": {
        "description": (
            "Combine raw-score evidence supervision with right-neighbor retrieval "
            "candidate expansion, then evaluate with needle copy readout."
        ),
        "override": {"retrieval_neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_LOSS_ALPHA,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_SCORE_MARGIN,
            "niah_readout_mode": "needle_copy",
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "retrieval_neighbor_span_pair_copy": {
        "description": (
            "Enable right-neighbor retrieval candidates and evaluate NIAH with "
            "a key/value pair-aware needle copy readout."
        ),
        "override": {"retrieval_neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "niah_readout_mode": "needle_pair_copy",
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "evidence_score_margin_neighbor_span_pair_copy": {
        "description": (
            "Combine raw-score evidence supervision with right-neighbor retrieval "
            "candidate expansion, then evaluate with key/value pair-aware copy readout."
        ),
        "override": {"retrieval_neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_LOSS_ALPHA,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_SCORE_MARGIN,
            "niah_readout_mode": "needle_pair_copy",
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "evidence_key_score_margin_neighbor_span_pair_copy": {
        "description": (
            "Train NIAH raw-score evidence supervision against the needle key, "
            "expand right-neighbor candidates, then evaluate with key/value pair-aware "
            "copy readout."
        ),
        "override": {"retrieval_neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_LOSS_ALPHA,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_SCORE_MARGIN,
            "retrieval_evidence_target_offset": NIAH_RETRIEVAL_EVIDENCE_KEY_OFFSET,
            "niah_readout_mode": "needle_pair_copy",
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "query_evidence_alignment_pair_copy": {
        "description": (
            "Add an opt-in query/evidence representation alignment loss, keep "
            "right-neighbor retrieval candidates, then evaluate with key/value "
            "pair-aware copy readout."
        ),
        "override": {"retrieval_neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "retrieval_evidence_target_offset": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
            "query_evidence_alignment_alpha": DEFAULT_NIAH_QUERY_EVIDENCE_ALIGNMENT_ALPHA,
            "niah_readout_mode": "needle_pair_copy",
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "retrieval_projection_contrastive": {
        "description": (
            "Train an opt-in contrastive loss directly on selected retrieval q/k "
            "projections so the known evidence key outranks retrieved negatives."
        ),
        "override": {},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": 0,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "retrieval_evidence_target_offset": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
            "query_evidence_alignment_alpha": 0.0,
            "retrieval_projection_contrastive_alpha": (
                DEFAULT_NIAH_RETRIEVAL_PROJECTION_CONTRASTIVE_ALPHA
            ),
            "retrieval_projection_temperature": DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
            "niah_readout_mode": "model",
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "retrieval_projection_pair_copy": {
        "description": (
            "Train direct retrieval q/k projection supervision, expand right-neighbor "
            "candidates, then evaluate with key/value pair-aware copy readout."
        ),
        "override": {"retrieval_neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "retrieval_evidence_target_offset": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
            "query_evidence_alignment_alpha": 0.0,
            "retrieval_projection_contrastive_alpha": (
                DEFAULT_NIAH_RETRIEVAL_PROJECTION_CONTRASTIVE_ALPHA
            ),
            "retrieval_projection_temperature": DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
            "niah_readout_mode": "needle_pair_copy",
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "retrieval_span_predictor": {
        "description": (
            "Train a default-off NIAH span predictor that scores selected retrieval "
            "candidates and evaluates by copying from the predicted source span."
        ),
        "override": {"retrieval_neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "retrieval_evidence_target_offset": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
            "query_evidence_alignment_alpha": 0.0,
            "retrieval_projection_contrastive_alpha": 0.0,
            "retrieval_projection_temperature": DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
            "retrieval_span_predictor_alpha": DEFAULT_NIAH_RETRIEVAL_SPAN_PREDICTOR_ALPHA,
            "niah_readout_mode": "span_predictor",
            "niah_span_candidate_filter": DEFAULT_NIAH_SPAN_CANDIDATE_FILTER,
            "niah_span_loss_mode": DEFAULT_NIAH_SPAN_LOSS_MODE,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "retrieval_structured_span_predictor": {
        "description": (
            "Train the default-off NIAH span predictor, but restrict its candidate "
            "pool to selected retrieval key/value pairs that were both recalled."
        ),
        "override": {"retrieval_neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "retrieval_evidence_target_offset": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
            "query_evidence_alignment_alpha": 0.0,
            "retrieval_projection_contrastive_alpha": 0.0,
            "retrieval_projection_temperature": DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
            "retrieval_span_predictor_alpha": DEFAULT_NIAH_RETRIEVAL_SPAN_PREDICTOR_ALPHA,
            "niah_readout_mode": "span_predictor",
            "niah_span_candidate_filter": STRUCTURED_NIAH_SPAN_CANDIDATE_FILTER,
            "niah_span_loss_mode": DEFAULT_NIAH_SPAN_LOSS_MODE,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "retrieval_prefer_structured_span_predictor": {
        "description": (
            "Train the default-off NIAH span predictor, prefer selected retrieval "
            "key/value pairs when available, and fall back to the original selected "
            "candidate pool otherwise."
        ),
        "override": {"retrieval_neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "retrieval_evidence_target_offset": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
            "query_evidence_alignment_alpha": 0.0,
            "retrieval_projection_contrastive_alpha": 0.0,
            "retrieval_projection_temperature": DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
            "retrieval_span_predictor_alpha": DEFAULT_NIAH_RETRIEVAL_SPAN_PREDICTOR_ALPHA,
            "niah_readout_mode": "span_predictor",
            "niah_span_candidate_filter": PREFERRED_STRUCTURED_NIAH_SPAN_CANDIDATE_FILTER,
            "niah_span_loss_mode": DEFAULT_NIAH_SPAN_LOSS_MODE,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "retrieval_multi_positive_span_predictor": {
        "description": (
            "Train the default-off NIAH span predictor with a multi-positive loss "
            "so every selected candidate that copies the right answer shares "
            "positive probability mass."
        ),
        "override": {"retrieval_neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "retrieval_evidence_target_offset": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
            "query_evidence_alignment_alpha": 0.0,
            "retrieval_projection_contrastive_alpha": 0.0,
            "retrieval_projection_temperature": DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
            "retrieval_span_predictor_alpha": DEFAULT_NIAH_RETRIEVAL_SPAN_PREDICTOR_ALPHA,
            "niah_readout_mode": "span_predictor",
            "niah_span_candidate_filter": PREFERRED_STRUCTURED_NIAH_SPAN_CANDIDATE_FILTER,
            "niah_span_loss_mode": MULTI_POSITIVE_NIAH_SPAN_LOSS_MODE,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "retrieval_bidirectional_structured_span_predictor": {
        "description": (
            "Train the default-off NIAH span predictor with bidirectional neighbor "
            "candidate expansion so selected values can bring their left-side key "
            "into the eval candidate set."
        ),
        "override": {
            "retrieval_neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "retrieval_neighbor_direction": BIDIRECTIONAL_NIAH_RETRIEVAL_NEIGHBOR_DIRECTION,
        },
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "neighbor_direction": BIDIRECTIONAL_NIAH_RETRIEVAL_NEIGHBOR_DIRECTION,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "retrieval_evidence_target_offset": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
            "query_evidence_alignment_alpha": 0.0,
            "retrieval_projection_contrastive_alpha": 0.0,
            "retrieval_projection_temperature": DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
            "retrieval_span_predictor_alpha": DEFAULT_NIAH_RETRIEVAL_SPAN_PREDICTOR_ALPHA,
            "niah_readout_mode": "span_predictor",
            "niah_span_candidate_filter": PREFERRED_STRUCTURED_NIAH_SPAN_CANDIDATE_FILTER,
            "niah_span_loss_mode": MULTI_POSITIVE_NIAH_SPAN_LOSS_MODE,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "retrieval_page_local_neighbor_span_predictor": {
        "description": (
            "Train the default-off NIAH span predictor with page-local neighbor "
            "materialization: retrieve extra in-page seed tokens before neighbor "
            "expansion so adjacent key/value evidence can survive token top-k."
        ),
        "override": {
            "retrieval_neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "retrieval_neighbor_direction": BIDIRECTIONAL_NIAH_RETRIEVAL_NEIGHBOR_DIRECTION,
            "retrieval_neighbor_seed_multiplier": (
                DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SEED_MULTIPLIER
            ),
        },
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "neighbor_direction": BIDIRECTIONAL_NIAH_RETRIEVAL_NEIGHBOR_DIRECTION,
            "neighbor_seed_multiplier": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SEED_MULTIPLIER,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "retrieval_evidence_target_offset": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
            "query_evidence_alignment_alpha": 0.0,
            "retrieval_projection_contrastive_alpha": 0.0,
            "retrieval_projection_temperature": DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
            "retrieval_span_predictor_alpha": DEFAULT_NIAH_RETRIEVAL_SPAN_PREDICTOR_ALPHA,
            "niah_readout_mode": "span_predictor",
            "niah_span_candidate_filter": PREFERRED_STRUCTURED_NIAH_SPAN_CANDIDATE_FILTER,
            "niah_span_loss_mode": MULTI_POSITIVE_NIAH_SPAN_LOSS_MODE,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "retrieval_structured_feature_span_predictor": {
        "description": (
            "Train the default-off NIAH span predictor with page-local neighbor "
            "materialization plus explicit pair/source structure features, so the "
            "readout head can distinguish complete key/value candidates from "
            "loose high-score retrieval hits."
        ),
        "override": {
            "retrieval_neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "retrieval_neighbor_direction": BIDIRECTIONAL_NIAH_RETRIEVAL_NEIGHBOR_DIRECTION,
            "retrieval_neighbor_seed_multiplier": (
                DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SEED_MULTIPLIER
            ),
        },
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "neighbor_direction": BIDIRECTIONAL_NIAH_RETRIEVAL_NEIGHBOR_DIRECTION,
            "neighbor_seed_multiplier": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SEED_MULTIPLIER,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "retrieval_evidence_target_offset": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
            "query_evidence_alignment_alpha": 0.0,
            "retrieval_projection_contrastive_alpha": 0.0,
            "retrieval_projection_temperature": DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
            "retrieval_span_predictor_alpha": DEFAULT_NIAH_RETRIEVAL_SPAN_PREDICTOR_ALPHA,
            "retrieval_span_structure_features": True,
            "niah_readout_mode": "span_predictor",
            "niah_span_candidate_filter": PREFERRED_STRUCTURED_NIAH_SPAN_CANDIDATE_FILTER,
            "niah_span_loss_mode": MULTI_POSITIVE_NIAH_SPAN_LOSS_MODE,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "retrieval_compact_page_local_span_predictor": {
        "description": (
            "Train the default-off NIAH span predictor with page-local neighbor "
            "materialization but a smaller external retrieval token budget, so "
            "the selected candidate table is less noisy."
        ),
        "override": {
            "retrieval_neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "retrieval_neighbor_direction": BIDIRECTIONAL_NIAH_RETRIEVAL_NEIGHBOR_DIRECTION,
            "retrieval_neighbor_seed_multiplier": (
                DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SEED_MULTIPLIER
            ),
            "retrieval_max_tokens": COMPACT_NIAH_RETRIEVAL_MAX_TOKENS,
        },
        "capabilities": {
            "query_pooling": "mean",
            "retrieval_max_tokens": COMPACT_NIAH_RETRIEVAL_MAX_TOKENS,
            "neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "neighbor_direction": BIDIRECTIONAL_NIAH_RETRIEVAL_NEIGHBOR_DIRECTION,
            "neighbor_seed_multiplier": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SEED_MULTIPLIER,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "retrieval_evidence_target_offset": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
            "query_evidence_alignment_alpha": 0.0,
            "retrieval_projection_contrastive_alpha": 0.0,
            "retrieval_projection_temperature": DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
            "retrieval_span_predictor_alpha": DEFAULT_NIAH_RETRIEVAL_SPAN_PREDICTOR_ALPHA,
            "niah_readout_mode": "span_predictor",
            "niah_span_candidate_filter": PREFERRED_STRUCTURED_NIAH_SPAN_CANDIDATE_FILTER,
            "niah_span_loss_mode": MULTI_POSITIVE_NIAH_SPAN_LOSS_MODE,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "retrieval_pair_aware_page_local_span_predictor": {
        "description": (
            "Train the default-off NIAH span predictor with page-local neighbor "
            "materialization and a pair-aware neighbor budget, so adjacent "
            "key/value candidates are materialized without unbounded noise."
        ),
        "override": {
            "retrieval_neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "retrieval_neighbor_direction": BIDIRECTIONAL_NIAH_RETRIEVAL_NEIGHBOR_DIRECTION,
            "retrieval_neighbor_seed_multiplier": (
                DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SEED_MULTIPLIER
            ),
            "retrieval_neighbor_budget_mode": (
                PAIR_AWARE_NIAH_RETRIEVAL_NEIGHBOR_BUDGET_MODE
            ),
        },
        "capabilities": {
            "query_pooling": "mean",
            "retrieval_max_tokens": 128,
            "neighbor_span": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SPAN,
            "neighbor_direction": BIDIRECTIONAL_NIAH_RETRIEVAL_NEIGHBOR_DIRECTION,
            "neighbor_seed_multiplier": DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_SEED_MULTIPLIER,
            "neighbor_budget_mode": PAIR_AWARE_NIAH_RETRIEVAL_NEIGHBOR_BUDGET_MODE,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "retrieval_evidence_target_offset": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
            "query_evidence_alignment_alpha": 0.0,
            "retrieval_projection_contrastive_alpha": 0.0,
            "retrieval_projection_temperature": DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
            "retrieval_span_predictor_alpha": DEFAULT_NIAH_RETRIEVAL_SPAN_PREDICTOR_ALPHA,
            "niah_readout_mode": "span_predictor",
            "niah_span_candidate_filter": PREFERRED_STRUCTURED_NIAH_SPAN_CANDIDATE_FILTER,
            "niah_span_loss_mode": MULTI_POSITIVE_NIAH_SPAN_LOSS_MODE,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "slot_readout_bias": {
        "description": (
            "Train the existing JSON slot decoder and apply its generation-time "
            "byte-level readout bias."
        ),
        "override": {},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": 0,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": DEFAULT_JSON_SLOT_DECODER_LOSS_WEIGHT,
            "json_slot_decoder_logit_bias": DEFAULT_JSON_SLOT_DECODER_LOGIT_BIAS,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "evidence_slot_readout": {
        "description": (
            "Train evidence-window supervision plus the existing JSON slot decoder, "
            "then use the slot readout bias during generation."
        ),
        "override": {},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": 0,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "json_evidence_loss_weight": DEFAULT_JSON_EVIDENCE_LOSS_WEIGHT,
            "json_slot_decoder_loss_weight": DEFAULT_JSON_SLOT_DECODER_LOSS_WEIGHT,
            "json_slot_decoder_logit_bias": DEFAULT_JSON_SLOT_DECODER_LOGIT_BIAS,
            "json_evidence_hint_weight": DEFAULT_JSON_EVIDENCE_HINT_WEIGHT,
            "json_generation_readout_mode": "model",
        },
    },
    "extract_compose_readout": {
        "description": (
            "Use the existing evidence-window decoder plus deterministic "
            "extract-then-compose answer readout for JSON generation metrics."
        ),
        "override": {},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": 0,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "json_evidence_loss_weight": DEFAULT_JSON_EVIDENCE_LOSS_WEIGHT,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "extract_then_compose",
        },
    },
    "needle_copy_readout": {
        "description": (
            "Evaluate NIAH with an explicit retrieval-candidate needle copy readout."
        ),
        "override": {},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": 0,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "niah_readout_mode": "needle_copy",
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "evidence_needle_copy_readout": {
        "description": (
            "Train NIAH evidence-hit supervision, then evaluate with retrieval-candidate "
            "needle copy readout."
        ),
        "override": {},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": 0,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": DEFAULT_NIAH_RETRIEVAL_EVIDENCE_LOSS_ALPHA,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "niah_readout_mode": "needle_copy",
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "retrieval_query_pooling": {
        "description": "Use max-token query pooling for external paged retrieval.",
        "override": {"retrieval_query_pooling": "max_token"},
        "capabilities": {
            "query_pooling": "max_token",
            "neighbor_span": 0,
            "gate_quality_bias": 0.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "retrieval_gate_quality": {
        "description": "Bias retrieval gate only when retrieval candidates exist.",
        "override": {"retrieval_quality_gate_bias": 2.0},
        "capabilities": {
            "query_pooling": "mean",
            "neighbor_span": 0,
            "gate_quality_bias": 2.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
    "combined": {
        "description": "Enable max-token query pooling and retrieval-quality gate bias.",
        "override": {
            "retrieval_query_pooling": "max_token",
            "retrieval_quality_gate_bias": 2.0,
        },
        "capabilities": {
            "query_pooling": "max_token",
            "neighbor_span": 0,
            "gate_quality_bias": 2.0,
            "learned_retrieval_gate": False,
            "retrieval_evidence_loss_alpha": 0.0,
            "retrieval_evidence_rank_margin": 0.0,
            "retrieval_evidence_score_margin": 0.0,
            "json_evidence_loss_weight": 0.0,
            "json_slot_decoder_loss_weight": 0.0,
            "json_slot_decoder_logit_bias": 0.0,
            "json_evidence_hint_weight": 0.0,
            "json_generation_readout_mode": "model",
        },
    },
}

DEFAULT_GROUPS = (
    "baseline",
    "evidence_hit_supervision",
    "learned_retrieval_gate",
    "evidence_plus_gate",
)
DEFAULT_TASKS = ("smoke", "niah", "json")
ALLOWED_TASKS = ("smoke", "niah", "json", "two_digit")


def parse_csv_ints(value: str | Iterable[int]) -> tuple[int, ...]:
    """Parse comma-separated integers used by the ablation CLI and tests."""
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise argparse.ArgumentTypeError("expected at least one integer")
        return tuple(int(part) for part in parts)
    return tuple(int(item) for item in value)


def parse_csv_floats(value: str | Iterable[float]) -> tuple[float, ...]:
    """Parse comma-separated floats used by the ablation CLI and tests."""
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise argparse.ArgumentTypeError("expected at least one float")
        return tuple(float(part) for part in parts)
    return tuple(float(item) for item in value)


def parse_csv_strings(value: str | Iterable[str]) -> tuple[str, ...]:
    """Parse comma-separated labels used by the ablation CLI and tests."""
    if isinstance(value, str):
        parts = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        parts = tuple(str(part).strip() for part in value if str(part).strip())
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one value")
    return parts


def resolve_device(value: str) -> torch.device:
    """Resolve the experiment device with the project cuda:0 policy."""
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        value = "cuda:0"
    device = torch.device(value)
    if device.type == "cuda" and device.index not in (0, None):
        raise ValueError("Only cuda:0 is supported for DSRA experiments.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return torch.device("cuda:0") if device.type == "cuda" else device


def group_override(group_name: str) -> dict[str, Any]:
    """Return a copy of the MHDSRA2 override for one ablation group."""
    if group_name not in GROUP_CONFIGS:
        allowed = ", ".join(GROUP_CONFIGS)
        raise ValueError(f"unknown ablation group {group_name!r}; allowed: {allowed}")
    return dict(GROUP_CONFIGS[group_name]["override"])


def group_capability(group_name: str, key: str, default: Any = None) -> Any:
    """Return one non-model capability flag for an ablation group."""
    if group_name not in GROUP_CONFIGS:
        allowed = ", ".join(GROUP_CONFIGS)
        raise ValueError(f"unknown ablation group {group_name!r}; allowed: {allowed}")
    capabilities = GROUP_CONFIGS[group_name].get("capabilities", {})
    if not isinstance(capabilities, Mapping):
        return default
    return capabilities.get(key, default)


def group_uses_niah_retrieval(group_name: str) -> bool:
    """Return whether this NIAH group should enable external paged retrieval.

    中文说明:
    - 调用方 / Called by: `build_run_rows`, `run_niah_rows`, `run_niah_row`
    - 调用对象 / Calls: `group_capability`, `group_override`
    - 作用 / Purpose: baseline 保持旧 NIAH 行为；只有 evidence、learned gate 或显式
      retrieval override 组打开外部分页召回，避免报告里看似测 retrieval、实际未启用。
    - 返回 / Returns: bool，是否给 `run_niah_verification_case(use_retrieval=...)` 传 True。
    - 错误处理 / Error handling: 未知 group 由现有 helper 抛 `ValueError`。
    - 副作用 / Side effects: 无。

    English documentation:
    Function name:
        group_uses_niah_retrieval
    Purpose:
        Keep the baseline NIAH path unchanged while ensuring retrieval ablation
        groups actually exercise external paged retrieval.
    """
    if float(group_capability(group_name, "retrieval_evidence_loss_alpha", 0.0)) > 0.0:
        return True
    if str(group_capability(group_name, "niah_readout_mode", "model")) in {
        "needle_copy",
        "needle_pair_copy",
    }:
        return True
    if bool(group_capability(group_name, "learned_retrieval_gate", False)):
        return True
    if float(group_capability(group_name, "retrieval_projection_contrastive_alpha", 0.0)) > 0.0:
        return True
    if float(group_capability(group_name, "retrieval_span_predictor_alpha", 0.0)) > 0.0:
        return True
    override = group_override(group_name)
    retrieval_keys = {
        "retrieval_query_pooling",
        "retrieval_neighbor_span",
        "retrieval_neighbor_direction",
        "retrieval_neighbor_seed_multiplier",
        "retrieval_neighbor_budget_mode",
        "retrieval_max_tokens",
        "retrieval_quality_gate_bias",
        "retrieval_quality_gate_adapter",
    }
    return any(key in override for key in retrieval_keys)


def json_group_capabilities(group_name: str) -> dict[str, Any]:
    """Return JSON-only readout and evidence switches for one ablation group.

    中文说明:
    - 调用方 / Called by: `build_run_rows`, `run_json_section`, `run_json_row`
    - 调用对象 / Calls: `group_capability`
    - 作用 / Purpose: 把 JSON 任务里的证据窗口监督、slot decoder 训练、
      生成期 slot logit bias 和 evidence hint 参数集中展开，避免在多个调用点漏传。
    - 返回 / Returns: 可直接写入 row config 并传给
      `run_json_retrieval_generalization_test()` 的 float 参数。
    - 错误处理 / Error handling: 未知 group 由 `group_capability()` 抛 `ValueError`。
    - 副作用 / Side effects: 无，不读取 validation/test 信号。

    English documentation:
    Function name:
        json_group_capabilities
    Purpose:
        Centralize JSON retrieval readout/evidence capability expansion so the
        ablation runner records and passes the same train-time switches.
    """
    return {
        "evidence_loss_weight": float(
            group_capability(group_name, "json_evidence_loss_weight", 0.0)
        ),
        "slot_decoder_loss_weight": float(
            group_capability(group_name, "json_slot_decoder_loss_weight", 0.0)
        ),
        "slot_decoder_logit_bias": float(
            group_capability(group_name, "json_slot_decoder_logit_bias", 0.0)
        ),
        "evidence_hint_weight": float(
            group_capability(group_name, "json_evidence_hint_weight", 0.0)
        ),
        "generation_readout_mode": str(
            group_capability(group_name, "json_generation_readout_mode", "model")
        ),
    }


def _optional_float(value: Any) -> float | None:
    """Convert scalar-like metric values to floats without forcing missing values."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_slot_collision_diagnostics(aux: Mapping[str, Any] | None) -> dict[str, Any]:
    """Summarize slot usage concentration as a diagnostic-only collision signal.

    这里不把 slot 碰撞诊断接入训练 loss，只把可用的 slot usage/confidence
    压缩成报告字段。有效槽越少、top1 占比越高，越说明记忆可能过度集中到少数槽位。
    """
    if not aux:
        return {
            "available": False,
            "reason": "no_mhdsra2_aux",
        }
    usage = aux.get("slot_usage")
    if not isinstance(usage, torch.Tensor):
        return {
            "available": False,
            "reason": "slot_usage_missing",
        }
    usage32 = usage.detach().to(dtype=torch.float32).cpu()
    usage_sum = usage32.sum()
    if float(usage_sum.item()) <= 0.0:
        return {
            "available": True,
            "slot_usage_sum": 0.0,
            "effective_slot_count": 0.0,
            "top1_usage_share": 0.0,
            "collision_risk": "empty",
        }
    probs = usage32.flatten() / usage_sum.clamp_min(1e-12)
    effective_slots = float(1.0 / probs.square().sum().clamp_min(1e-12).item())
    top1_share = float(probs.max().item())
    slot_count = int(probs.numel())
    if top1_share >= 0.50 or effective_slots < max(1.0, slot_count * 0.10):
        collision_risk = "high"
    elif top1_share >= 0.25 or effective_slots < max(1.0, slot_count * 0.25):
        collision_risk = "medium"
    else:
        collision_risk = "low"
    confidence = aux.get("slot_confidence")
    confidence_mean = (
        float(confidence.detach().to(dtype=torch.float32).mean().cpu().item())
        if isinstance(confidence, torch.Tensor)
        else None
    )
    return {
        "available": True,
        "slot_count": slot_count,
        "slot_usage_sum": float(usage_sum.item()),
        "effective_slot_count": effective_slots,
        "top1_usage_share": top1_share,
        "slot_confidence_mean": confidence_mean,
        "collision_risk": collision_risk,
    }


def row_key(row: Mapping[str, Any]) -> str:
    """Build a stable checkpoint key for one planned or completed row."""
    payload = {
        "group": row.get("group"),
        "task": row.get("task"),
        "seed": row.get("seed"),
        "config": row.get("config", {}),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def load_checkpoint_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    """Read completed JSONL checkpoint rows keyed by task/group/seed/config."""
    if path is None or not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row_key(row)] = row
    return rows


def append_checkpoint_row(path: Path | None, row: Mapping[str, Any]) -> None:
    """Append one completed row to JSONL checkpoint immediately."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_run_rows(
    *,
    groups: Sequence[str],
    tasks: Sequence[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Build dry-run rows so planned experiment scope is inspectable."""
    rows: list[dict[str, Any]] = []
    if "smoke" in tasks:
        rows.append(
            {
                "group": "shared",
                "task": "smoke",
                "seed": args.smoke_seed,
                "status": "planned",
                "config": {
                    "batch_sizes": tuple(int(value) for value in args.smoke_batch_sizes),
                    "tokens": tuple(int(value) for value in args.smoke_tokens),
                    "page_size": int(args.smoke_page_size),
                    "top_pages": int(args.smoke_top_pages),
                    "max_tokens": int(args.smoke_max_tokens),
                },
                "validation_metrics": {},
                "test_metrics": {},
            }
        )
    if "niah" in tasks:
        for group_name in groups:
            for seq_len in args.niah_seq_lengths:
                for seed in args.niah_seeds:
                    rows.append(
                        {
                            "group": group_name,
                            "task": "niah",
                            "seed": seed,
                            "status": "planned",
                            "config": {
                                "seq_len": int(seq_len),
                                "dim": int(args.niah_dim),
                                "num_layers": int(args.niah_layers),
                                "slots": int(args.niah_slots),
                                "read_topk": int(args.niah_read_topk),
                                "epochs": int(args.niah_epochs),
                                "eval_batches_per_depth": int(
                                    args.niah_eval_batches_per_depth
                                ),
                                "test_batches_per_depth": int(
                                    args.niah_test_batches_per_depth
                                ),
                                "retrieval_evidence_loss_alpha": float(
                                    group_capability(
                                        group_name,
                                        "retrieval_evidence_loss_alpha",
                                        0.0,
                                    )
                                ),
                                "retrieval_evidence_rank_margin": float(
                                    group_capability(
                                        group_name,
                                        "retrieval_evidence_rank_margin",
                                        0.0,
                                    )
                                ),
                                "retrieval_evidence_score_margin": float(
                                    group_capability(
                                        group_name,
                                        "retrieval_evidence_score_margin",
                                        0.0,
                                    )
                                ),
                                "retrieval_evidence_target_offset": int(
                                    group_capability(
                                        group_name,
                                        "retrieval_evidence_target_offset",
                                        DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
                                    )
                                ),
                                "query_evidence_alignment_alpha": float(
                                    group_capability(
                                        group_name,
                                        "query_evidence_alignment_alpha",
                                        0.0,
                                    )
                                ),
                                "retrieval_projection_contrastive_alpha": float(
                                    group_capability(
                                        group_name,
                                        "retrieval_projection_contrastive_alpha",
                                        0.0,
                                    )
                                ),
                                "retrieval_projection_temperature": float(
                                    group_capability(
                                        group_name,
                                        "retrieval_projection_temperature",
                                        DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
                                    )
                                ),
                                "retrieval_span_predictor_alpha": float(
                                    group_capability(
                                        group_name,
                                        "retrieval_span_predictor_alpha",
                                        0.0,
                                    )
                                ),
                                "retrieval_span_structure_features": bool(
                                    group_capability(
                                        group_name,
                                        "retrieval_span_structure_features",
                                        False,
                                    )
                                ),
                                "niah_span_candidate_filter": str(
                                    group_capability(
                                        group_name,
                                        "niah_span_candidate_filter",
                                        DEFAULT_NIAH_SPAN_CANDIDATE_FILTER,
                                    )
                                ),
                                "niah_span_loss_mode": str(
                                    group_capability(
                                        group_name,
                                        "niah_span_loss_mode",
                                        DEFAULT_NIAH_SPAN_LOSS_MODE,
                                    )
                                ),
                                "retrieval_neighbor_span": int(
                                    group_capability(group_name, "neighbor_span", 0)
                                ),
                                "retrieval_neighbor_direction": str(
                                    group_capability(
                                        group_name,
                                        "neighbor_direction",
                                        DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_DIRECTION,
                                    )
                                ),
                                "retrieval_neighbor_seed_multiplier": int(
                                    group_capability(
                                        group_name,
                                        "neighbor_seed_multiplier",
                                        1,
                                    )
                                ),
                                "retrieval_neighbor_budget_mode": str(
                                    group_capability(
                                        group_name,
                                        "neighbor_budget_mode",
                                        "unbounded",
                                    )
                                ),
                                "retrieval_max_tokens": int(
                                    group_capability(
                                        group_name,
                                        "retrieval_max_tokens",
                                        128,
                                    )
                                ),
                                "niah_readout_mode": str(
                                    group_capability(
                                        group_name,
                                        "niah_readout_mode",
                                        "model",
                                    )
                                ),
                                "use_retrieval": group_uses_niah_retrieval(group_name),
                                "mhdsra2_config_override": group_override(group_name),
                            },
                            "validation_metrics": {},
                            "test_metrics": {},
                        }
                    )
    if "json" in tasks:
        for group_name in groups:
            for seed_root in args.json_task_seed_roots:
                seed_bundle = build_task_seed_bundle(seed_root)
                json_capabilities = json_group_capabilities(group_name)
                rows.append(
                    {
                        "group": group_name,
                        "task": "json",
                        "seed": seed_root,
                        "status": "planned",
                        "config": {
                            "epochs": int(args.json_epochs),
                            "dim": int(args.json_dim),
                            "slots": int(args.json_slots),
                            "read_topk": int(args.json_read_topk),
                            "chunk_size": int(args.json_chunk_size),
                            "train_dataset_size": int(args.json_train_dataset_size),
                            "validation_dataset_size": int(
                                args.json_validation_dataset_size
                            ),
                            "test_dataset_size": int(args.json_test_dataset_size),
                            "distractor_records_per_case": int(
                                args.json_distractor_records_per_case
                            ),
                            "answer_template_mode": str(args.json_answer_template_mode),
                            "generalization_score_mode": "generation",
                            **json_capabilities,
                            "seed_bundle": seed_bundle,
                            "mhdsra2_config_override": group_override(group_name),
                        },
                        "validation_metrics": {},
                        "test_metrics": {},
                    }
                )
    if "two_digit" in tasks:
        for seed in args.two_digit_seeds:
            for num_layers in args.two_digit_layers:
                for max_steps in args.two_digit_steps:
                    for learning_rate in args.two_digit_learning_rates:
                        for training_strategy in args.two_digit_strategies:
                            rows.append(
                                {
                                    "group": "baseline_holdout",
                                    "task": "two_digit",
                                    "seed": seed,
                                    "status": "planned",
                                    "config": {
                                        "dataset": TWO_DIGIT_ONLY,
                                        "training_strategy": training_strategy,
                                        "learning_rate": float(learning_rate),
                                        "max_steps_per_stage": int(max_steps),
                                        "num_layers": int(num_layers),
                                    },
                                    "validation_metrics": {},
                                    "test_metrics": {},
                                }
                            )
    return rows


def run_smoke_section(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Run the shared retrieval quality smoke before trained-task ablations."""
    started_at = time.perf_counter()
    payload = run_quality_smoke(
        batch_sizes=args.smoke_batch_sizes,
        tokens=args.smoke_tokens,
        page_size=args.smoke_page_size,
        top_pages=args.smoke_top_pages,
        max_tokens=args.smoke_max_tokens,
        device=device,
        seed=args.smoke_seed,
    )
    elapsed_sec = time.perf_counter() - started_at
    row = {
        "group": "shared",
        "task": "smoke",
        "seed": int(args.smoke_seed),
        "status": "passed" if payload["summary"]["passed"] else "failed",
        "config": {
            "batch_sizes": tuple(int(value) for value in args.smoke_batch_sizes),
            "tokens": tuple(int(value) for value in args.smoke_tokens),
            "page_size": int(args.smoke_page_size),
            "top_pages": int(args.smoke_top_pages),
            "max_tokens": int(args.smoke_max_tokens),
        },
        "validation_metrics": {
            "passed": payload["summary"]["passed"],
            "no_cross_sample_leak": payload["summary"]["no_cross_sample_leak"],
            "no_future_leak": payload["summary"]["no_future_leak"],
            "all_batch_loop_positions_match": payload["summary"][
                "all_batch_loop_positions_match"
            ],
        },
        "test_metrics": {},
        "elapsed_sec": elapsed_sec,
    }
    return {
        "task": "smoke",
        "status": "passed" if payload["summary"]["passed"] else "failed",
        "elapsed_sec": elapsed_sec,
        "rows": [row],
        "payload": payload,
    }


def split_niah_span_candidate_diagnostics(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract final span-candidate diagnostics without making them selectors.

    中文说明:
    - 调用方 / Called by: `run_niah_section`, `run_niah_row`
    - 调用对象 / Calls: 无。
    - 作用 / Purpose: 将 span predictor 的 eval 候选池结构指标统一放到
      diagnostic_metrics；这些字段解释为什么答错，但不能进入 validation/test
      主指标，也不能用来选择实验组。
    - 返回 / Returns: dict，缺字段时值为 None。
    - 错误处理 / Error handling: 输入为空或字段缺失时返回 None 值。
    - 副作用 / Side effects: 无。

    English documentation:
    Function name:
        split_niah_span_candidate_diagnostics
    Purpose:
        Keep final span-candidate diagnostics in the diagnostic report block only.
    """
    source = metrics.get("final_span_candidate_diagnostics", {})
    if not isinstance(source, Mapping):
        source = {}
    return {
        field: source.get(field, metrics.get(field))
        for field in NIAH_SPAN_CANDIDATE_DIAGNOSTIC_FIELDS
    }


def split_niah_test_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Extract held-out NIAH test metrics without touching validation selection.

    中文说明:
    - 调用方 / Called by: `run_niah_section`, `run_niah_row`
    - 调用对象 / Calls: 无。
    - 作用 / Purpose: 只把最终 held-out test 的主指标放入 `test_metrics`；
      这些字段在训练与 best-state 选择之后产生，不得参与组选择。
    - 返回 / Returns: 未启用 test eval 时返回空 dict。
    - 副作用 / Side effects: 无。

    English documentation:
    Function name:
        split_niah_test_metrics
    Purpose:
        Keep held-out NIAH test headline metrics separate from validation metrics.
    """
    if metrics.get("test_accuracy") is None:
        return {}
    return {
        "test_eval_mean_accuracy": metrics.get("test_accuracy"),
        "test_eval_min_depth_accuracy": metrics.get("test_min_depth_accuracy"),
        "test_eval_loss": metrics.get("test_eval_loss"),
        "test_readout_available_rate": metrics.get("test_readout_available_rate"),
        "test_target_candidate_hit_rate": metrics.get(
            "test_target_candidate_hit_rate"
        ),
    }


def prefix_niah_test_diagnostics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Extract held-out NIAH test diagnostics for explanation only.

    中文说明:
    - 调用方 / Called by: `run_niah_section`, `run_niah_row`
    - 作用 / Purpose: 将 test 候选 rank/pair 等解释性字段放入 diagnostics，
      不进入 validation/test 主指标，避免被误用作选择依据。
    - 返回 / Returns: 带 `test_` 前缀的诊断字段。
    - 副作用 / Side effects: 无。

    English documentation:
    Function name:
        prefix_niah_test_diagnostics
    Purpose:
        Prefix held-out NIAH test diagnostics so they remain explanatory only.
    """
    source = metrics.get("test_span_candidate_diagnostics", {})
    if not isinstance(source, Mapping):
        source = {}
    diagnostics = {
        "test_mean_target_candidate_rank": metrics.get(
            "test_mean_target_candidate_rank"
        )
    }
    diagnostics.update(
        {
            f"test_{field}": source.get(field)
            for field in NIAH_SPAN_CANDIDATE_DIAGNOSTIC_FIELDS
        }
    )
    return diagnostics


def run_niah_section(
    args: argparse.Namespace,
    device: torch.device,
    groups: Sequence[str],
) -> dict[str, Any]:
    """Run NIAH rows for all ablation groups."""
    rows: list[dict[str, Any]] = []
    for group_name in groups:
        override = group_override(group_name)
        for seq_len in args.niah_seq_lengths:
            for seed in args.niah_seeds:
                seed_everything(seed)
                started_at = time.perf_counter()
                row: dict[str, Any] = {
                    "group": group_name,
                    "task": "niah",
                    "seed": int(seed),
                    "status": "running",
                    "config": {
                        "seq_len": int(seq_len),
                        "dim": int(args.niah_dim),
                        "num_layers": int(args.niah_layers),
                        "slots": int(args.niah_slots),
                        "read_topk": int(args.niah_read_topk),
                        "epochs": int(args.niah_epochs),
                        "eval_batches_per_depth": int(args.niah_eval_batches_per_depth),
                        "retrieval_evidence_loss_alpha": float(
                            group_capability(
                                group_name,
                                "retrieval_evidence_loss_alpha",
                                0.0,
                            )
                        ),
                        "retrieval_evidence_rank_margin": float(
                            group_capability(
                                group_name,
                                "retrieval_evidence_rank_margin",
                                0.0,
                            )
                        ),
                        "retrieval_evidence_score_margin": float(
                            group_capability(
                                group_name,
                                "retrieval_evidence_score_margin",
                                0.0,
                            )
                        ),
                        "retrieval_evidence_target_offset": int(
                            group_capability(
                                group_name,
                                "retrieval_evidence_target_offset",
                                DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
                            )
                        ),
                        "query_evidence_alignment_alpha": float(
                            group_capability(
                                group_name,
                                "query_evidence_alignment_alpha",
                                0.0,
                            )
                        ),
                        "retrieval_projection_contrastive_alpha": float(
                            group_capability(
                                group_name,
                                "retrieval_projection_contrastive_alpha",
                                0.0,
                            )
                        ),
                        "retrieval_projection_temperature": float(
                            group_capability(
                                group_name,
                                "retrieval_projection_temperature",
                                DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
                            )
                        ),
                        "retrieval_span_predictor_alpha": float(
                            group_capability(
                                group_name,
                                "retrieval_span_predictor_alpha",
                                0.0,
                            )
                        ),
                        "niah_span_candidate_filter": str(
                            group_capability(
                                group_name,
                                "niah_span_candidate_filter",
                                DEFAULT_NIAH_SPAN_CANDIDATE_FILTER,
                            )
                        ),
                        "niah_span_loss_mode": str(
                            group_capability(
                                group_name,
                                "niah_span_loss_mode",
                                DEFAULT_NIAH_SPAN_LOSS_MODE,
                            )
                        ),
                        "retrieval_neighbor_span": int(
                            group_capability(group_name, "neighbor_span", 0)
                        ),
                        "retrieval_neighbor_direction": str(
                            group_capability(
                                group_name,
                                "neighbor_direction",
                                DEFAULT_NIAH_RETRIEVAL_NEIGHBOR_DIRECTION,
                            )
                        ),
                        "retrieval_neighbor_seed_multiplier": int(
                            group_capability(
                                group_name,
                                "neighbor_seed_multiplier",
                                1,
                            )
                        ),
                        "retrieval_neighbor_budget_mode": str(
                            group_capability(
                                group_name,
                                "neighbor_budget_mode",
                                "unbounded",
                            )
                        ),
                        "retrieval_max_tokens": int(
                            group_capability(
                                group_name,
                                "retrieval_max_tokens",
                                128,
                            )
                        ),
                        "niah_readout_mode": str(
                            group_capability(group_name, "niah_readout_mode", "model")
                        ),
                        "use_retrieval": group_uses_niah_retrieval(group_name),
                        "mhdsra2_config_override": override,
                    },
                }
                try:
                    metrics = run_niah_verification_case(
                        seq_len=seq_len,
                        device=device,
                        vocab_size=args.niah_vocab_size,
                        data_vocab_size=args.niah_vocab_size,
                        dim=args.niah_dim,
                        num_layers=args.niah_layers,
                        K=args.niah_slots,
                        kr=args.niah_read_topk,
                        chunk_size=args.niah_chunk_size,
                        batch_size=args.niah_batch_size,
                        epochs=args.niah_epochs,
                        learning_rate=args.niah_lr,
                        seed=seed,
                        model_type="mhdsra2",
                        mhdsra2_config_override=override,
                        eval_batches_per_depth=args.niah_eval_batches_per_depth,
                        robust_eval_interval=args.niah_robust_eval_interval,
                        robust_eval_batches_per_depth=args.niah_eval_batches_per_depth,
                        niah_test_batches_per_depth=int(
                            row["config"].get(
                                "test_batches_per_depth",
                                DEFAULT_NIAH_TEST_BATCHES_PER_DEPTH,
                            )
                        ),
                        swanlab_mode="disabled",
                        needle_loss_alpha=args.niah_needle_loss_alpha,
                        hidden_mse_alpha=args.niah_hidden_mse_alpha,
                        retrieval_evidence_loss_alpha=float(
                            row["config"]["retrieval_evidence_loss_alpha"]
                        ),
                        retrieval_evidence_rank_margin=float(
                            row["config"].get("retrieval_evidence_rank_margin", 0.0)
                        ),
                        retrieval_evidence_score_margin=float(
                            row["config"].get("retrieval_evidence_score_margin", 0.0)
                        ),
                        retrieval_evidence_target_offset=int(
                            row["config"].get(
                                "retrieval_evidence_target_offset",
                                DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
                            )
                        ),
                        query_evidence_alignment_alpha=float(
                            row["config"].get("query_evidence_alignment_alpha", 0.0)
                        ),
                        retrieval_projection_contrastive_alpha=float(
                            row["config"].get(
                                "retrieval_projection_contrastive_alpha",
                                0.0,
                            )
                        ),
                        retrieval_projection_temperature=float(
                            row["config"].get(
                                "retrieval_projection_temperature",
                                DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
                            )
                        ),
                        retrieval_span_predictor_alpha=float(
                            row["config"].get("retrieval_span_predictor_alpha", 0.0)
                        ),
                        retrieval_span_structure_features=bool(
                            row["config"].get("retrieval_span_structure_features", False)
                        ),
                        niah_span_candidate_filter=str(
                            row["config"].get(
                                "niah_span_candidate_filter",
                                DEFAULT_NIAH_SPAN_CANDIDATE_FILTER,
                            )
                        ),
                        niah_span_loss_mode=str(
                            row["config"].get(
                                "niah_span_loss_mode",
                                DEFAULT_NIAH_SPAN_LOSS_MODE,
                            )
                        ),
                        use_retrieval=bool(row["config"]["use_retrieval"]),
                        niah_readout_mode=str(
                            row["config"].get("niah_readout_mode", "model")
                        ),
                    )
                    span_candidate_diagnostics = split_niah_span_candidate_diagnostics(
                        metrics
                    )
                    test_metrics = split_niah_test_metrics(metrics)
                    test_diagnostics = prefix_niah_test_diagnostics(metrics)
                    final_span_candidate_diagnostics = {
                        f"final_{key}": value
                        for key, value in span_candidate_diagnostics.items()
                    }
                    row.update(
                        {
                            "status": "completed",
                            "validation_metrics": {
                                "final_eval_mean_accuracy": metrics.get(
                                    "final_accuracy"
                                ),
                                "final_eval_min_depth_accuracy": metrics.get(
                                    "final_min_depth_accuracy"
                                ),
                                "best_eval_mean_accuracy": metrics.get(
                                    "best_accuracy"
                                ),
                                "final_eval_loss": metrics.get("final_eval_loss"),
                                "final_readout_available_rate": metrics.get(
                                    "final_readout_available_rate"
                                ),
                                "final_target_candidate_hit_rate": metrics.get(
                                    "final_target_candidate_hit_rate"
                                ),
                            },
                            "test_metrics": test_metrics,
                            "diagnostic_metrics": {
                                "final_train_loss": metrics.get("final_train_loss"),
                                "peak_memory_allocated_mb": metrics.get(
                                    "peak_memory_allocated_mb"
                                ),
                                "peak_memory_reserved_mb": metrics.get(
                                    "peak_memory_reserved_mb"
                                ),
                                "final_readout_mode": metrics.get(
                                    "final_readout_mode"
                                ),
                                "final_readout_available_rate": metrics.get(
                                    "final_readout_available_rate"
                                ),
                                "final_target_candidate_hit_rate": metrics.get(
                                    "final_target_candidate_hit_rate"
                                ),
                                "final_mean_target_candidate_rank": metrics.get(
                                    "final_mean_target_candidate_rank"
                                ),
                                "final_span_candidate_diagnostics": (
                                    span_candidate_diagnostics
                                ),
                                **final_span_candidate_diagnostics,
                                **test_diagnostics,
                                "retrieval_evidence": metrics.get(
                                    "final_retrieval_evidence_metrics"
                                ),
                                "train_retrieval_evidence_summary": metrics.get(
                                    "train_retrieval_evidence_summary"
                                ),
                                "train_query_evidence_alignment_summary": metrics.get(
                                    "train_query_evidence_alignment_summary"
                                ),
                                "train_retrieval_projection_summary": metrics.get(
                                    "train_retrieval_projection_summary"
                                ),
                                "retrieval_projection_mean_loss": (
                                    metrics.get(
                                        "train_retrieval_projection_summary",
                                        {},
                                    )
                                    or {}
                                ).get("mean_loss"),
                                "retrieval_projection_mean_target_rank": (
                                    metrics.get(
                                        "train_retrieval_projection_summary",
                                        {},
                                    )
                                    or {}
                                ).get("mean_target_rank"),
                                "retrieval_projection_mean_top1_rate": (
                                    metrics.get(
                                        "train_retrieval_projection_summary",
                                        {},
                                    )
                                    or {}
                                ).get("mean_top1_rate"),
                                "final_retrieval_projection": metrics.get(
                                    "final_retrieval_projection_metrics"
                                ),
                                "train_retrieval_span_predictor_summary": metrics.get(
                                    "train_retrieval_span_predictor_summary"
                                ),
                                "retrieval_span_predictor_mean_loss": (
                                    metrics.get(
                                        "train_retrieval_span_predictor_summary",
                                        {},
                                    )
                                    or {}
                                ).get("mean_loss"),
                                "retrieval_span_predictor_mean_target_rank": (
                                    metrics.get(
                                        "train_retrieval_span_predictor_summary",
                                        {},
                                    )
                                    or {}
                                ).get("mean_target_rank"),
                                "retrieval_span_predictor_mean_top1_rate": (
                                    metrics.get(
                                        "train_retrieval_span_predictor_summary",
                                        {},
                                    )
                                    or {}
                                ).get("mean_top1_rate"),
                                "retrieval_span_predictor_mean_logit_margin": (
                                    metrics.get(
                                        "train_retrieval_span_predictor_summary",
                                        {},
                                    )
                                    or {}
                                ).get("mean_logit_margin"),
                                "final_retrieval_span_predictor": metrics.get(
                                    "final_retrieval_span_predictor_metrics"
                                ),
                                "query_evidence_alignment_mean_loss": (
                                    metrics.get(
                                        "train_query_evidence_alignment_summary",
                                        {},
                                    )
                                    or {}
                                ).get("mean_loss"),
                                "query_evidence_alignment_mean_cosine": (
                                    metrics.get(
                                        "train_query_evidence_alignment_summary",
                                        {},
                                    )
                                    or {}
                                ).get("mean_cosine"),
                                "query_evidence_alignment_mean_mse": (
                                    metrics.get(
                                        "train_query_evidence_alignment_summary",
                                        {},
                                    )
                                    or {}
                                ).get("mean_mse"),
                                "retrieval_evidence_available": (
                                    metrics.get("final_retrieval_evidence_metrics", {}) or {}
                                ).get("available"),
                                "retrieval_evidence_unavailable_reason": (
                                    metrics.get("final_retrieval_evidence_metrics", {}) or {}
                                ).get("unavailable_reason"),
                                "retrieval_evidence_hit_rate": (
                                    metrics.get("final_retrieval_evidence_metrics", {}) or {}
                                ).get("hit_rate"),
                                "retrieval_evidence_weight_mean": (
                                    metrics.get("final_retrieval_evidence_metrics", {}) or {}
                                ).get("evidence_weight_mean"),
                                "retrieval_evidence_best_negative_weight_mean": (
                                    metrics.get("final_retrieval_evidence_metrics", {}) or {}
                                ).get("best_negative_weight_mean"),
                                "retrieval_evidence_margin_mean": (
                                    metrics.get("final_retrieval_evidence_metrics", {}) or {}
                                ).get("evidence_margin_mean"),
                                "retrieval_evidence_target_rank_mean": (
                                    metrics.get("final_retrieval_evidence_metrics", {}) or {}
                                ).get("target_rank_mean"),
                                "retrieval_evidence_top1_rate": (
                                    metrics.get("final_retrieval_evidence_metrics", {}) or {}
                                ).get("top1_rate"),
                                "retrieval_evidence_score_margin_loss": (
                                    metrics.get("final_retrieval_evidence_metrics", {}) or {}
                                ).get("score_margin_loss"),
                                "retrieval_evidence_score_mean": (
                                    metrics.get("final_retrieval_evidence_metrics", {}) or {}
                                ).get("evidence_score_mean"),
                                "retrieval_evidence_best_negative_score_mean": (
                                    metrics.get("final_retrieval_evidence_metrics", {}) or {}
                                ).get("best_negative_score_mean"),
                                "retrieval_evidence_score_margin_mean": (
                                    metrics.get("final_retrieval_evidence_metrics", {}) or {}
                                ).get("evidence_score_margin_mean"),
                                "retrieval_evidence_score_target_rank_mean": (
                                    metrics.get("final_retrieval_evidence_metrics", {}) or {}
                                ).get("score_target_rank_mean"),
                                "retrieval_evidence_score_top1_rate": (
                                    metrics.get("final_retrieval_evidence_metrics", {}) or {}
                                ).get("score_top1_rate"),
                                "slot_collision": summarize_slot_collision_diagnostics(
                                    (
                                        metrics.get("final_aux_diagnostics", {})
                                        .get("last_layer")
                                        if isinstance(
                                            metrics.get("final_aux_diagnostics"),
                                            Mapping,
                                        )
                                        else None
                                    )
                                ),
                            },
                            "metrics": {
                                "status": metrics.get("status"),
                                "final_accuracy": metrics.get("final_accuracy"),
                                "final_min_depth_accuracy": metrics.get(
                                    "final_min_depth_accuracy"
                                ),
                                "final_eval_loss": metrics.get("final_eval_loss"),
                                "final_train_loss": metrics.get("final_train_loss"),
                                "best_accuracy": metrics.get("best_accuracy"),
                                "best_min_depth_accuracy": metrics.get(
                                    "best_min_depth_accuracy"
                                ),
                                "best_accuracy_loss": metrics.get("best_accuracy_loss"),
                                "final_readout_mode": metrics.get(
                                    "final_readout_mode"
                                ),
                                "final_readout_available_rate": metrics.get(
                                    "final_readout_available_rate"
                                ),
                                "final_target_candidate_hit_rate": metrics.get(
                                    "final_target_candidate_hit_rate"
                                ),
                                "final_mean_target_candidate_rank": metrics.get(
                                    "final_mean_target_candidate_rank"
                                ),
                                "final_retrieval_evidence_metrics": metrics.get(
                                    "final_retrieval_evidence_metrics"
                                ),
                                "train_retrieval_evidence_summary": metrics.get(
                                    "train_retrieval_evidence_summary"
                                ),
                                "train_query_evidence_alignment_summary": metrics.get(
                                    "train_query_evidence_alignment_summary"
                                ),
                                "train_retrieval_projection_summary": metrics.get(
                                    "train_retrieval_projection_summary"
                                ),
                                "final_retrieval_projection_metrics": metrics.get(
                                    "final_retrieval_projection_metrics"
                                ),
                                "final_retrieval_span_predictor_metrics": metrics.get(
                                    "final_retrieval_span_predictor_metrics"
                                ),
                                "final_span_candidate_diagnostics": metrics.get(
                                    "final_span_candidate_diagnostics"
                                ),
                                "train_retrieval_span_predictor_summary": metrics.get(
                                    "train_retrieval_span_predictor_summary"
                                ),
                                "peak_memory_allocated_mb": metrics.get(
                                    "peak_memory_allocated_mb"
                                ),
                                "peak_memory_reserved_mb": metrics.get(
                                    "peak_memory_reserved_mb"
                                ),
                            },
                        }
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    cleanup_after_oom()
                    row.update({"status": "oom", "error": str(exc)})
                except RuntimeError as exc:
                    if not is_oom_error(exc):
                        raise
                    cleanup_after_oom()
                    row.update({"status": "oom", "error": str(exc)})
                finally:
                    row["elapsed_sec"] = time.perf_counter() - started_at
                    rows.append(row)
    return {"task": "niah", "rows": rows}


def run_niah_row(
    args: argparse.Namespace,
    device: torch.device,
    planned_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one NIAH ablation row for resumable execution."""
    group_name = str(planned_row["group"])
    seed = int(planned_row["seed"])
    config = dict(planned_row["config"])
    override = group_override(group_name)
    seed_everything(seed)
    started_at = time.perf_counter()
    row: dict[str, Any] = {
        "group": group_name,
        "task": "niah",
        "seed": seed,
        "status": "running",
        "config": config,
    }
    try:
        metrics = run_niah_verification_case(
            seq_len=int(config["seq_len"]),
            device=device,
            vocab_size=args.niah_vocab_size,
            data_vocab_size=args.niah_vocab_size,
            dim=int(config["dim"]),
            num_layers=int(config["num_layers"]),
            K=int(config["slots"]),
            kr=int(config["read_topk"]),
            chunk_size=args.niah_chunk_size,
            batch_size=args.niah_batch_size,
            epochs=int(config["epochs"]),
            learning_rate=args.niah_lr,
            seed=seed,
            model_type="mhdsra2",
            mhdsra2_config_override=override,
            eval_batches_per_depth=int(config["eval_batches_per_depth"]),
            robust_eval_interval=args.niah_robust_eval_interval,
            robust_eval_batches_per_depth=int(config["eval_batches_per_depth"]),
            niah_test_batches_per_depth=int(
                config.get(
                    "test_batches_per_depth",
                    DEFAULT_NIAH_TEST_BATCHES_PER_DEPTH,
                )
            ),
            swanlab_mode="disabled",
            needle_loss_alpha=args.niah_needle_loss_alpha,
            hidden_mse_alpha=args.niah_hidden_mse_alpha,
            retrieval_evidence_loss_alpha=float(
                config.get("retrieval_evidence_loss_alpha", 0.0)
            ),
            retrieval_evidence_rank_margin=float(
                config.get("retrieval_evidence_rank_margin", 0.0)
            ),
            retrieval_evidence_score_margin=float(
                config.get("retrieval_evidence_score_margin", 0.0)
            ),
            retrieval_evidence_target_offset=int(
                config.get(
                    "retrieval_evidence_target_offset",
                    DEFAULT_NIAH_RETRIEVAL_EVIDENCE_TARGET_OFFSET,
                )
            ),
            query_evidence_alignment_alpha=float(
                config.get("query_evidence_alignment_alpha", 0.0)
            ),
            retrieval_projection_contrastive_alpha=float(
                config.get("retrieval_projection_contrastive_alpha", 0.0)
            ),
            retrieval_projection_temperature=float(
                config.get(
                    "retrieval_projection_temperature",
                    DEFAULT_NIAH_RETRIEVAL_PROJECTION_TEMPERATURE,
                )
            ),
            retrieval_span_predictor_alpha=float(
                config.get("retrieval_span_predictor_alpha", 0.0)
            ),
            retrieval_span_structure_features=bool(
                config.get("retrieval_span_structure_features", False)
            ),
            niah_span_candidate_filter=str(
                config.get(
                    "niah_span_candidate_filter",
                    DEFAULT_NIAH_SPAN_CANDIDATE_FILTER,
                )
            ),
            niah_span_loss_mode=str(
                config.get(
                    "niah_span_loss_mode",
                    DEFAULT_NIAH_SPAN_LOSS_MODE,
                )
            ),
            use_retrieval=bool(config.get("use_retrieval", group_uses_niah_retrieval(group_name))),
            niah_readout_mode=str(config.get("niah_readout_mode", "model")),
        )
        span_candidate_diagnostics = split_niah_span_candidate_diagnostics(metrics)
        test_metrics = split_niah_test_metrics(metrics)
        test_diagnostics = prefix_niah_test_diagnostics(metrics)
        final_span_candidate_diagnostics = {
            f"final_{key}": value
            for key, value in span_candidate_diagnostics.items()
        }
        row.update(
            {
                "status": "completed",
                "validation_metrics": {
                    "final_eval_mean_accuracy": metrics.get("final_accuracy"),
                    "final_eval_min_depth_accuracy": metrics.get(
                        "final_min_depth_accuracy"
                    ),
                    "best_eval_mean_accuracy": metrics.get("best_accuracy"),
                    "final_eval_loss": metrics.get("final_eval_loss"),
                    "final_readout_available_rate": metrics.get(
                        "final_readout_available_rate"
                    ),
                    "final_target_candidate_hit_rate": metrics.get(
                        "final_target_candidate_hit_rate"
                    ),
                },
                "test_metrics": test_metrics,
                "diagnostic_metrics": {
                    "final_train_loss": metrics.get("final_train_loss"),
                    "peak_memory_allocated_mb": metrics.get(
                        "peak_memory_allocated_mb"
                    ),
                    "peak_memory_reserved_mb": metrics.get(
                        "peak_memory_reserved_mb"
                    ),
                    "final_readout_mode": metrics.get("final_readout_mode"),
                    "final_readout_available_rate": metrics.get(
                        "final_readout_available_rate"
                    ),
                    "final_target_candidate_hit_rate": metrics.get(
                        "final_target_candidate_hit_rate"
                    ),
                    "final_mean_target_candidate_rank": metrics.get(
                        "final_mean_target_candidate_rank"
                    ),
                    "final_span_candidate_diagnostics": span_candidate_diagnostics,
                    **final_span_candidate_diagnostics,
                    **test_diagnostics,
                    "retrieval_evidence": metrics.get(
                        "final_retrieval_evidence_metrics"
                    ),
                    "train_retrieval_evidence_summary": metrics.get(
                        "train_retrieval_evidence_summary"
                    ),
                    "train_query_evidence_alignment_summary": metrics.get(
                        "train_query_evidence_alignment_summary"
                    ),
                    "train_retrieval_projection_summary": metrics.get(
                        "train_retrieval_projection_summary"
                    ),
                    "retrieval_projection_mean_loss": (
                        metrics.get("train_retrieval_projection_summary", {}) or {}
                    ).get("mean_loss"),
                    "retrieval_projection_mean_target_rank": (
                        metrics.get("train_retrieval_projection_summary", {}) or {}
                    ).get("mean_target_rank"),
                    "retrieval_projection_mean_top1_rate": (
                        metrics.get("train_retrieval_projection_summary", {}) or {}
                    ).get("mean_top1_rate"),
                    "final_retrieval_projection": metrics.get(
                        "final_retrieval_projection_metrics"
                    ),
                    "train_retrieval_span_predictor_summary": metrics.get(
                        "train_retrieval_span_predictor_summary"
                    ),
                    "retrieval_span_predictor_mean_loss": (
                        metrics.get("train_retrieval_span_predictor_summary", {}) or {}
                    ).get("mean_loss"),
                    "retrieval_span_predictor_mean_target_rank": (
                        metrics.get("train_retrieval_span_predictor_summary", {}) or {}
                    ).get("mean_target_rank"),
                    "retrieval_span_predictor_mean_top1_rate": (
                        metrics.get("train_retrieval_span_predictor_summary", {}) or {}
                    ).get("mean_top1_rate"),
                    "retrieval_span_predictor_mean_logit_margin": (
                        metrics.get("train_retrieval_span_predictor_summary", {}) or {}
                    ).get("mean_logit_margin"),
                    "final_retrieval_span_predictor": metrics.get(
                        "final_retrieval_span_predictor_metrics"
                    ),
                    "query_evidence_alignment_mean_loss": (
                        metrics.get("train_query_evidence_alignment_summary", {}) or {}
                    ).get("mean_loss"),
                    "query_evidence_alignment_mean_cosine": (
                        metrics.get("train_query_evidence_alignment_summary", {}) or {}
                    ).get("mean_cosine"),
                    "query_evidence_alignment_mean_mse": (
                        metrics.get("train_query_evidence_alignment_summary", {}) or {}
                    ).get("mean_mse"),
                    "retrieval_evidence_available": (
                        metrics.get("final_retrieval_evidence_metrics", {}) or {}
                    ).get("available"),
                    "retrieval_evidence_unavailable_reason": (
                        metrics.get("final_retrieval_evidence_metrics", {}) or {}
                    ).get("unavailable_reason"),
                    "retrieval_evidence_hit_rate": (
                        metrics.get("final_retrieval_evidence_metrics", {}) or {}
                    ).get("hit_rate"),
                    "retrieval_evidence_weight_mean": (
                        metrics.get("final_retrieval_evidence_metrics", {}) or {}
                    ).get("evidence_weight_mean"),
                    "retrieval_evidence_best_negative_weight_mean": (
                        metrics.get("final_retrieval_evidence_metrics", {}) or {}
                    ).get("best_negative_weight_mean"),
                    "retrieval_evidence_margin_mean": (
                        metrics.get("final_retrieval_evidence_metrics", {}) or {}
                    ).get("evidence_margin_mean"),
                    "retrieval_evidence_target_rank_mean": (
                        metrics.get("final_retrieval_evidence_metrics", {}) or {}
                    ).get("target_rank_mean"),
                    "retrieval_evidence_top1_rate": (
                        metrics.get("final_retrieval_evidence_metrics", {}) or {}
                    ).get("top1_rate"),
                    "retrieval_evidence_score_margin_loss": (
                        metrics.get("final_retrieval_evidence_metrics", {}) or {}
                    ).get("score_margin_loss"),
                    "retrieval_evidence_score_mean": (
                        metrics.get("final_retrieval_evidence_metrics", {}) or {}
                    ).get("evidence_score_mean"),
                    "retrieval_evidence_best_negative_score_mean": (
                        metrics.get("final_retrieval_evidence_metrics", {}) or {}
                    ).get("best_negative_score_mean"),
                    "retrieval_evidence_score_margin_mean": (
                        metrics.get("final_retrieval_evidence_metrics", {}) or {}
                    ).get("evidence_score_margin_mean"),
                    "retrieval_evidence_score_target_rank_mean": (
                        metrics.get("final_retrieval_evidence_metrics", {}) or {}
                    ).get("score_target_rank_mean"),
                    "retrieval_evidence_score_top1_rate": (
                        metrics.get("final_retrieval_evidence_metrics", {}) or {}
                    ).get("score_top1_rate"),
                    "slot_collision": summarize_slot_collision_diagnostics(
                        (
                            metrics.get("final_aux_diagnostics", {}).get("last_layer")
                            if isinstance(metrics.get("final_aux_diagnostics"), Mapping)
                            else None
                        )
                    ),
                },
                "metrics": {
                    "status": metrics.get("status"),
                    "final_accuracy": metrics.get("final_accuracy"),
                    "final_min_depth_accuracy": metrics.get("final_min_depth_accuracy"),
                    "final_eval_loss": metrics.get("final_eval_loss"),
                    "final_train_loss": metrics.get("final_train_loss"),
                    "best_accuracy": metrics.get("best_accuracy"),
                    "best_min_depth_accuracy": metrics.get("best_min_depth_accuracy"),
                    "best_accuracy_loss": metrics.get("best_accuracy_loss"),
                    "final_readout_mode": metrics.get("final_readout_mode"),
                    "final_readout_available_rate": metrics.get(
                        "final_readout_available_rate"
                    ),
                    "final_target_candidate_hit_rate": metrics.get(
                        "final_target_candidate_hit_rate"
                    ),
                    "final_mean_target_candidate_rank": metrics.get(
                        "final_mean_target_candidate_rank"
                    ),
                    "final_retrieval_evidence_metrics": metrics.get(
                        "final_retrieval_evidence_metrics"
                    ),
                    "train_retrieval_evidence_summary": metrics.get(
                        "train_retrieval_evidence_summary"
                    ),
                    "train_query_evidence_alignment_summary": metrics.get(
                        "train_query_evidence_alignment_summary"
                    ),
                    "train_retrieval_projection_summary": metrics.get(
                        "train_retrieval_projection_summary"
                    ),
                    "final_retrieval_projection_metrics": metrics.get(
                        "final_retrieval_projection_metrics"
                    ),
                    "final_retrieval_span_predictor_metrics": metrics.get(
                        "final_retrieval_span_predictor_metrics"
                    ),
                    "final_span_candidate_diagnostics": metrics.get(
                        "final_span_candidate_diagnostics"
                    ),
                    "test_span_candidate_diagnostics": metrics.get(
                        "test_span_candidate_diagnostics"
                    ),
                    "test_accuracy": metrics.get("test_accuracy"),
                    "test_min_depth_accuracy": metrics.get(
                        "test_min_depth_accuracy"
                    ),
                    "test_eval_loss": metrics.get("test_eval_loss"),
                    "train_retrieval_span_predictor_summary": metrics.get(
                        "train_retrieval_span_predictor_summary"
                    ),
                    "peak_memory_allocated_mb": metrics.get(
                        "peak_memory_allocated_mb"
                    ),
                    "peak_memory_reserved_mb": metrics.get(
                        "peak_memory_reserved_mb"
                    ),
                },
            }
        )
    except torch.cuda.OutOfMemoryError as exc:
        cleanup_after_oom()
        row.update(
            {
                "status": "oom",
                "error": str(exc),
                "validation_metrics": {},
                "test_metrics": {},
                "diagnostic_metrics": {},
            }
        )
    except RuntimeError as exc:
        if not is_oom_error(exc):
            raise
        cleanup_after_oom()
        row.update(
            {
                "status": "oom",
                "error": str(exc),
                "validation_metrics": {},
                "test_metrics": {},
                "diagnostic_metrics": {},
            }
        )
    finally:
        row["elapsed_sec"] = time.perf_counter() - started_at
    return row


def summarize_json_seed_result(result: Mapping[str, Any]) -> dict[str, float | None]:
    """Extract JSON validation/test metrics for one seed run.

    中文说明:
    - 调用方 / Called by: `run_json_section`, `run_json_row`
    - 调用对象 / Calls: 无，只读取 `run_json_retrieval_generalization_test()` 的结果字典。
    - 作用 / Purpose: 汇总 generation 主指标，同时保留 teacher-forced、slot decoder、
      evidence decoder 的诊断指标；selection 仍只看 validation generation 指标。
    - 返回 / Returns: 扁平 metrics 字典，值可能为 float 或 None。
    - 错误处理 / Error handling: 缺少必需 pool evaluation 时沿用 KeyError 暴露调用错误。
    - 副作用 / Side effects: 无，不访问 test 选择逻辑。

    English documentation:
    Function name:
        summarize_json_seed_result
    Purpose:
        Flatten generation metrics and readout diagnostics from one JSON
        generalization run without changing validation-only selection policy.
    """
    validation = result["validation_pool_evaluation"]
    test = result["test_pool_evaluation"]
    return {
        "validation_generation_exact_match_rate": validation[
            "generation_exact_match_rate"
        ],
        "validation_generation_mean_sequence_accuracy": validation[
            "generation_mean_sequence_accuracy"
        ],
        "validation_teacher_forced_exact_match_rate": validation[
            "teacher_forced_exact_match_rate"
        ],
        "validation_teacher_forced_mean_sequence_accuracy": validation[
            "teacher_forced_mean_sequence_accuracy"
        ],
        "test_generation_exact_match_rate": test["generation_exact_match_rate"],
        "test_generation_mean_sequence_accuracy": test[
            "generation_mean_sequence_accuracy"
        ],
        "test_teacher_forced_exact_match_rate": test[
            "teacher_forced_exact_match_rate"
        ],
        "test_teacher_forced_mean_sequence_accuracy": test[
            "teacher_forced_mean_sequence_accuracy"
        ],
        "validation_slot_decoder_full_answer_accuracy": validation.get(
            "slot_decoder_full_answer_accuracy"
        ),
        "validation_slot_decoder_museum_accuracy": validation.get(
            "slot_decoder_museum_accuracy"
        ),
        "validation_slot_decoder_artifact_accuracy": validation.get(
            "slot_decoder_artifact_accuracy"
        ),
        "validation_slot_decoder_artist_accuracy": validation.get(
            "slot_decoder_artist_accuracy"
        ),
        "validation_slot_decoder_dynasty_accuracy": validation.get(
            "slot_decoder_dynasty_accuracy"
        ),
        "test_slot_decoder_full_answer_accuracy": test.get(
            "slot_decoder_full_answer_accuracy"
        ),
        "test_slot_decoder_museum_accuracy": test.get("slot_decoder_museum_accuracy"),
        "test_slot_decoder_artifact_accuracy": test.get(
            "slot_decoder_artifact_accuracy"
        ),
        "test_slot_decoder_artist_accuracy": test.get("slot_decoder_artist_accuracy"),
        "test_slot_decoder_dynasty_accuracy": test.get("slot_decoder_dynasty_accuracy"),
        "validation_evidence_window_accuracy": validation.get("evidence_window_accuracy"),
        "validation_evidence_window_mean_distance": validation.get(
            "evidence_window_mean_distance"
        ),
        "test_evidence_window_accuracy": test.get("evidence_window_accuracy"),
        "test_evidence_window_mean_distance": test.get("evidence_window_mean_distance"),
        "validation_extract_then_compose_exact_match_rate": validation.get(
            "extract_then_compose_exact_match_rate"
        ),
        "validation_extract_then_compose_mean_sequence_accuracy": validation.get(
            "extract_then_compose_mean_sequence_accuracy"
        ),
        "test_extract_then_compose_exact_match_rate": test.get(
            "extract_then_compose_exact_match_rate"
        ),
        "test_extract_then_compose_mean_sequence_accuracy": test.get(
            "extract_then_compose_mean_sequence_accuracy"
        ),
    }


def split_json_row_metrics(
    metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split flattened JSON metrics into report validation/test/diagnostic blocks.

    中文说明:
    - 调用方 / Called by: `run_json_section`, `run_json_row`
    - 调用对象 / Calls: 无。
    - 作用 / Purpose: generation 指标放入 validation/test 主栏；teacher-forced、
      slot decoder 和 evidence decoder 留作诊断，避免 test 诊断被误用作组选择。
    - 返回 / Returns: `(validation_metrics, test_metrics, diagnostic_metrics)`。
    - 错误处理 / Error handling: 输入为空时返回三个空字典。
    - 副作用 / Side effects: 无。

    English documentation:
    Function name:
        split_json_row_metrics
    Purpose:
        Keep report field selection consistent between non-resumable and
        resumable JSON ablation paths.
    """
    validation_metrics = {
        key: value
        for key, value in metrics.items()
        if key.startswith("validation_generation_")
    }
    test_metrics = {
        key: value
        for key, value in metrics.items()
        if key.startswith("test_generation_")
    }
    diagnostic_metric_markers = (
        "_teacher_forced_",
        "_slot_decoder_",
        "_evidence_",
        "_extract_then_compose_",
    )
    diagnostic_metrics = {
        key: value
        for key, value in metrics.items()
        if any(marker in key for marker in diagnostic_metric_markers)
    }
    return validation_metrics, test_metrics, diagnostic_metrics


def run_json_section(
    args: argparse.Namespace,
    groups: Sequence[str],
    device: torch.device,
) -> dict[str, Any]:
    """Run JSON retrieval generalization rows for all ablation groups."""
    group_results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for group_name in groups:
        seed_runs: list[dict[str, Any]] = []
        override = group_override(group_name)
        for seed_root in args.json_task_seed_roots:
            seed_bundle = build_task_seed_bundle(seed_root)
            seed_everything(seed_bundle["model_seed"])
            started_at = time.perf_counter()
            json_capabilities = json_group_capabilities(group_name)
            row = {
                "group": group_name,
                "task": "json",
                "seed": int(seed_root),
                "status": "running",
                "config": {
                    "epochs": int(args.json_epochs),
                    "dim": int(args.json_dim),
                    "slots": int(args.json_slots),
                    "read_topk": int(args.json_read_topk),
                    "chunk_size": int(args.json_chunk_size),
                    "train_dataset_size": int(args.json_train_dataset_size),
                    "validation_dataset_size": int(args.json_validation_dataset_size),
                    "test_dataset_size": int(args.json_test_dataset_size),
                    "distractor_records_per_case": int(
                        args.json_distractor_records_per_case
                    ),
                    "answer_template_mode": str(args.json_answer_template_mode),
                    "generalization_score_mode": "generation",
                    **json_capabilities,
                    "mhdsra2_config_override": override,
                },
            }
            try:
                result = run_json_retrieval_generalization_test(
                    reports_dir=None,
                    epochs=args.json_epochs,
                    epochs_grid=[args.json_epochs],
                    eval_interval=args.json_eval_interval,
                    dim=args.json_dim,
                    K=args.json_slots,
                    kr_grid=[args.json_read_topk],
                    chunk_size_grid=[args.json_chunk_size],
                    lr_grid=[args.json_lr],
                    warmup_ratio_grid=[args.json_warmup_ratio],
                    scheduled_sampling_max_ratio_grid=[
                        args.json_scheduled_sampling_max_ratio
                    ],
                    train_dataset_size=args.json_train_dataset_size,
                    validation_dataset_size=args.json_validation_dataset_size,
                    test_dataset_size=args.json_test_dataset_size,
                    distractor_records_per_case=int(
                        row["config"].get("distractor_records_per_case", 0)
                    ),
                    answer_template_mode=str(
                        row["config"].get("answer_template_mode", "canonical")
                    ),
                    train_dataset_seed=seed_bundle["train_dataset_seed"],
                    validation_dataset_seed=seed_bundle["validation_dataset_seed"],
                    test_dataset_seed=seed_bundle["test_dataset_seed"],
                    pair_split_seed=seed_bundle["pair_split_seed"],
                    model_type="mhdsra2",
                    generalization_score_mode="generation",
                    local_context_size=DEFAULT_LOCAL_CONTEXT_SIZE,
                    local_context_mode=DEFAULT_LOCAL_CONTEXT_MODE,
                    final_polish_epochs=0,
                    final_generation_polish_epochs=0,
                    evidence_loss_weight=float(row["config"]["evidence_loss_weight"]),
                    slot_decoder_loss_weight=float(
                        row["config"]["slot_decoder_loss_weight"]
                    ),
                    slot_decoder_logit_bias=float(
                        row["config"]["slot_decoder_logit_bias"]
                    ),
                    evidence_hint_weight=float(row["config"]["evidence_hint_weight"]),
                    mhdsra2_config_override=override,
                    generation_readout_mode=str(
                        row["config"].get("generation_readout_mode", "model")
                    ),
                    device=device,
                )
                metrics = summarize_json_seed_result(result)
                validation_metrics, test_metrics, diagnostic_metrics = split_json_row_metrics(
                    metrics
                )
                seed_runs.append(
                    {
                        "seed_root": int(seed_root),
                        "seed_bundle": seed_bundle,
                        "metrics": metrics,
                    }
                )
                row.update(
                    {
                        "status": "completed",
                        "validation_metrics": validation_metrics,
                        "test_metrics": test_metrics,
                        "diagnostic_metrics": diagnostic_metrics,
                    }
                )
            except torch.cuda.OutOfMemoryError as exc:
                cleanup_after_oom()
                row.update({"status": "oom", "error": str(exc)})
            except RuntimeError as exc:
                if not is_oom_error(exc):
                    raise
                cleanup_after_oom()
                row.update({"status": "oom", "error": str(exc)})
            finally:
                row["elapsed_sec"] = time.perf_counter() - started_at
                rows.append(row)
        group_results[group_name] = {
            "seed_runs": seed_runs,
            "aggregate_metrics": aggregate_model_seed_runs(seed_runs),
        }
    return {"task": "json", "rows": rows, "group_results": group_results}


def run_json_row(
    args: argparse.Namespace,
    planned_row: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Run one JSON retrieval generalization row for resumable execution."""
    group_name = str(planned_row["group"])
    seed_root = int(planned_row["seed"])
    config = dict(planned_row["config"])
    seed_bundle = build_task_seed_bundle(seed_root)
    seed_everything(seed_bundle["model_seed"])
    override = group_override(group_name)
    started_at = time.perf_counter()
    row: dict[str, Any] = {
        "group": group_name,
        "task": "json",
        "seed": seed_root,
        "status": "running",
        "config": config,
    }
    try:
        result = run_json_retrieval_generalization_test(
            reports_dir=None,
            epochs=int(config["epochs"]),
            epochs_grid=[int(config["epochs"])],
            eval_interval=args.json_eval_interval,
            dim=int(config["dim"]),
            K=int(config["slots"]),
            kr_grid=[int(config["read_topk"])],
            chunk_size_grid=[int(config["chunk_size"])],
            lr_grid=[args.json_lr],
            warmup_ratio_grid=[args.json_warmup_ratio],
            scheduled_sampling_max_ratio_grid=[
                args.json_scheduled_sampling_max_ratio
            ],
            train_dataset_size=int(config["train_dataset_size"]),
            validation_dataset_size=int(config["validation_dataset_size"]),
            test_dataset_size=int(config["test_dataset_size"]),
            distractor_records_per_case=int(
                config.get("distractor_records_per_case", 0)
            ),
            answer_template_mode=str(config.get("answer_template_mode", "canonical")),
            train_dataset_seed=seed_bundle["train_dataset_seed"],
            validation_dataset_seed=seed_bundle["validation_dataset_seed"],
            test_dataset_seed=seed_bundle["test_dataset_seed"],
            pair_split_seed=seed_bundle["pair_split_seed"],
            model_type="mhdsra2",
            generalization_score_mode="generation",
            local_context_size=DEFAULT_LOCAL_CONTEXT_SIZE,
            local_context_mode=DEFAULT_LOCAL_CONTEXT_MODE,
            final_polish_epochs=0,
            final_generation_polish_epochs=0,
            evidence_loss_weight=float(config.get("evidence_loss_weight", 0.0)),
            slot_decoder_loss_weight=float(
                config.get("slot_decoder_loss_weight", 0.0)
            ),
            slot_decoder_logit_bias=float(config.get("slot_decoder_logit_bias", 0.0)),
            evidence_hint_weight=float(config.get("evidence_hint_weight", 0.0)),
            mhdsra2_config_override=override,
            generation_readout_mode=str(config.get("generation_readout_mode", "model")),
            device=device,
        )
        metrics = summarize_json_seed_result(result)
        validation_metrics, test_metrics, diagnostic_metrics = split_json_row_metrics(
            metrics
        )
        row.update(
            {
                "status": "completed",
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "diagnostic_metrics": diagnostic_metrics,
            }
        )
    except torch.cuda.OutOfMemoryError as exc:
        cleanup_after_oom()
        row.update(
            {
                "status": "oom",
                "error": str(exc),
                "validation_metrics": {},
                "test_metrics": {},
                "diagnostic_metrics": {},
            }
        )
    except RuntimeError as exc:
        if not is_oom_error(exc):
            raise
        cleanup_after_oom()
        row.update(
            {
                "status": "oom",
                "error": str(exc),
                "validation_metrics": {},
                "test_metrics": {},
                "diagnostic_metrics": {},
            }
        )
    finally:
        row["elapsed_sec"] = time.perf_counter() - started_at
    return row


def run_two_digit_section(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Run the two-digit arithmetic retention grid once as a baseline holdout."""
    started_at = time.perf_counter()
    payload = build_two_digit_diagnostic_grid_payload(
        datasets=(TWO_DIGIT_ONLY,),
        layer_counts=args.two_digit_layers,
        max_steps_per_stage_values=args.two_digit_steps,
        learning_rates=args.two_digit_learning_rates,
        training_strategies=args.two_digit_strategies,
        seeds=args.two_digit_seeds,
        replay_ratio=args.two_digit_replay_ratio,
        stage_patience=args.two_digit_stage_patience,
        two_digit_replay_ratios=(DEFAULT_TWO_DIGIT_REPLAY_RATIO,),
        stage_loss_weights=DEFAULT_TWO_DIGIT_STAGE_LOSS_WEIGHTS,
        device=device,
    )
    rows = [
        {
            "group": "baseline_holdout",
            "task": "two_digit",
            "seed": row["seed"],
            "status": "completed",
            "config": {
                "dataset": row["dataset_name"],
                "training_strategy": row["training_strategy"],
                "learning_rate": row["learning_rate"],
                "max_steps_per_stage": row["max_steps_per_stage"],
                "num_layers": row["num_layers"],
            },
            "validation_metrics": {},
            "test_metrics": {
                "two_digit_exact_match": _two_digit_exact_match(row),
            },
            "diagnostic_metrics": {
                "slot_collision": {
                    "available": False,
                    "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux",
                }
            },
        }
        for row in payload["runs"]
    ]
    return {
        "task": "two_digit",
        "elapsed_sec": time.perf_counter() - started_at,
        "rows": rows,
        "payload": payload,
        "note": (
            "Arithmetic retention uses the current baseline arithmetic factory. "
            "The retrieval ablation switches are not force-applied because this task "
            "does not exercise external paged retrieval in the current model factory."
        ),
    }


def run_two_digit_row(
    args: argparse.Namespace,
    device: torch.device,
    planned_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one two-digit arithmetic retention row for resumable execution."""
    seed = int(planned_row["seed"])
    config = dict(planned_row["config"])
    started_at = time.perf_counter()
    row: dict[str, Any] = {
        "group": "baseline_holdout",
        "task": "two_digit",
        "seed": seed,
        "status": "running",
        "config": config,
    }
    dataset_spec = select_two_digit_diagnostic_dataset_specs((str(config["dataset"]),))[0]
    diagnostic_run = run_one_two_digit_diagnostic_grid_point(
        dataset_spec=dataset_spec,
        training_strategy=str(config["training_strategy"]),
        learning_rate=float(config["learning_rate"]),
        max_steps_per_stage=int(config["max_steps_per_stage"]),
        num_layers=int(config["num_layers"]),
        seed=seed,
        replay_ratio=args.two_digit_replay_ratio,
        stage_patience=args.two_digit_stage_patience,
        two_digit_replay_ratio=DEFAULT_TWO_DIGIT_REPLAY_RATIO,
        stage_loss_weights=DEFAULT_TWO_DIGIT_STAGE_LOSS_WEIGHTS,
        device=device,
    )
    serialized = serialize_two_digit_diagnostic_run(diagnostic_run)
    row.update(
        {
            "status": "completed",
            "validation_metrics": {},
            "test_metrics": {
                "two_digit_exact_match": _two_digit_exact_match(serialized),
            },
            "diagnostic_metrics": {
                "slot_collision": {
                    "available": False,
                    "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux",
                }
            },
            "metrics": serialized,
            "elapsed_sec": time.perf_counter() - started_at,
        }
    )
    return row


def _two_digit_exact_match(row: Mapping[str, Any]) -> float:
    run = row.get("run", {})
    if not isinstance(run, Mapping):
        return 0.0
    for stage in run.get("stage_exact_matches", ()):
        if isinstance(stage, Mapping) and stage.get("stage_name") == "two_digit_rules":
            return float(stage.get("exact_match", 0.0))
    return float(run.get("train_exact_match", 0.0))


def evaluate_success(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Compute conservative success flags without using test for selection."""
    rows = payload.get("rows", [])
    niah_rows = [
        row for row in rows if row.get("task") == "niah" and row.get("status") == "completed"
    ]
    json_rows = [
        row for row in rows if row.get("task") == "json" and row.get("status") == "completed"
    ]
    two_digit_rows = [
        row
        for row in rows
        if row.get("task") == "two_digit" and row.get("status") == "completed"
    ]
    return {
        "niah_rows_completed": len(niah_rows),
        "json_rows_completed": len(json_rows),
        "two_digit_rows_completed": len(two_digit_rows),
        "two_digit_min_exact_match": min(
            (
                float(row.get("test_metrics", {}).get("two_digit_exact_match", 0.0))
                for row in two_digit_rows
            ),
            default=None,
        ),
        "selection_policy": (
            "Select candidates by validation metrics only; inspect held-out test "
            "metrics only after a candidate improves validation."
        ),
    }


def build_markdown(payload: Mapping[str, Any]) -> list[str]:
    """Render a compact Markdown report for the ablation payload."""
    report_name = str(payload.get("name") or DEFAULT_REPORT_NAME)
    lines = [
        f"# {report_name}",
        "",
        f"- device: `{payload['config']['device']}`",
        f"- groups: `{', '.join(payload['config']['groups'])}`",
        f"- tasks: `{', '.join(payload['config']['tasks'])}`",
        f"- dry_run: `{payload['config']['dry_run']}`",
        "",
        "## Groups",
        "",
    ]
    for group_name in payload["config"]["groups"]:
        group = GROUP_CONFIGS[group_name]
        lines.append(
            f"- `{group_name}`: {group['description']} "
            f"override=`{json.dumps(group['override'], sort_keys=True)}`"
        )
    lines.extend(["", "## Rows", ""])
    lines.append(
        "| task | group | seed | status | config | validation | test | elapsed_sec |"
    )
    lines.append("|---|---|---:|---|---|---|---|---:|")
    for row in payload.get("rows", []):
        config = json.dumps(row.get("config", {}), sort_keys=True)
        validation = json.dumps(row.get("validation_metrics", {}), sort_keys=True)
        test = json.dumps(row.get("test_metrics", {}), sort_keys=True)
        elapsed = row.get("elapsed_sec")
        elapsed_text = "" if elapsed is None else f"{float(elapsed):.2f}"
        lines.append(
            f"| {row.get('task')} | {row.get('group')} | {row.get('seed', '')} | "
            f"{row.get('status')} | `{config}` | `{validation}` | `{test}` | "
            f"{elapsed_text} |"
        )
    lines.extend(["", "## Diagnostics", ""])
    lines.append("| task | group | seed | diagnostics |")
    lines.append("|---|---|---:|---|")
    for row in payload.get("rows", []):
        diagnostics = json.dumps(row.get("diagnostic_metrics", {}), sort_keys=True)
        lines.append(
            f"| {row.get('task')} | {row.get('group')} | {row.get('seed', '')} | "
            f"`{diagnostics}` |"
        )
    lines.extend(["", "## Success Summary", ""])
    for key, value in payload.get("success_summary", {}).items():
        lines.append(f"- {key}: `{value}`")
    return lines


def save_reports(payload: dict[str, Any], reports_dir: Path | str) -> tuple[Path, Path]:
    """Write JSON and Markdown reports for the unified ablation."""
    resolved_reports_dir = ensure_reports_dir(reports_dir)
    report_name = str(payload.get("name") or DEFAULT_REPORT_NAME)
    json_path = resolved_reports_dir / f"{report_name}.json"
    markdown_path = resolved_reports_dir / f"{report_name}.md"
    write_json(json_path, payload)
    write_markdown(markdown_path, build_markdown(payload))
    return json_path, markdown_path


def build_sections_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group completed rows into report sections."""
    sections: dict[str, dict[str, Any]] = {}
    for task_name in ALLOWED_TASKS:
        task_rows = [dict(row) for row in rows if row.get("task") == task_name]
        if task_rows:
            sections[task_name] = {"task": task_name, "rows": task_rows}
    return sections


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the unified quality-improvement ablation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "docs" / "reports")
    parser.add_argument("--report-name", default=DEFAULT_REPORT_NAME)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--groups", type=parse_csv_strings, default=DEFAULT_GROUPS)
    parser.add_argument("--tasks", type=parse_csv_strings, default=DEFAULT_TASKS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="JSONL row checkpoint for resumable full-grid runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in --checkpoint-path.",
    )

    parser.add_argument("--smoke-batch-sizes", type=parse_csv_ints, default=(1, 4, 8))
    parser.add_argument("--smoke-tokens", type=parse_csv_ints, default=(256, 1024))
    parser.add_argument("--smoke-page-size", type=int, default=64)
    parser.add_argument("--smoke-top-pages", type=int, default=4)
    parser.add_argument("--smoke-max-tokens", type=int, default=8)
    parser.add_argument("--smoke-seed", type=int, default=20260602)

    parser.add_argument("--niah-seq-lengths", type=parse_csv_ints, default=(8192,))
    parser.add_argument("--niah-seeds", type=parse_csv_ints, default=(101, 202, 303))
    parser.add_argument("--niah-vocab-size", type=int, default=100)
    parser.add_argument("--niah-dim", type=int, default=64)
    parser.add_argument("--niah-layers", type=int, default=2)
    parser.add_argument("--niah-slots", type=int, default=64)
    parser.add_argument("--niah-read-topk", type=int, default=8)
    parser.add_argument("--niah-chunk-size", type=int, default=1024)
    parser.add_argument("--niah-batch-size", type=int, default=1)
    parser.add_argument("--niah-epochs", type=int, default=60)
    parser.add_argument("--niah-lr", type=float, default=1e-3)
    parser.add_argument("--niah-robust-eval-interval", type=int, default=20)
    parser.add_argument("--niah-needle-loss-alpha", type=float, default=0.5)
    parser.add_argument("--niah-hidden-mse-alpha", type=float, default=0.0)
    parser.add_argument("--niah-eval-batches-per-depth", type=int, default=2)
    parser.add_argument(
        "--niah-test-batches-per-depth",
        type=int,
        default=DEFAULT_NIAH_TEST_BATCHES_PER_DEPTH,
        help=(
            "Held-out NIAH test batches per depth. Default 0 keeps historical "
            "reports unchanged; set >0 to report test metrics after validation."
        ),
    )

    parser.add_argument("--json-task-seed-roots", type=parse_csv_ints, default=(7, 11, 19))
    parser.add_argument("--json-epochs", type=int, default=80)
    parser.add_argument("--json-eval-interval", type=int, default=10)
    parser.add_argument("--json-dim", type=int, default=128)
    parser.add_argument("--json-slots", type=int, default=128)
    parser.add_argument("--json-read-topk", type=int, default=32)
    parser.add_argument("--json-chunk-size", type=int, default=256)
    parser.add_argument("--json-lr", type=float, default=5e-4)
    parser.add_argument("--json-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--json-scheduled-sampling-max-ratio", type=float, default=0.0)
    parser.add_argument("--json-train-dataset-size", type=int, default=12)
    parser.add_argument("--json-validation-dataset-size", type=int, default=4)
    parser.add_argument("--json-test-dataset-size", type=int, default=4)
    parser.add_argument("--json-distractor-records-per-case", type=int, default=0)
    parser.add_argument(
        "--json-answer-template-mode",
        choices=("canonical", "mixed"),
        default="canonical",
    )

    parser.add_argument("--two-digit-layers", type=parse_csv_ints, default=(4, 8))
    parser.add_argument("--two-digit-steps", type=parse_csv_ints, default=(512,))
    parser.add_argument("--two-digit-learning-rates", type=parse_csv_floats, default=(0.01,))
    parser.add_argument(
        "--two-digit-strategies",
        type=parse_csv_strings,
        default=("baseline", "two_digit_replay", "two_digit_weighted_loss", "combined"),
    )
    parser.add_argument("--two-digit-seeds", type=parse_csv_ints, default=(101, 202, 303))
    parser.add_argument("--two-digit-replay-ratio", type=float, default=0.75)
    parser.add_argument("--two-digit-stage-patience", type=int, default=3)
    return parser


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    """Run or dry-run the full quality-improvement ablation."""
    groups = tuple(args.groups)
    tasks = tuple(args.tasks)
    for task in tasks:
        if task not in ALLOWED_TASKS:
            allowed = ", ".join(ALLOWED_TASKS)
            raise ValueError(f"unknown task {task!r}; allowed: {allowed}")
    for group_name in groups:
        group_override(group_name)
    device = resolve_device(args.device)
    reports_dir = ensure_reports_dir(args.reports_dir)
    report_name = str(getattr(args, "report_name", DEFAULT_REPORT_NAME))
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = reports_dir / f"{report_name}.checkpoint.jsonl"
    payload: dict[str, Any] = {
        "name": report_name,
        "config": {
            "device": str(device),
            "groups": list(groups),
            "tasks": list(tasks),
            "dry_run": bool(args.dry_run),
            "checkpoint_path": str(checkpoint_path),
        },
        "planned_rows": build_run_rows(groups=groups, tasks=tasks, args=args),
        "rows": [],
        "sections": {},
    }
    if args.dry_run:
        payload["success_summary"] = evaluate_success(payload)
        return payload

    completed_by_key = load_checkpoint_rows(checkpoint_path) if args.resume else {}
    rows: list[dict[str, Any]] = []
    for planned_row in payload["planned_rows"]:
        key = row_key(planned_row)
        if key in completed_by_key:
            rows.append(completed_by_key[key])
            continue
        task_name = planned_row["task"]
        if task_name == "smoke":
            section = run_smoke_section(args, device)
            row = section["rows"][0]
        elif task_name == "niah":
            row = run_niah_row(args, device, planned_row)
        elif task_name == "json":
            row = run_json_row(args, planned_row, device)
        elif task_name == "two_digit":
            row = run_two_digit_row(args, device, planned_row)
        else:
            raise ValueError(f"unknown planned task: {task_name}")
        rows.append(row)
        append_checkpoint_row(checkpoint_path, row)
        payload["rows"] = rows
        payload["sections"] = build_sections_from_rows(rows)
        payload["success_summary"] = evaluate_success(payload)
        save_reports(payload, reports_dir)
        print(
            "MHDSRA2_QUALITY_ABLATION_ROW="
            f"{len(rows)}/{len(payload['planned_rows'])} "
            f"task={row.get('task')} group={row.get('group')} "
            f"seed={row.get('seed')} status={row.get('status')}",
            flush=True,
        )

    payload["rows"] = rows
    payload["sections"] = build_sections_from_rows(rows)
    payload["success_summary"] = evaluate_success(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point for MHDSRA2 quality ablation reports."""
    args = build_parser().parse_args(argv)
    payload = run_ablation(args)
    json_path, markdown_path = save_reports(payload, args.reports_dir)
    print(f"MHDSRA2_QUALITY_ABLATION_JSON={json_path}")
    print(f"MHDSRA2_QUALITY_ABLATION_MARKDOWN={markdown_path}")
    print(f"MHDSRA2_QUALITY_ABLATION_ROWS={len(payload['rows'])}")
    print(f"MHDSRA2_QUALITY_ABLATION_DRY_RUN={payload['config']['dry_run']}")
    return payload


if __name__ == "__main__":
    main()
