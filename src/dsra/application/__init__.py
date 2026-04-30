"""Application services and unit-of-work boundaries for DSRA attention."""

from .arithmetic_emergence_service import (
    DecimalArithmeticTokenizer,
    GeneratedArithmeticAnswer,
    build_carry_diagnostic_grid_markdown,
    build_carry_diagnostic_grid_payload,
    build_carry_diagnostic_dataset_specs,
    build_curriculum_arithmetic_spec,
    build_curriculum_strategy_grid_markdown,
    build_curriculum_strategy_grid_payload,
    build_default_arithmetic_spec,
    build_layer_emergence_markdown,
    build_layer_emergence_payload,
    build_two_digit_diagnostic_grid_markdown,
    build_two_digit_diagnostic_grid_payload,
    build_unit_with_carry_only_spec,
    is_exact_generated_answer,
    select_two_digit_diagnostic_dataset_specs,
)
from .attention_unit_of_work import StreamingAttentionUnitOfWork
from .json_retrieval_search_service import JsonRetrievalSearchService
from .retrieval_model_factory import RetrievalModelFactory, RetrievalModelBuilder

__all__ = [
    "DecimalArithmeticTokenizer",
    "GeneratedArithmeticAnswer",
    "JsonRetrievalSearchService",
    "RetrievalModelBuilder",
    "RetrievalModelFactory",
    "StreamingAttentionUnitOfWork",
    "build_carry_diagnostic_dataset_specs",
    "build_carry_diagnostic_grid_markdown",
    "build_carry_diagnostic_grid_payload",
    "build_curriculum_arithmetic_spec",
    "build_curriculum_strategy_grid_markdown",
    "build_curriculum_strategy_grid_payload",
    "build_default_arithmetic_spec",
    "build_layer_emergence_markdown",
    "build_layer_emergence_payload",
    "build_two_digit_diagnostic_grid_markdown",
    "build_two_digit_diagnostic_grid_payload",
    "build_unit_with_carry_only_spec",
    "is_exact_generated_answer",
    "select_two_digit_diagnostic_dataset_specs",
]
