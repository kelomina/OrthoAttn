"""Domain objects for DSRA attention configuration."""

from .attention_spec import AttentionLayerSpec, select_mhdsra2_heads
from .arithmetic_emergence import (
    ArithmeticCurriculumStage,
    ArithmeticEmergenceResult,
    ArithmeticExample,
    ArithmeticRuleDatasetSpec,
)
from .model_spec import ARCHIVED_MODEL_ALIASES, RetrievalModelSpec, normalize_model_type

from .mqar import MQARConfig, generate_mqar_batch

__all__ = [
    "ARCHIVED_MODEL_ALIASES",
    "ArithmeticCurriculumStage",
    "ArithmeticEmergenceResult",
    "ArithmeticExample",
    "ArithmeticRuleDatasetSpec",
    "AttentionLayerSpec",
    "MQARConfig",
    "RetrievalModelSpec",
    "generate_mqar_batch",
    "normalize_model_type",
    "select_mhdsra2_heads",
]

