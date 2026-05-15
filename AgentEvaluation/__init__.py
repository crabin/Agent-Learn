"""Local agent evaluation utilities based on the Chapter 12 notes.

The package provides three small evaluation tracks:

- BFCL-style function-call evaluation with AST-based argument matching.
- GAIA-style final-answer evaluation with quasi exact matching.
- Generated-data quality evaluation with LLM Judge and Win Rate helpers.
"""

from .bfcl import BFCLDataset, BFCLEvaluationTool, BFCLEvaluator, ASTMatcher
from .data_generation import (
    AIDataset,
    LLMJudgeEvaluator,
    LLMJudgeTool,
    WinRateEvaluator,
    WinRateTool,
)
from .dataset_config import (
    DATASET_CONFIGS,
    VALIDATION_METHOD_CONFIGS,
    DatasetConfig,
    DatasetValidation,
    ValidationMethodConfig,
    agent_validation_plan,
    dataset_validation_matrix,
    export_dataset_configs,
    get_dataset_config,
    get_validation_method_config,
    list_dataset_configs,
    list_validation_method_configs,
    validate_all_dataset_configs,
    validate_dataset_config,
    validation_methods_for_dataset,
)
from .gaia import GAIA_SYSTEM_PROMPT, GAIADataset, GAIAEvaluationTool, GAIAEvaluator, QuasiExactMatcher

__all__ = [
    "AIDataset",
    "ASTMatcher",
    "BFCLDataset",
    "BFCLEvaluationTool",
    "BFCLEvaluator",
    "DATASET_CONFIGS",
    "DatasetConfig",
    "DatasetValidation",
    "GAIAEvaluationTool",
    "GAIAEvaluator",
    "GAIA_SYSTEM_PROMPT",
    "GAIADataset",
    "LLMJudgeEvaluator",
    "LLMJudgeTool",
    "QuasiExactMatcher",
    "VALIDATION_METHOD_CONFIGS",
    "ValidationMethodConfig",
    "WinRateEvaluator",
    "WinRateTool",
    "agent_validation_plan",
    "dataset_validation_matrix",
    "export_dataset_configs",
    "get_dataset_config",
    "get_validation_method_config",
    "list_dataset_configs",
    "list_validation_method_configs",
    "validate_all_dataset_configs",
    "validate_dataset_config",
    "validation_methods_for_dataset",
]
