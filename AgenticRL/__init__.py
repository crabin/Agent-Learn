"""Agentic RL utilities for math reasoning experiments.

This package mirrors the Chapter 11 Agentic-RL examples from HelloAgents in a
small, local form: dataset formatting, reward functions, trainer wrappers, and
a unified ``RLTrainingTool`` entry point.
"""

from .datasets import (
    GSM8KDataset,
    create_rl_dataset,
    create_sft_dataset,
    extract_gsm8k_answer,
    format_math_dataset,
)
from .rewards import (
    AccuracyReward,
    CombinedReward,
    LengthPenaltyReward,
    MathRewardFunction,
    StepReward,
    create_reward_function,
)
from .tool import RLTrainingTool

__all__ = [
    "AccuracyReward",
    "CombinedReward",
    "GSM8KDataset",
    "LengthPenaltyReward",
    "MathRewardFunction",
    "RLTrainingTool",
    "StepReward",
    "create_reward_function",
    "create_rl_dataset",
    "create_sft_dataset",
    "extract_gsm8k_answer",
    "format_math_dataset",
]
