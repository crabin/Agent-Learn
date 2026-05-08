"""Reward functions for math-reasoning Agentic RL."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


class RewardCallable(Protocol):
    def __call__(self, completions: list[str], **kwargs: Any) -> list[float]:
        ...


def extract_answer(text: str) -> str:
    """Extract a likely final numeric answer from a model completion."""

    if text is None:
        return ""

    patterns = [
        r"Final\s*Answer\s*[:：]\s*([^\n<|]+)",
        r"####\s*([^\n<|]+)",
        r"答案\s*[:：]\s*([^\n<|]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, str(text), flags=re.IGNORECASE)
        if match:
            return _clean_text_answer(match.group(1))

    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(text))
    if numbers:
        return _clean_text_answer(numbers[-1])
    return str(text).strip()


def _clean_text_answer(value: str) -> str:
    text = str(value).strip().replace(",", "")
    text = re.sub(r"<\|.*$", "", text).strip()
    return text.rstrip(".。")


def _to_float(value: Any) -> float | None:
    text = _clean_text_answer(str(value))
    try:
        return float(text)
    except ValueError:
        fraction_match = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", text)
        if fraction_match:
            denominator = float(fraction_match.group(2))
            if denominator != 0:
                return float(fraction_match.group(1)) / denominator
        return None


def answers_equal(prediction: Any, ground_truth: Any, tolerance: float = 1e-6) -> bool:
    """Compare answers with numeric tolerance when possible."""

    pred_num = _to_float(prediction)
    truth_num = _to_float(ground_truth)
    if pred_num is not None and truth_num is not None:
        return math.isclose(pred_num, truth_num, rel_tol=tolerance, abs_tol=tolerance)
    return _clean_text_answer(str(prediction)).lower() == _clean_text_answer(
        str(ground_truth)
    ).lower()


def count_reasoning_steps(completion: str) -> int:
    """Count visible reasoning-step markers in a completion."""

    text = str(completion)
    explicit_steps = re.findall(r"\bStep\s*\d+\b|步骤\s*\d+", text, flags=re.IGNORECASE)
    if explicit_steps:
        return len(explicit_steps)
    equation_lines = [
        line
        for line in text.splitlines()
        if "=" in line and re.search(r"\d", line)
    ]
    return len(equation_lines)


@dataclass
class MathRewardFunction:
    """Base reward function for math completions."""

    tolerance: float = 1e-6

    def reward_one(self, completion: str, ground_truth: Any) -> float:
        raise NotImplementedError

    def __call__(self, completions: list[str], **kwargs: Any) -> list[float]:
        ground_truths = _resolve_ground_truths(kwargs, expected=len(completions))
        return [
            float(self.reward_one(completion, truth))
            for completion, truth in zip(completions, ground_truths)
        ]

    def is_correct(self, completion: str, ground_truth: Any) -> bool:
        return answers_equal(
            extract_answer(completion),
            ground_truth,
            tolerance=self.tolerance,
        )


@dataclass
class AccuracyReward(MathRewardFunction):
    """Binary reward: 1.0 when the final answer is correct, else 0.0."""

    def reward_one(self, completion: str, ground_truth: Any) -> float:
        return 1.0 if self.is_correct(completion, ground_truth) else 0.0


@dataclass
class LengthPenaltyReward(MathRewardFunction):
    """Reward correct answers while penalizing overly long completions."""

    max_length: int = 1024
    penalty_weight: float = 0.001

    def reward_one(self, completion: str, ground_truth: Any) -> float:
        if not self.is_correct(completion, ground_truth):
            return 0.0
        overflow = max(0, len(str(completion)) - self.max_length)
        return max(0.0, 1.0 - self.penalty_weight * overflow)


@dataclass
class StepReward(MathRewardFunction):
    """Reward correct answers with a small bonus for clear reasoning steps."""

    step_bonus: float = 0.1
    max_bonus_steps: int = 5

    def reward_one(self, completion: str, ground_truth: Any) -> float:
        if not self.is_correct(completion, ground_truth):
            return 0.0
        steps = min(count_reasoning_steps(completion), self.max_bonus_steps)
        return 1.0 + self.step_bonus * steps


@dataclass
class CombinedReward(MathRewardFunction):
    """Weighted composition of multiple reward functions."""

    components: list[tuple[MathRewardFunction, float]] = field(default_factory=list)

    def reward_one(self, completion: str, ground_truth: Any) -> float:
        if not self.components:
            return AccuracyReward(tolerance=self.tolerance).reward_one(
                completion, ground_truth
            )
        return sum(
            weight * reward.reward_one(completion, ground_truth)
            for reward, weight in self.components
        )


def _resolve_ground_truths(kwargs: dict[str, Any], expected: int) -> list[Any]:
    values = (
        kwargs.get("ground_truth")
        or kwargs.get("ground_truths")
        or kwargs.get("answer")
        or kwargs.get("answers")
    )
    if values is None:
        raise ValueError("Reward functions require `ground_truth` values.")
    if isinstance(values, (str, int, float)):
        return [values] * expected
    ground_truths = list(values)
    if len(ground_truths) != expected:
        raise ValueError(
            f"Expected {expected} ground-truth values, got {len(ground_truths)}."
        )
    return ground_truths


def _component_from_config(config: dict[str, Any]) -> tuple[MathRewardFunction, float]:
    weight = float(config.get("weight", 1.0))
    reward_config = dict(config)
    reward_type = str(reward_config.pop("type"))
    reward_config.pop("weight", None)
    return create_reward_function(reward_type, **reward_config), weight


def create_reward_function(reward_type: str = "accuracy", **config: Any) -> MathRewardFunction:
    """Create a reward function by name."""

    normalized = reward_type.lower()
    if normalized == "accuracy":
        return AccuracyReward(tolerance=float(config.get("tolerance", 1e-6)))
    if normalized == "length_penalty":
        max_length = int(config.get("max_length", config.get("target_length", 1024)))
        return LengthPenaltyReward(
            tolerance=float(config.get("tolerance", 1e-6)),
            max_length=max_length,
            penalty_weight=float(config.get("penalty_weight", 0.001)),
        )
    if normalized == "step":
        return StepReward(
            tolerance=float(config.get("tolerance", 1e-6)),
            step_bonus=float(config.get("step_bonus", 0.1)),
            max_bonus_steps=int(config.get("max_bonus_steps", 5)),
        )
    if normalized == "combined":
        raw_components: Iterable[dict[str, Any]] = config.get(
            "components",
            [
                {"type": "accuracy", "weight": 1.0},
                {"type": "length_penalty", "weight": 0.5},
                {"type": "step", "weight": 0.3},
            ],
        )
        return CombinedReward(
            tolerance=float(config.get("tolerance", 1e-6)),
            components=[_component_from_config(item) for item in raw_components],
        )
    raise ValueError(f"Unsupported reward_type: {reward_type}")
