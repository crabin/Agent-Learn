"""Generated-data quality evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from .common import RunnableAgent, write_markdown_report


JUDGE_DIMENSIONS = ("correctness", "clarity", "difficulty_match", "completeness")


class AIDataset:
    """Load generated or reference problem datasets from local JSON/JSONL files."""

    def __init__(self, data: list[dict[str, Any]] | None = None, data_path: str | Path | None = None):
        self.data = data or []
        self.data_path = Path(data_path) if data_path else None

    def load(self) -> list[dict[str, Any]]:
        if self.data:
            return self.data
        if self.data_path is None:
            return []
        text = self.data_path.read_text(encoding="utf-8")
        if self.data_path.suffix == ".jsonl":
            self.data = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            loaded = json.loads(text)
            self.data = loaded if isinstance(loaded, list) else loaded.get("data", [])
        return self.data


class LLMJudgeEvaluator:
    """Score generated problems on four 1-5 dimensions using a judge agent."""

    def __init__(self, judge_agent: RunnableAgent):
        self.judge_agent = judge_agent

    def evaluate_batch(
        self,
        generated_problems: list[dict[str, Any]],
        max_samples: int | None = None,
    ) -> dict[str, Any]:
        samples = generated_problems if max_samples in (None, 0) else generated_problems[:max_samples]
        details = []
        for problem in samples:
            response = self.judge_agent.run(self._build_prompt(problem))
            scores = parse_judge_scores(response)
            average_score = mean(scores[dimension] for dimension in JUDGE_DIMENSIONS)
            details.append(
                {
                    "problem_id": problem.get("problem_id", problem.get("id", len(details))),
                    "scores": scores,
                    "average_score": average_score,
                    "comments": scores.get("comments", ""),
                    "judge_response": response,
                }
            )
        return summarize_judge_results(details)

    def _build_prompt(self, problem: dict[str, Any]) -> str:
        return (
            "请评估以下 AIME 风格数学题目的质量，并只输出 JSON。\n"
            f"题目: {problem.get('problem', '')}\n"
            f"答案: {problem.get('answer', '')}\n"
            f"解答: {problem.get('solution', '')}\n"
            "维度: correctness, clarity, difficulty_match, completeness，分数为 1-5。"
        )


class WinRateEvaluator:
    """Pairwise quality comparison between generated and reference problems."""

    def __init__(self, judge_agent: RunnableAgent):
        self.judge_agent = judge_agent

    def evaluate(
        self,
        generated_problems: list[dict[str, Any]],
        reference_problems: list[dict[str, Any]],
        num_comparisons: int | None = None,
    ) -> dict[str, Any]:
        total = num_comparisons or min(len(generated_problems), len(reference_problems))
        details = []
        for index in range(total):
            generated = generated_problems[index % len(generated_problems)]
            reference = reference_problems[index % len(reference_problems)]
            response = self.judge_agent.run(self._build_prompt(generated, reference))
            winner = parse_winner(response)
            details.append(
                {
                    "comparison_id": index,
                    "generated_id": generated.get("problem_id", generated.get("id", index)),
                    "reference_id": reference.get("problem_id", reference.get("id", index)),
                    "winner": winner,
                    "judge_response": response,
                }
            )
        return summarize_win_rate(details)

    def _build_prompt(self, generated: dict[str, Any], reference: dict[str, Any]) -> str:
        return (
            "请比较两个 AIME 风格数学题目的质量，并只输出 JSON: "
            "{\"winner\": \"A\" 或 \"B\" 或 \"Tie\", \"reason\": \"...\"}\n"
            f"题目A: {generated.get('problem', '')}\n"
            f"答案A: {generated.get('answer', '')}\n"
            f"解答A: {generated.get('solution', '')}\n"
            f"题目B: {reference.get('problem', '')}\n"
            f"答案B: {reference.get('answer', '')}\n"
            f"解答B: {reference.get('solution', '')}"
        )


class LLMJudgeTool:
    """Tool wrapper for generated-data absolute scoring."""

    def __init__(self, judge_agent: RunnableAgent):
        self.judge_agent = judge_agent

    def run(self, params: dict[str, Any]) -> str:
        generated = AIDataset(data=params.get("generated_data"), data_path=params.get("generated_data_path")).load()
        results = LLMJudgeEvaluator(self.judge_agent).evaluate_batch(
            generated,
            max_samples=params.get("max_samples"),
        )
        if params.get("report_path"):
            write_judge_report(results, params["report_path"])
        return json.dumps(results, ensure_ascii=False, indent=2)


class WinRateTool:
    """Tool wrapper for generated-vs-reference pairwise evaluation."""

    def __init__(self, judge_agent: RunnableAgent):
        self.judge_agent = judge_agent

    def run(self, params: dict[str, Any]) -> str:
        generated = AIDataset(data=params.get("generated_data"), data_path=params.get("generated_data_path")).load()
        reference = AIDataset(data=params.get("reference_data"), data_path=params.get("reference_data_path")).load()
        results = WinRateEvaluator(self.judge_agent).evaluate(
            generated,
            reference,
            num_comparisons=params.get("num_comparisons"),
        )
        if params.get("report_path"):
            write_win_rate_report(results, params["report_path"])
        return json.dumps(results, ensure_ascii=False, indent=2)


def parse_judge_scores(response: str) -> dict[str, Any]:
    parsed = _parse_json_object(response)
    scores: dict[str, Any] = {"comments": parsed.get("comments", "")}
    for dimension in JUDGE_DIMENSIONS:
        value = int(parsed.get(dimension, 0))
        scores[dimension] = max(1, min(5, value))
    return scores


def parse_winner(response: str) -> str:
    parsed = _parse_json_object(response)
    winner = str(parsed.get("winner", "Tie")).strip()
    return winner if winner in {"A", "B", "Tie"} else "Tie"


def summarize_judge_results(details: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(details)
    average_score = mean(item["average_score"] for item in details) if details else 0.0
    dimension_scores = {
        dimension: mean(item["scores"][dimension] for item in details) if details else 0.0
        for dimension in JUDGE_DIMENSIONS
    }
    return {
        "total_samples": total,
        "average_score": average_score,
        "pass_rate": _rate(details, lambda item: item["average_score"] >= 3.5),
        "excellent_rate": _rate(details, lambda item: item["average_score"] >= 4.5),
        "dimension_scores": dimension_scores,
        "details": details,
    }


def summarize_win_rate(details: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(details)
    wins = sum(1 for item in details if item["winner"] == "A")
    losses = sum(1 for item in details if item["winner"] == "B")
    ties = sum(1 for item in details if item["winner"] == "Tie")
    return {
        "total_comparisons": total,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / total if total else 0.0,
        "loss_rate": losses / total if total else 0.0,
        "tie_rate": ties / total if total else 0.0,
        "details": details,
    }


def write_judge_report(results: dict[str, Any], output_path: str | Path) -> Path:
    rows = [
        {
            "problem_id": item["problem_id"],
            "average_score": f"{item['average_score']:.2f}",
            "scores": item["scores"],
        }
        for item in results.get("details", [])[:20]
    ]
    metrics = {
        "average_score": f"{results['average_score']:.2f}/5.00",
        "pass_rate": f"{results['pass_rate']:.2%}",
        "excellent_rate": f"{results['excellent_rate']:.2%}",
    }
    return write_markdown_report("LLM Judge 评估报告", metrics, rows, output_path)


def write_win_rate_report(results: dict[str, Any], output_path: str | Path) -> Path:
    metrics = {
        "win_rate": f"{results['win_rate']:.2%}",
        "loss_rate": f"{results['loss_rate']:.2%}",
        "tie_rate": f"{results['tie_rate']:.2%}",
        "comparisons": results["total_comparisons"],
    }
    return write_markdown_report("Win Rate 评估报告", metrics, results.get("details", [])[:20], output_path)


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
    return {}


def _rate(items: list[dict[str, Any]], predicate) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if predicate(item)) / len(items)
