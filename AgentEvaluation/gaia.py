"""GAIA-style general-assistant evaluation."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import EvaluationResult, RunnableAgent, accuracy, limited, write_markdown_report


GAIA_SYSTEM_PROMPT = """You are a general AI assistant. Report your thoughts, and finish your answer with:
FINAL ANSWER: [YOUR FINAL ANSWER].

Use a number, a short string, or a comma separated list of numbers/strings."""


class GAIADataset:
    """Load GAIA-like samples from memory or a local JSON/JSONL file."""

    def __init__(
        self,
        data: list[dict[str, Any]] | None = None,
        data_path: str | Path | None = None,
        level: int | None = None,
    ):
        self.data = data or []
        self.data_path = Path(data_path) if data_path else None
        self.level = level

    def load(self) -> list[dict[str, Any]]:
        if not self.data and self.data_path is not None:
            text = self.data_path.read_text(encoding="utf-8")
            if self.data_path.suffix == ".jsonl":
                self.data = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                loaded = json.loads(text)
                self.data = loaded if isinstance(loaded, list) else loaded.get("data", [])

        if self.level is None:
            return self.data
        return [item for item in self.data if int(item.get("Level", item.get("level", 0))) == self.level]


class QuasiExactMatcher:
    """GAIA-inspired answer normalizer and matcher."""

    def match(self, predicted: str, expected: str) -> bool:
        return self.normalize(predicted) == self.normalize(expected)

    def normalize(self, answer: Any) -> str:
        if answer is None:
            return ""
        text = str(answer).strip().strip("[]")
        numberish = text.replace("$", "").replace("%", "").replace("€", "").replace("£", "")
        if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", numberish):
            return self._normalize_single(text)
        if "," in text:
            parts = [self._normalize_single(part) for part in text.split(",")]
            return ",".join(sorted(part for part in parts if part))
        return self._normalize_single(text)

    def _normalize_single(self, text: str) -> str:
        text = text.strip().lower()
        text = text.replace("$", "").replace("%", "").replace("€", "").replace("£", "")
        text = re.sub(r"(?<=\d),(?=\d)", "", text)
        text = " ".join(text.split())
        words = text.split()
        if words and words[0] in {"the", "a", "an"}:
            text = " ".join(words[1:])
        text = text.rstrip(".,;:!?")
        if re.fullmatch(r"-?\d+\.0+", text):
            text = text.split(".")[0]
        return text


class GAIAEvaluator:
    """Evaluate final-answer quality with GAIA-style normalization."""

    def __init__(self, dataset: GAIADataset, level: int | None = None):
        self.dataset = dataset
        self.level = level
        self.matcher = QuasiExactMatcher()

    def evaluate(self, agent: RunnableAgent, max_samples: int | None = None) -> dict[str, Any]:
        samples = limited(self.dataset.load(), max_samples)
        details: list[EvaluationResult] = []

        for sample in samples:
            prompt = self._build_prompt(sample)
            response = agent.run(prompt)
            predicted = extract_final_answer(response)
            expected = sample.get("Final answer", sample.get("final_answer", ""))
            is_correct = self.matcher.match(predicted, expected)
            details.append(
                EvaluationResult(
                    sample_id=str(sample.get("task_id", sample.get("id", len(details)))),
                    prediction=predicted,
                    expected=expected,
                    is_correct=is_correct,
                    metadata={
                        "level": int(sample.get("Level", sample.get("level", 0))),
                        "response": response,
                    },
                )
            )

        return self._summarize(agent, details)

    def export_to_gaia_format(
        self,
        results: dict[str, Any],
        output_path: str | Path,
        include_reasoning: bool = True,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for item in results.get("details", []):
            row = {"task_id": item["task_id"], "model_answer": item["prediction"]}
            if include_reasoning:
                row["reasoning_trace"] = item.get("response", "")
            lines.append(json.dumps(row, ensure_ascii=False))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def generate_report(self, results: dict[str, Any], output_path: str | Path) -> Path:
        rows = [
            {
                "task_id": item["task_id"],
                "level": item["level"],
                "correct": item["exact_match"],
                "prediction": item["prediction"],
                "expected": item["expected"],
            }
            for item in results.get("details", [])[:20]
        ]
        metrics = {
            "agent": results["agent_name"],
            "exact_match_rate": f"{results['exact_match_rate']:.2%}",
            "exact_matches": f"{results['exact_matches']}/{results['total_samples']}",
        }
        return write_markdown_report("GAIA 评估报告", metrics, rows, output_path)

    def _build_prompt(self, sample: dict[str, Any]) -> str:
        question = sample.get("Question", sample.get("question", ""))
        file_name = sample.get("file_name") or sample.get("file_path") or ""
        attachment = f"\n附件: {file_name}" if file_name else ""
        return f"{GAIA_SYSTEM_PROMPT}\n\nQuestion: {question}{attachment}"

    def _summarize(self, agent: RunnableAgent, details: list[EvaluationResult]) -> dict[str, Any]:
        level_stats: dict[int, list[EvaluationResult]] = defaultdict(list)
        for item in details:
            level_stats[int(item.metadata.get("level", 0))].append(item)
        level_metrics = {level: accuracy(items) for level, items in level_stats.items()}
        return {
            "agent_name": getattr(agent, "name", agent.__class__.__name__),
            "level_filter": self.level,
            "total_samples": len(details),
            "exact_matches": sum(1 for item in details if item.is_correct),
            "exact_match_rate": accuracy(details),
            "level_metrics": level_metrics,
            "details": [
                {
                    "task_id": item.sample_id,
                    "level": item.metadata.get("level"),
                    "prediction": item.prediction,
                    "expected": item.expected,
                    "exact_match": item.is_correct,
                    "response": item.metadata.get("response"),
                }
                for item in details
            ],
        }


class GAIAEvaluationTool:
    """Small tool wrapper for GAIA-style evaluation."""

    def run(self, params: dict[str, Any]) -> str:
        agent = params["agent"]
        dataset = GAIADataset(
            data=params.get("data"),
            data_path=params.get("data_path"),
            level=params.get("level"),
        )
        evaluator = GAIAEvaluator(dataset=dataset, level=params.get("level"))
        results = evaluator.evaluate(agent, max_samples=params.get("max_samples"))
        if params.get("output_path"):
            evaluator.export_to_gaia_format(results, params["output_path"])
        if params.get("report_path"):
            evaluator.generate_report(results, params["report_path"])
        return json.dumps(results, ensure_ascii=False, indent=2)


def extract_final_answer(response: str) -> str:
    patterns = [
        r"FINAL ANSWER:\s*(.+?)(?:\n|$)",
        r"Final answer[：:]\s*(.+?)(?:\n|$)",
        r"最终答案[：:]\s*(.+?)(?:\n|$)",
        r"答案[：:]\s*(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip().strip("[]")
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else response.strip()
