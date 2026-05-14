"""Shared helpers for local agent evaluation examples."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol


class RunnableAgent(Protocol):
    """Minimal protocol used by evaluators in this package."""

    name: str

    def run(self, prompt: str) -> str:
        """Return the agent response for one prompt."""


@dataclass(frozen=True)
class EvaluationResult:
    """One evaluated sample."""

    sample_id: str
    prediction: Any
    expected: Any
    is_correct: bool
    metadata: dict[str, Any]


def accuracy(results: list[EvaluationResult]) -> float:
    """Compute the exact accuracy for evaluated samples."""

    if not results:
        return 0.0
    return sum(1 for item in results if item.is_correct) / len(results)


def limited(items: list[dict[str, Any]], max_samples: int | None) -> list[dict[str, Any]]:
    """Return a sample prefix; 0 and None mean all samples."""

    if max_samples in (None, 0):
        return items
    return items[:max_samples]


def write_markdown_report(
    title: str,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write a compact Markdown report for an evaluation run."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 指标",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")

    if rows:
        headers = list(rows[0].keys())
        lines.extend(["", "## 样本", ""])
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            cells = [str(row.get(header, "")).replace("\n", "<br>") for header in headers]
            lines.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
