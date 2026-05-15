"""Evaluation compatibility exports for local HelloAgents examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from AgentEvaluation import BFCLDataset as _BFCLDataset
from AgentEvaluation import BFCLEvaluator as _BFCLEvaluator


class BFCLDataset(_BFCLDataset):
    """BFCL dataset loader compatible with the chapter examples."""

    def __init__(
        self,
        bfcl_data_dir: str | Path | None = None,
        category: str = "simple_python",
        data: list[dict[str, Any]] | None = None,
        data_path: str | Path | None = None,
    ):
        self.bfcl_data_dir = Path(bfcl_data_dir) if bfcl_data_dir else None
        self.category = category
        resolved_path = data_path or self._resolve_category_path()
        super().__init__(data=data, data_path=resolved_path)

    def _resolve_category_path(self) -> Path | None:
        if self.bfcl_data_dir is None:
            return None
        candidates = [
            self.bfcl_data_dir / f"BFCL_v4_{self.category}.json",
            self.bfcl_data_dir / f"{self.category}.json",
            self.bfcl_data_dir / f"BFCL_v4_{self.category}.jsonl",
            self.bfcl_data_dir / f"{self.category}.jsonl",
        ]
        return next((path for path in candidates if path.exists()), candidates[0])


class BFCLEvaluator(_BFCLEvaluator):
    """BFCL evaluator compatible with the chapter examples."""

    def __init__(
        self,
        dataset: BFCLDataset,
        category: str = "simple_python",
        evaluation_mode: str = "ast",
    ):
        self.evaluation_mode = evaluation_mode
        super().__init__(dataset=dataset, category=category)

    def export_to_bfcl_format(self, results: dict[str, Any], output_path: str | Path) -> Path:
        return self.export_predictions(results, output_path)


__all__ = ["BFCLDataset", "BFCLEvaluator"]
