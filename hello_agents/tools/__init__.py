"""Tool compatibility exports for local HelloAgents examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hello_agents.evaluation import BFCLDataset, BFCLEvaluator


class BFCLEvaluationTool:
    """One-call BFCL evaluation helper compatible with the chapter examples."""

    def __init__(
        self,
        bfcl_data_dir: str | Path | None = None,
        output_dir: str | Path = "evaluation_results/bfcl_official",
        report_dir: str | Path = "evaluation_reports",
    ):
        self.bfcl_data_dir = Path(bfcl_data_dir) if bfcl_data_dir else None
        self.output_dir = Path(output_dir)
        self.report_dir = Path(report_dir)

    def run(
        self,
        agent: Any,
        category: str = "simple_python",
        max_samples: int | None = 5,
        data: list[dict[str, Any]] | None = None,
        data_path: str | Path | None = None,
        export_results: bool = True,
        generate_report: bool = True,
        run_official_eval: bool = False,
        model_name: str = "local-rule-llm",
    ) -> dict[str, Any]:
        dataset = BFCLDataset(
            bfcl_data_dir=self.bfcl_data_dir,
            category=category,
            data=data,
            data_path=data_path,
        )
        evaluator = BFCLEvaluator(dataset=dataset, category=category)
        results = evaluator.evaluate(agent, max_samples=max_samples)
        results["category"] = category
        results["model_name"] = model_name
        results["official_eval_enabled"] = run_official_eval

        if export_results:
            output_path = self.output_dir / f"BFCL_v4_{category}_result.json"
            evaluator.export_to_bfcl_format(results, output_path)
            results["bfcl_result_path"] = str(output_path)

        if generate_report:
            report_path = self.report_dir / f"bfcl_report_{category}.md"
            evaluator.generate_report(results, report_path)
            results["report_path"] = str(report_path)

        return results


class SearchTool:
    """Minimal placeholder SearchTool for chapter snippets."""

    name = "search"
    description = "Return a local placeholder search result."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def run(self, query: str) -> str:
        return json.dumps({"query": query, "result": "local search placeholder"}, ensure_ascii=False)


__all__ = ["BFCLEvaluationTool", "SearchTool"]
