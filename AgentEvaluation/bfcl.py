"""BFCL-style function-call evaluation.

This is a lightweight local implementation inspired by BFCL. It loads JSON or
JSONL samples, asks an agent to produce function calls, and compares predicted
calls with ground truth using AST-normalized arguments.
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import EvaluationResult, RunnableAgent, accuracy, limited, write_markdown_report


FunctionCall = dict[str, Any]


class BFCLDataset:
    """Load BFCL-like samples from memory or a local JSON/JSONL file."""

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


class ASTMatcher:
    """Compare function calls while ignoring argument order and simple expression form."""

    def calls_match(self, predicted: list[FunctionCall], expected: list[FunctionCall]) -> bool:
        if len(predicted) != len(expected):
            return False

        unmatched = list(expected)
        for pred_call in predicted:
            match_index = next(
                (
                    index
                    for index, expected_call in enumerate(unmatched)
                    if self.single_call_match(pred_call, expected_call)
                ),
                None,
            )
            if match_index is None:
                return False
            unmatched.pop(match_index)
        return not unmatched

    def single_call_match(self, predicted: FunctionCall, expected: FunctionCall) -> bool:
        pred_name, pred_args = normalize_call(predicted)
        exp_name, exp_args = normalize_call(expected)
        if pred_name != exp_name:
            return False
        if set(pred_args) != set(exp_args):
            return False
        return all(self._values_equal(pred_args[key], exp_args[key]) for key in pred_args)

    def _values_equal(self, predicted: Any, expected: Any) -> bool:
        return self._value_key(predicted) == self._value_key(expected)

    def _value_key(self, value: Any) -> Any:
        if isinstance(value, str):
            expression = value.strip()
            try:
                parsed = ast.parse(expression, mode="eval")
                literal = ast.literal_eval(parsed)
                return ("literal", literal)
            except (SyntaxError, ValueError):
                return ("string", expression)
        try:
            parsed = ast.parse(repr(value), mode="eval")
            return ("ast", ast.dump(parsed, include_attributes=False))
        except SyntaxError:
            return ("repr", repr(value))


class BFCLEvaluator:
    """Evaluate an agent on BFCL-like function-calling samples."""

    def __init__(self, dataset: BFCLDataset, category: str | None = None):
        self.dataset = dataset
        self.category = category
        self.matcher = ASTMatcher()

    def evaluate(self, agent: RunnableAgent, max_samples: int | None = None) -> dict[str, Any]:
        samples = limited(self.dataset.load(), max_samples)
        details: list[EvaluationResult] = []

        for sample in samples:
            prompt = self._build_prompt(sample)
            response = agent.run(prompt)
            prediction = extract_function_calls(response)
            expected = sample.get("ground_truth") or sample.get("expected") or []
            is_correct = self.matcher.calls_match(prediction, expected)
            details.append(
                EvaluationResult(
                    sample_id=str(sample.get("id", len(details))),
                    prediction=prediction,
                    expected=expected,
                    is_correct=is_correct,
                    metadata={"category": sample.get("category", self.category), "response": response},
                )
            )

        return self._summarize(agent, details)

    def export_predictions(self, results: dict[str, Any], output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"id": item["id"], "result": item["prediction"]}
            for item in results.get("details", [])
        ]
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def generate_report(self, results: dict[str, Any], output_path: str | Path) -> Path:
        rows = [
            {
                "id": item["id"],
                "category": item["category"],
                "correct": item["is_correct"],
                "prediction": item["prediction"],
                "expected": item["expected"],
            }
            for item in results.get("details", [])[:20]
        ]
        metrics = {
            "agent": results["agent_name"],
            "accuracy": f"{results['overall_accuracy']:.2%}",
            "correct": f"{results['correct_samples']}/{results['total_samples']}",
        }
        return write_markdown_report("BFCL 评估报告", metrics, rows, output_path)

    def _build_prompt(self, sample: dict[str, Any]) -> str:
        functions = json.dumps(sample.get("functions", sample.get("function", [])), ensure_ascii=False, indent=2)
        return (
            "请根据用户问题选择需要调用的函数，并只输出 JSON 函数调用列表。\n"
            f"用户问题: {sample.get('question', '')}\n"
            f"可用函数: {functions}"
        )

    def _summarize(self, agent: RunnableAgent, details: list[EvaluationResult]) -> dict[str, Any]:
        category_stats: dict[str, list[EvaluationResult]] = defaultdict(list)
        for item in details:
            category_stats[str(item.metadata.get("category") or "unknown")].append(item)
        return {
            "agent_name": getattr(agent, "name", agent.__class__.__name__),
            "total_samples": len(details),
            "correct_samples": sum(1 for item in details if item.is_correct),
            "overall_accuracy": accuracy(details),
            "category_accuracy": {
                category: accuracy(items) for category, items in category_stats.items()
            },
            "details": [
                {
                    "id": item.sample_id,
                    "category": item.metadata.get("category"),
                    "prediction": item.prediction,
                    "expected": item.expected,
                    "is_correct": item.is_correct,
                    "response": item.metadata.get("response"),
                }
                for item in details
            ],
        }


class BFCLEvaluationTool:
    """Small tool wrapper for BFCL-style evaluation."""

    def run(self, params: dict[str, Any]) -> str:
        agent = params["agent"]
        dataset = BFCLDataset(data=params.get("data"), data_path=params.get("data_path"))
        evaluator = BFCLEvaluator(dataset=dataset, category=params.get("category"))
        results = evaluator.evaluate(agent, max_samples=params.get("max_samples"))
        if params.get("output_path"):
            evaluator.export_predictions(results, params["output_path"])
        if params.get("report_path"):
            evaluator.generate_report(results, params["report_path"])
        return json.dumps(results, ensure_ascii=False, indent=2)


def normalize_call(call: FunctionCall) -> tuple[str, dict[str, Any]]:
    name = call.get("name") or call.get("function") or call.get("function_name")
    arguments = call.get("arguments") or call.get("args") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return str(name), dict(arguments)


def extract_function_calls(response: str) -> list[FunctionCall]:
    """Extract function calls from JSON, code blocks, or plain Python-call text."""

    response = response.strip()
    calls = _extract_json_calls(response)
    if calls:
        return calls

    code_blocks = re.findall(r"```(?:python|json)?\s*(.*?)```", response, flags=re.DOTALL)
    for block in code_blocks:
        calls.extend(_extract_json_calls(block))
        calls.extend(_extract_python_calls(block))
    if calls:
        return calls

    return _extract_python_calls(response)


def _extract_json_calls(text: str) -> list[FunctionCall]:
    candidates = [text]
    candidates.extend(re.findall(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return [parsed] if ("name" in parsed or "function_name" in parsed) else []
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def _extract_python_calls(text: str) -> list[FunctionCall]:
    calls: list[FunctionCall] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return calls

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        arguments = {}
        for index, arg in enumerate(node.args):
            arguments[f"arg{index}"] = _safe_literal(arg)
        for keyword in node.keywords:
            if keyword.arg:
                arguments[keyword.arg] = _safe_literal(keyword.value)
        calls.append({"name": name, "arguments": arguments})
    return calls


def _safe_literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except ValueError:
        return ast.unparse(node)
