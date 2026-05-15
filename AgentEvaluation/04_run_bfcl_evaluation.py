"""Command-line BFCL evaluation demo compatible with the chapter section."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hello_agents import HelloAgentsLLM, SimpleAgent  # noqa: E402
from hello_agents.tools import BFCLEvaluationTool  # noqa: E402


DEMO_DATA = [
    {
        "id": "simple_python_0",
        "category": "simple_python",
        "question": "What's the weather like in Beijing in celsius?",
        "functions": [
            {
                "name": "get_weather",
                "description": "Get weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string"},
                    },
                    "required": ["location", "unit"],
                },
            }
        ],
        "ground_truth": [
            {"name": "get_weather", "arguments": {"location": "Beijing", "unit": "celsius"}}
        ],
    }
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local BFCL-style evaluation.")
    parser.add_argument("--category", default="simple_python")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--model-name", default="local-rule-llm")
    parser.add_argument("--bfcl-data-dir", default=None)
    args = parser.parse_args()

    llm = HelloAgentsLLM(model=args.model_name)
    agent = SimpleAgent(name="TestAgent", llm=llm)
    tool = BFCLEvaluationTool(bfcl_data_dir=args.bfcl_data_dir)
    results = tool.run(
        agent=agent,
        category=args.category,
        max_samples=args.samples,
        data=None if args.bfcl_data_dir else DEMO_DATA,
        model_name=args.model_name,
    )

    print(f"准确率: {results['overall_accuracy']:.2%}")
    print(f"正确数: {results['correct_samples']}/{results['total_samples']}")
    print(f"结果文件: {results.get('bfcl_result_path')}")
    print(f"报告文件: {results.get('report_path')}")


if __name__ == "__main__":
    main()
