"""Smoke test for the local AgentEvaluation package."""

from __future__ import annotations

import json

from AgentEvaluation import (
    BFCLDataset,
    BFCLEvaluator,
    GAIADataset,
    GAIAEvaluator,
    LLMJudgeEvaluator,
    WinRateEvaluator,
    agent_validation_plan,
    list_dataset_configs,
)


class EchoAgent:
    name = "EchoAgent"

    def run(self, prompt: str) -> str:
        if "get_weather" in prompt:
            return '[{"name": "get_weather", "arguments": {"unit": "celsius", "location": "Beijing"}}]'
        if "California" in prompt:
            return "I checked the question.\nFINAL ANSWER: 12847521"
        return "FINAL ANSWER: unknown"


class JudgeAgent:
    name = "JudgeAgent"

    def run(self, prompt: str) -> str:
        if "题目B" in prompt:
            return json.dumps({"winner": "Tie", "reason": "质量接近"}, ensure_ascii=False)
        return json.dumps(
            {
                "correctness": 5,
                "clarity": 4,
                "difficulty_match": 4,
                "completeness": 5,
                "comments": "结构完整，答案明确",
            },
            ensure_ascii=False,
        )


def main() -> None:
    agent = EchoAgent()

    bfcl_dataset = BFCLDataset(
        data=[
            {
                "id": "simple_001",
                "category": "simple",
                "question": "What's the weather like in Beijing today?",
                "functions": [{"name": "get_weather"}],
                "ground_truth": [
                    {"name": "get_weather", "arguments": {"location": "Beijing", "unit": "celsius"}}
                ],
            }
        ]
    )
    bfcl_results = BFCLEvaluator(bfcl_dataset).evaluate(agent)
    print("BFCL:", json.dumps(bfcl_results, ensure_ascii=False, indent=2))

    gaia_dataset = GAIADataset(
        data=[
            {
                "task_id": "gaia_001",
                "Question": "What is the total population of the top 3 most populous cities in California?",
                "Level": 2,
                "Final answer": "12,847,521",
            }
        ]
    )
    gaia_results = GAIAEvaluator(gaia_dataset).evaluate(agent)
    print("GAIA:", json.dumps(gaia_results, ensure_ascii=False, indent=2))

    generated = [
        {
            "problem_id": "gen_001",
            "problem": "Find x if 2x + 3 = 11.",
            "answer": 4,
            "solution": "2x = 8, so x = 4.",
        }
    ]
    reference = [
        {
            "problem_id": "ref_001",
            "problem": "Find y if y + 5 = 9.",
            "answer": 4,
            "solution": "y = 4.",
        }
    ]
    judge = JudgeAgent()
    judge_results = LLMJudgeEvaluator(judge).evaluate_batch(generated)
    print("LLM Judge:", json.dumps(judge_results, ensure_ascii=False, indent=2))
    win_results = WinRateEvaluator(judge).evaluate(generated, reference)
    print("Win Rate:", json.dumps(win_results, ensure_ascii=False, indent=2))

    configs = list_dataset_configs()
    print("数据集配置数量:", len(configs))
    print("Agent 验证计划:", json.dumps(agent_validation_plan(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
