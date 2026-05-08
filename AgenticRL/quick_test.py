"""Quick smoke test for the local AgenticRL API."""

from __future__ import annotations

import json

from AgenticRL import RLTrainingTool


def main() -> None:
    rl_tool = RLTrainingTool()

    reward_result = json.loads(
        rl_tool.run({"action": "create_reward", "reward_type": "accuracy"})
    )
    print("奖励函数:", reward_result)

    eval_result = json.loads(
        rl_tool.run(
            {
                "action": "evaluate",
                "predictions": [
                    "Step 1: 48 / 2 = 24\nStep 2: 48 + 24 = 72\nFinal Answer: 72",
                    "Final Answer: 70",
                ],
                "ground_truth": ["72", "72"],
                "return_details": True,
            }
        )
    )
    print("轻量评估:", json.dumps(eval_result, ensure_ascii=False, indent=2))

    print(
        "\n如需真实 SFT/GRPO 训练，请安装 rl extra 后运行 "
        "`python -m AgenticRL.pipeline`。"
    )


if __name__ == "__main__":
    main()
