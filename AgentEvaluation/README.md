# AgentEvaluation

`AgentEvaluation` 是从《智能体性能评估》笔记整理出的本地代码版本，聚焦三个评估方向：

- `BFCL`：工具调用能力评估，使用 AST 风格匹配函数名和参数。
- `GAIA`：通用助手能力评估，使用准精确匹配归一化最终答案。
- `data_generation`：生成数据质量评估，包含 LLM Judge 与 Win Rate。

## 快速验证

```bash
python -m AgentEvaluation.quick_test
```

该命令不需要联网，也不需要真实 LLM，会使用内置的模拟 Agent 跑通三类评估。

## BFCL 风格评估

```python
from AgentEvaluation import BFCLDataset, BFCLEvaluator

dataset = BFCLDataset(data=[
    {
        "id": "simple_001",
        "question": "What's the weather like in Beijing today?",
        "functions": [{"name": "get_weather"}],
        "ground_truth": [
            {"name": "get_weather", "arguments": {"location": "Beijing"}}
        ],
    }
])

results = BFCLEvaluator(dataset).evaluate(agent, max_samples=10)
```

`agent` 只需要实现 `run(prompt: str) -> str`。预测结果可以是 JSON、JSON 代码块，或 Python 函数调用文本。

## GAIA 风格评估

```python
from AgentEvaluation import GAIADataset, GAIAEvaluator

dataset = GAIADataset(data=[
    {
        "task_id": "gaia_001",
        "Question": "What is 40 + 2?",
        "Level": 1,
        "Final answer": "42",
    }
])

results = GAIAEvaluator(dataset).evaluate(agent)
```

评估器会从模型回复中提取 `FINAL ANSWER: ...`，并做大小写、冠词、数字逗号、末尾标点等归一化。

## 数据生成质量评估

```python
from AgentEvaluation import LLMJudgeEvaluator, WinRateEvaluator

judge_results = LLMJudgeEvaluator(judge_agent).evaluate_batch(generated_problems)
win_results = WinRateEvaluator(judge_agent).evaluate(
    generated_problems,
    reference_problems,
)
```

`judge_agent` 同样只需要实现 `run(prompt: str) -> str`，并返回 JSON 评分或比较结果。

## 数据集配置

文档中提到的所有基准和数据集都集中注册在 `dataset_config.py`：

```python
from AgentEvaluation import (
    agent_validation_plan,
    dataset_validation_matrix,
    get_dataset_config,
    get_validation_method_config,
    list_dataset_configs,
    list_validation_method_configs,
    validate_dataset_config,
    validation_methods_for_dataset,
)

print([config.key for config in list_dataset_configs()])
print(get_dataset_config("gaia"))
print(get_validation_method_config("llm_judge"))
print([method.key for method in validation_methods_for_dataset("aime25")])
print([method.key for method in list_validation_method_configs()])
print(validate_dataset_config("gaia", project_root="."))
print(agent_validation_plan())
print(dataset_validation_matrix())
```

当前注册项：

- `bfcl`
- `toolbench`
- `api_bank`
- `gaia`
- `agentbench`
- `webarena`
- `chateval`
- `sotopia`
- `aime_1983_2025`
- `aime25`

其中 `bfcl`、`gaia`、`aime_1983_2025`、`aime25` 已映射到本项目的轻量加载器/评估器；其他环境型基准保留官方来源、子集、环境变量和本地路径提示，方便后续接入官方 harness。

当前注册的验证方法：

- `ast_match`：BFCL 工具调用匹配。
- `quasi_exact_match`：GAIA 最终答案匹配。
- `llm_judge`：LLM 多维评分。
- `win_rate`：生成样本与参考样本成对对比。
- `human_verification`：人工验证配置占位，适合最终质量把关。
- `official_harness`：官方评估环境配置占位。
