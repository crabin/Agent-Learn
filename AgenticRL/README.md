# AgenticRL

`AgenticRL` 是这个仓库里关于 Agentic Reinforcement Learning 的实验目录。它参考了 Datawhale HelloAgents 第十一章的结构，把数学推理任务中的数据准备、奖励函数、监督微调和强化学习训练整理成了一套本地可运行的最小实现。

这里的核心目标不是单纯让模型“背答案”，而是让模型逐步学会更像一个解题 agent 那样工作：先形成稳定的推理格式，再通过奖励机制鼓励更好的解题行为。

## 什么是 Agentic RL

在这个目录里，`Agentic RL` 可以理解为：

1. 先用 `SFT` 教模型学会基本输出格式和基础推理方式。
2. 再用 `GRPO` 这类强化学习方法，根据奖励函数继续优化模型行为。

和普通监督学习相比，它关注的不只是“最后答对没”，还会关注：

- 推理过程是否清晰
- 输出是否符合预期格式
- 回答是否过长
- 行为模式是否更稳定

比如同一道题，如果两个模型都答对了，能清楚写出步骤、格式稳定、输出不过度冗长的那个，通常会拿到更高奖励。

## 目录结构

```text
AgenticRL/
├── __init__.py
├── README.md
├── datasets.py              # 数据集加载与 SFT/RL 格式转换
├── rewards.py               # 奖励函数定义
├── trainers.py              # SFT / GRPO 训练封装
├── tool.py                  # 统一工具入口 RLTrainingTool
├── pipeline.py              # 完整训练流水线
├── quick_test.py            # 不依赖大模型下载的轻量验证
├── mdp_rl.py                # 经典强化学习概念演示
└── CliffWalking_Q-learning.py
```

其中前六个文件组成了 Agentic RL 的主要实现；`mdp_rl.py` 和 `CliffWalking_Q-learning.py` 更偏传统强化学习入门示例。

## 模块说明

### `datasets.py`

负责把数学题数据整理成模型可训练的格式。

- `GSM8KDataset`
  加载 GSM8K 数据集
- `format_math_dataset()`
  把样本转换成 SFT 或 RL 所需格式
- `create_sft_dataset()`
  生成监督微调数据
- `create_rl_dataset()`
  生成强化学习数据

当前默认假设样本长这样：

```python
{
    "question": "What is 2 + 2?",
    "answer": "2 + 2 = 4. #### 4"
}
```

转换后：

- `SFT` 格式会包含 `prompt + completion + text`
- `RL` 格式会保留 `prompt + ground_truth`

### `rewards.py`

这里定义了训练时用于打分的奖励函数。

- `AccuracyReward`
  答案正确给 1，错误给 0
- `LengthPenaltyReward`
  对正确答案施加长度惩罚，避免输出过长
- `StepReward`
  对包含清晰步骤的回答给额外奖励
- `CombinedReward`
  把多个奖励函数按权重组合起来

这部分是 Agentic RL 的关键，因为强化学习最终优化的就是这些奖励信号。

### `trainers.py`

封装了两类训练器：

- `SFTTrainerWrapper`
  用监督数据对模型做微调
- `GRPOTrainerWrapper`
  在已有模型上用奖励函数继续训练

这里依赖 Hugging Face 生态：

- `transformers`
- `trl`
- `peft`
- `torch`
- `datasets`
- `accelerate`

默认支持 LoRA 参数高效微调。

### `tool.py`

`RLTrainingTool` 是统一入口。它把常用动作包装成一个工具接口：

- `load_dataset`
- `create_reward`
- `train`
- `evaluate`

这让你既可以从 Python 里直接调，也可以把它当成一个上层训练 orchestration 接口来用。

### `pipeline.py`

这里定义了完整训练流程 `AgenticRLPipeline`，默认按下面顺序执行：

1. 加载数据
2. 进行 SFT 训练
3. 评估 SFT 模型
4. 进行 GRPO 训练
5. 评估 GRPO 模型
6. 保存结果

如果当前目录没有 `config.json`，它会自动生成一个默认配置。

## 训练流程怎么理解

可以把这套流程理解成两段式训练：

### 第一阶段：SFT

模型先通过标准答案学习：

- 输入格式长什么样
- 推理过程大概怎么写
- 最终答案应该怎么落到统一格式

这一步像是在给模型打基础。

### 第二阶段：GRPO

模型已经具备基本解题能力后，再通过奖励函数继续优化：

- 答案是否正确
- 步骤是否清楚
- 输出是否简洁

这一步更像是在做行为调优，而不是单纯拟合训练文本。

## 快速上手

### 1. 轻量验证

这个命令不会下载大模型，只会验证工具入口和奖励逻辑：

```bash
python -m AgenticRL.quick_test
```

它会做两件事：

- 创建一个 `accuracy` 奖励函数
- 用几条手工预测结果做评估

### 2. 安装训练依赖

如果要做真实 SFT / GRPO 训练，需要安装 `rl` extra：

```bash
uv sync --extra rl
```

如果你不用 `uv`，也可以自己安装这些包：

```bash
pip install torch transformers datasets trl peft accelerate tensorboard
```

### 3. 运行完整流水线

```bash
python -m AgenticRL.pipeline
```

默认情况下：

- 基础模型是 `Qwen/Qwen3-0.6B`
- 数据集是 `gsm8k`
- SFT 输出目录是 `./models/sft_model`
- GRPO 输出目录是 `./models/grpo_model`

## 模型和数据下载位置

这里需要区分“缓存下载位置”和“训练产物输出位置”。

### Hugging Face 缓存位置

模型和数据集默认会下载到 Hugging Face 缓存目录，通常是：

```text
~/.cache/huggingface/hub
~/.cache/huggingface/datasets
```

在这台机器上通常对应：

```text
/Users/lpb/.cache/huggingface/hub
/Users/lpb/.cache/huggingface/datasets
```

可以通过环境变量修改：

```bash
export HF_HOME=/Users/lpb/workspace/myProjects/AgentLearn/.hf_cache
```

或者分别设置：

```bash
export HF_HUB_CACHE=/path/to/model_cache
export HF_DATASETS_CACHE=/path/to/dataset_cache
```

### 训练输出位置

训练产生的模型参数、LoRA 权重和 tokenizer 会保存到配置里的 `output_dir`：

- `./models/sft_model`
- `./models/grpo_model`

也就是相对于当前仓库：

```text
/Users/lpb/workspace/myProjects/AgentLearn/models/sft_model
/Users/lpb/workspace/myProjects/AgentLearn/models/grpo_model
```

## 示例

### 使用 `RLTrainingTool` 做轻量评估

```python
import json
from AgenticRL import RLTrainingTool

tool = RLTrainingTool()

result = tool.run(
    {
        "action": "evaluate",
        "predictions": [
            "Step 1: 48 / 2 = 24\nStep 2: 48 + 24 = 72\nFinal Answer: 72"
        ],
        "ground_truth": ["72"],
        "return_details": True,
    }
)

print(json.loads(result))
```

### 创建组合奖励函数

```python
from AgenticRL.rewards import create_reward_function

reward = create_reward_function(
    "combined",
    components=[
        {"type": "accuracy", "weight": 1.0},
        {"type": "step", "weight": 0.1},
        {"type": "length_penalty", "weight": 0.3},
    ],
)
```

## 当前边界

这份实现已经把第十一章里的主要结构落到了本地，但它目前更适合教学、实验和小规模流程验证，不是生产级训练平台。需要注意：

- 真实训练依赖外部模型下载和合适算力
- 默认流程主要围绕数学题推理任务设计
- 奖励函数仍是规则型实现，不是 learned reward model
- 评估逻辑目前以准确率和格式统计为主

如果后面要继续扩展，一个自然方向是补上：

- 更多数据集适配器
- 更丰富的 reward shaping
- 多卡训练配置
- 训练日志与结果可视化
