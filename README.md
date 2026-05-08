# AgentLearn

AI Agent 学习项目，当前围绕 **ReAct**、**CodeAct**、**RAG** 与 **AgenticRL** 四条主线组织示例代码和 Notebook，用来帮助理解 Agent 推理执行、代码即行动、检索增强生成，以及基于 SFT/GRPO 的 Agentic RL 训练流程。

## 项目结构

```text
AgentLearn/
├── main.py                  # 项目入口占位示例
├── pyproject.toml           # 项目依赖声明
├── CodeAct/                 # CodeAct (Code as Action) 实现
├── AgenticRL/               # Agentic RL 数据集、奖励、训练封装与流水线
├── react/                   # ReAct (Reasoning + Acting) 实现
└── RAG/
    └── chapter07-RAG/       # RAG 脚本与 Notebook 示例
```

## 功能概览

### ReAct

基于 [ReAct](https://arxiv.org/abs/2210.03629) 思路实现。Agent 通过 **推理 → 调用工具 → 观察结果 → 继续推理** 的循环完成任务。

- 支持工具调用型问题求解
- 使用 Pydantic 定义工具参数 schema
- 内置 `Calculator` 工具示例，演示结构化工具执行
- 提供 `react/mian.py` 作为最小可运行入口

**核心流程：** 用户提问 → LLM 推理 → 选择工具并传参 → 获取观察结果 → 重复直到得出最终答案

### CodeAct

基于 [CodeAct](https://arxiv.org/abs/2402.01030) 思路实现。Agent 将 **代码本身作为行动**，模型输出 Python 代码并交给沙箱执行。

- LLM 生成 Python 代码块作为中间行动
- 内置 `Sandbox` 执行环境并捕获标准输出
- 沙箱暴露基础数学函数 `sum/sub/mul/div`
- 提供 `CodeAct/main.py` 演示从问题到代码执行的完整流程

**核心流程：** 用户提问 → LLM 生成 Python 代码 → Sandbox 执行 → 返回结果

### RAG

`RAG/chapter07-RAG` 目录包含检索增强生成的入门示例，覆盖从文档加载、切分、嵌入到向量检索的基础流程，并配套 Notebook 用于实验。

- 使用 `TextLoader` 加载本地文档
- 使用 `CharacterTextSplitter` 做文本切分
- 使用 `OpenAIEmbeddings` 接入 OpenAI 兼容 embeddings 接口
- 使用 `FAISS` 构建向量索引并执行检索
- 包含 6 个循序渐进的 Notebook 示例

**核心流程：** 加载文档 → 切分文本 → 生成向量 → 构建索引 → 检索相关内容

### AgenticRL

基于 Datawhale HelloAgents 第十一章 Agentic-RL 的结构落地到本地 `AgenticRL/` 目录，提供从 GSM8K 数据格式化、奖励函数到 SFT/GRPO 训练封装的最小可用框架。

- `AgenticRL/datasets.py`：`GSM8KDataset`、`format_math_dataset()`、`create_sft_dataset()`、`create_rl_dataset()`
- `AgenticRL/rewards.py`：`AccuracyReward`、`LengthPenaltyReward`、`StepReward`、`CombinedReward`
- `AgenticRL/trainers.py`：`SFTTrainerWrapper`、`GRPOTrainerWrapper`，训练依赖 Hugging Face Transformers、TRL 与 PEFT
- `AgenticRL/tool.py`：统一入口 `RLTrainingTool`，支持 `load_dataset`、`create_reward`、`train`、`evaluate`
- `AgenticRL/pipeline.py`：端到端训练流水线示例

轻量验证不需要下载模型：

```bash
python -m AgenticRL.quick_test
```

真实训练需要额外安装 RL 依赖，并准备可用 GPU/模型下载环境：

```bash
uv sync --extra rl
python -m AgenticRL.pipeline
```

也可以用统一工具做快速评估：

```python
import json
from AgenticRL import RLTrainingTool

rl_tool = RLTrainingTool()
result = rl_tool.run({
    "action": "evaluate",
    "predictions": ["Step 1: 48 + 24 = 72\nFinal Answer: 72"],
    "ground_truth": ["72"],
})
print(json.loads(result)["accuracy"])
```

## 安装

### 依赖要求

- Python 3.11+
- 推荐使用 `uv`

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/crabin/Agent-Learn.git
cd Agent-Learn

# 2. 安装 pyproject.toml 中声明的依赖
uv sync

# 3. 激活环境（可选）
source .venv/bin/activate

# 4. 安装当前示例额外依赖
pip install openai pydantic python-dotenv \
    langchain-openai langchain-text-splitters faiss-cpu
```

当前仓库已经提供 `pyproject.toml`，但部分示例依赖尚未完全收敛到该文件中；如果运行示例时报缺包，请优先补齐上述依赖。

## 环境变量配置

项目中的 Agent 与 RAG 示例依赖兼容 OpenAI 协议的模型或嵌入接口，常用环境变量如下：

```bash
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://your-endpoint
OPENAI_MODEL=your_chat_model
OPENAI_EMBEDDING_MODEL=nomic-embed-text:latest
DEBUG=1
```

说明：

- `OPENAI_API_KEY`：模型或向量接口所需的密钥
- `OPENAI_API_BASE`：可选，自定义兼容 OpenAI 协议的服务地址时使用；`react/ChatOpenAI.py` 会自动补齐 `/v1`，`CodeAct/ChatOpenAI.py` 直接使用该值
- `OPENAI_MODEL`：ReAct 与 CodeAct 示例运行所需的聊天模型名称
- `OPENAI_EMBEDDING_MODEL`：可选，RAG 示例中的嵌入模型名称，默认是 `nomic-embed-text:latest`
- `DEBUG`：可选，CodeAct 示例的调试开关，默认开启

## 使用

### 运行 CodeAct 示例

```bash
python CodeAct/main.py
```

### 运行 ReAct 示例

```bash
python react/mian.py
```

### 运行 RAG 脚本示例

```bash
python RAG/chapter07-RAG/main.py
```

### 打开 RAG Notebook

`RAG/chapter07-RAG/` 当前包含以下 Notebook：

- `01-文档加载器的使用.ipynb`
- `02-文档拆分器的使用.ipynb`
- `03-文档嵌入模型的使用.ipynb`
- `04-向量数据库的使用.ipynb`
- `05-检索器的使用.ipynb`
- `06-综合案例：智能对话助手.ipynb`

### RAG 脚本说明

`RAG/chapter07-RAG/main.py` 当前演示的是基础向量检索流程，默认会：

1. 读取 `asset/load/09-ai1.txt`
2. 切分文档内容
3. 调用 embeddings 接口生成向量
4. 使用 `FAISS` 构建索引
5. 根据查询词执行检索并输出结果

如果你使用的是本地或代理的 OpenAI 兼容服务，请确保该服务支持 embeddings API。

## 依赖概览

当前 `pyproject.toml` 中声明的主要依赖包括：

- `langchain`
- `langchain-community`
- `markdown`
- `unstructured`
- `openai`
- `pandas`

AgenticRL 真实训练所需的重型依赖放在可选 extra 中：

```bash
uv sync --extra rl
```

RAG 相关代码还会用到 `langchain-openai`、`langchain-text-splitters` 与向量库依赖；如果本地环境缺失，请一并安装。

## 架构对比

| | ReAct | CodeAct | RAG |
|---|---|---|---|
| 核心思想 | 推理驱动工具调用 | 生成并执行代码 | 检索外部知识后再生成 |
| 行动方式 | 调用预定义工具 | 生成 Python 代码 | 查询向量数据库/检索器 |
| 灵活性 | 受限于已注册工具 | 可执行更开放的逻辑 | 依赖知识库质量与召回效果 |
| 安全性 | 工具边界明确 | 依赖沙箱隔离 | 重点在数据质量与接口配置 |
| 适用场景 | 结构化任务、API 调用 | 计算、数据处理、开放式问题 | 基于私有知识的问答与检索 |

## 当前状态

- 当前仓库以学习和实验为主
- `CodeAct/` 与 `react/` 目录聚焦 Agent 基础实现
- `RAG/chapter07-RAG/` 同时包含脚本示例和 Notebook 演示
- `RAG/chapter07-RAG/asset/` 下保留了部分向量数据库实验产物

## 开发

```bash
# 安装依赖
uv sync

# 运行示例
python CodeAct/main.py
python react/mian.py
python RAG/chapter07-RAG/main.py
```

如需补充更多依赖，请优先更新 `pyproject.toml` 并重新执行 `uv sync`。
