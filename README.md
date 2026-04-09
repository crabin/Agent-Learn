# AgentLearn

AI Agent 学习项目，通过从零实现两种经典 Agent 架构来理解其核心原理。

## 项目结构

```
AgentLearn/
├── react/       # ReAct (Reasoning + Acting) 实现
└── CodeAct/     # CodeAct (Code as Action) 实现
```

## ReAct

基于 [ReAct](https://arxiv.org/abs/2210.03629) 论文的实现。Agent 通过 **推理 → 调用工具 → 观察结果 → 继续推理** 的循环来解决问题。

- 支持 OpenAI 原生 Function Calling 和文本解析两种工具调用方式
- 使用 Pydantic 定义工具参数 schema
- 内置 Calculator 工具作为示例

**核心流程：** 用户提问 → LLM 推理 → 选择工具并传参 → 获取观察结果 → 重复直到得出最终答案

## CodeAct

基于 [CodeAct](https://arxiv.org/abs/2402.01030) 论文的实现。Agent 将 **代码本身作为行动**，LLM 生成 Python 代码并在沙箱中执行。

- LLM 生成 Python 代码块作为解决方案
- 内置 Sandbox 安全执行环境，捕获标准输出
- 沙箱提供基础数学函数（sum, sub, mul, div）

**核心流程：** 用户提问 → LLM 生成 Python 代码 → Sandbox 执行 → 返回结果

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/crabin/Agent-Learn.git
cd Agent-Learn

# 2. 安装依赖
pip install openai pydantic python-dotenv

# 3. 配置环境变量
cp CodeAct/.env.example CodeAct/.env
cp react/.env.example react/.env
# 编辑 .env 文件填入你的 API Key

# 4. 运行示例
python CodeAct/main.py
python react/mian.py
```

## 两种架构对比

| | ReAct | CodeAct |
|---|---|---|
| 行动方式 | 调用预定义工具 | 生成并执行代码 |
| 灵活性 | 受限于已注册的工具 | 可执行任意 Python 代码 |
| 安全性 | 工具边界明确 | 依赖沙箱隔离 |
| 适用场景 | 结构化任务、API 调用 | 计算、数据处理、开放式问题 |
