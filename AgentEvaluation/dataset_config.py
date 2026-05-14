"""Dataset and benchmark registry for agent evaluation.

The registry captures every benchmark or dataset mentioned in the source notes.
Some entries are directly supported by the lightweight local evaluators in this
package; others are environment benchmarks that need their official harnesses.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration metadata for one dataset or benchmark."""

    key: str
    name: str
    capability: str
    source_type: str
    source: str
    loader: str | None
    evaluator: str | None
    local_path_hint: str
    splits: tuple[str, ...] = ()
    subsets: tuple[str, ...] = ()
    env_vars: tuple[str, ...] = ()
    gated: bool = False
    directly_supported: bool = False
    notes: str = ""


@dataclass(frozen=True)
class DatasetValidation:
    """Local readiness check for a configured dataset."""

    key: str
    is_configured: bool
    local_path: str | None
    missing_env_vars: tuple[str, ...]
    warnings: tuple[str, ...] = ()


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "bfcl": DatasetConfig(
        key="bfcl",
        name="BFCL (Berkeley Function Calling Leaderboard)",
        capability="工具调用能力",
        source_type="github",
        source="https://github.com/ShishirPatil/gorilla",
        loader="BFCLDataset",
        evaluator="BFCLEvaluator",
        local_path_hint="temp_gorilla/berkeley-function-call-leaderboard/bfcl_eval/data",
        subsets=(
            "simple_python",
            "simple_java",
            "simple_javascript",
            "multiple",
            "parallel",
            "parallel_multiple",
            "irrelevance",
        ),
        directly_supported=True,
        notes="本项目支持 BFCL 风格 JSON/JSONL 样本的本地评估；官方榜单评估需安装 bfcl-eval。",
    ),
    "toolbench": DatasetConfig(
        key="toolbench",
        name="ToolBench",
        capability="真实 API 工具调用",
        source_type="github",
        source="https://github.com/OpenBMB/ToolBench",
        loader=None,
        evaluator=None,
        local_path_hint="data/toolbench",
        notes="环境和 API 检索链路较重，建议通过官方 ToolBench harness 运行。",
    ),
    "api_bank": DatasetConfig(
        key="api_bank",
        name="API-Bank",
        capability="API 文档理解与调用",
        source_type="github",
        source="https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank",
        loader=None,
        evaluator=None,
        local_path_hint="data/api-bank",
        notes="文档中作为 API 调用基准提及；本地尚未封装官方评估器。",
    ),
    "gaia": DatasetConfig(
        key="gaia",
        name="GAIA",
        capability="通用 AI 助手能力",
        source_type="huggingface",
        source="gaia-benchmark/GAIA",
        loader="GAIADataset",
        evaluator="GAIAEvaluator",
        local_path_hint="data/gaia",
        splits=("validation", "test"),
        subsets=("level_1", "level_2", "level_3"),
        env_vars=("HF_TOKEN",),
        gated=True,
        directly_supported=True,
        notes="受限 HuggingFace 数据集；本项目支持 GAIA 风格 JSON/JSONL 样本的本地评估。",
    ),
    "agentbench": DatasetConfig(
        key="agentbench",
        name="AgentBench",
        capability="多环境通用 Agent 能力",
        source_type="github",
        source="https://github.com/THUDM/AgentBench",
        loader=None,
        evaluator=None,
        local_path_hint="data/agentbench",
        subsets=(
            "os",
            "db",
            "kg",
            "dcg",
            "ltp",
            "alfworld",
            "webshop",
            "mind2web",
        ),
        notes="包含交互环境和程序化评分，建议使用官方 AgentBench 运行器。",
    ),
    "webarena": DatasetConfig(
        key="webarena",
        name="WebArena",
        capability="真实网页环境任务完成",
        source_type="github",
        source="https://github.com/web-arena-x/webarena",
        loader=None,
        evaluator=None,
        local_path_hint="data/webarena",
        notes="需要部署配套网站环境和浏览器代理，不能仅靠静态数据文件完成验证。",
    ),
    "chateval": DatasetConfig(
        key="chateval",
        name="ChatEval",
        capability="多智能体对话质量评估",
        source_type="github",
        source="https://github.com/thunlp/ChatEval",
        loader=None,
        evaluator=None,
        local_path_hint="data/chateval",
        env_vars=("OPENAI_API_KEY",),
        notes="更接近评估框架而非单一静态数据集；通常依赖 LLM 评委。",
    ),
    "sotopia": DatasetConfig(
        key="sotopia",
        name="SOTOPIA",
        capability="社交智能与多智能体互动",
        source_type="github",
        source="https://github.com/sotopia-lab/sotopia",
        loader=None,
        evaluator=None,
        local_path_hint="data/sotopia",
        env_vars=("OPENAI_API_KEY",),
        subsets=("sotopia_all", "sotopia_hard"),
        notes="需要官方环境运行多轮社交模拟，并用 SOTOPIA-EVAL 评分。",
    ),
    "aime_1983_2025": DatasetConfig(
        key="aime_1983_2025",
        name="AIME 1983-2025",
        capability="数学题生成参考数据",
        source_type="huggingface",
        source="TianHongZXY/aime-1983-2025",
        loader="AIDataset",
        evaluator="LLMJudgeEvaluator",
        local_path_hint="data/aime-1983-2025",
        splits=("test",),
        directly_supported=True,
        notes="文档中用于 AIME 风格题目生成的参考样例库。",
    ),
    "aime25": DatasetConfig(
        key="aime25",
        name="AIME 2025",
        capability="数学题生成质量对比参考",
        source_type="huggingface",
        source="math-ai/aime25",
        loader="AIDataset",
        evaluator="WinRateEvaluator",
        local_path_hint="data/aime25",
        splits=("train", "test"),
        directly_supported=True,
        notes="文档中用于生成题目质量评估的真题参考集。",
    ),
}


def list_dataset_configs(
    capability: str | None = None,
    directly_supported: bool | None = None,
) -> list[DatasetConfig]:
    """List dataset configs, optionally filtering by capability or support status."""

    configs = list(DATASET_CONFIGS.values())
    if capability is not None:
        configs = [config for config in configs if capability in config.capability]
    if directly_supported is not None:
        configs = [config for config in configs if config.directly_supported == directly_supported]
    return configs


def get_dataset_config(key: str) -> DatasetConfig:
    """Return one dataset config by key."""

    try:
        return DATASET_CONFIGS[key]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_CONFIGS))
        raise KeyError(f"Unknown dataset config: {key}. Available: {available}") from exc


def validate_dataset_config(
    key: str,
    project_root: str | Path = ".",
    env: dict[str, str] | None = None,
) -> DatasetValidation:
    """Check whether local files and required environment variables are present."""

    config = get_dataset_config(key)
    root = Path(project_root)
    local_path = root / config.local_path_hint
    active_env = env if env is not None else os.environ
    missing_env_vars = tuple(name for name in config.env_vars if not active_env.get(name))
    warnings = []
    if not config.directly_supported:
        warnings.append("本项目当前只保存配置；请使用官方 harness 运行完整评估。")
    if config.gated and missing_env_vars:
        warnings.append("这是受限数据集，需要先申请访问并配置 token。")

    return DatasetValidation(
        key=key,
        is_configured=local_path.exists() and not missing_env_vars,
        local_path=str(local_path) if local_path.exists() else None,
        missing_env_vars=missing_env_vars,
        warnings=tuple(warnings),
    )


def validate_all_dataset_configs(project_root: str | Path = ".") -> dict[str, DatasetValidation]:
    """Validate every registered dataset against the local workspace."""

    return {
        key: validate_dataset_config(key, project_root=project_root)
        for key in DATASET_CONFIGS
    }


def agent_validation_plan() -> dict[str, list[str]]:
    """Group dataset keys by the agent capability they are useful for testing."""

    plan: dict[str, list[str]] = {}
    for config in DATASET_CONFIGS.values():
        plan.setdefault(config.capability, []).append(config.key)
    return plan


def export_dataset_configs(output_path: str | Path) -> Path:
    """Export the registry to JSON for scripts or notebooks."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        key: asdict(config)
        for key, config in DATASET_CONFIGS.items()
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
