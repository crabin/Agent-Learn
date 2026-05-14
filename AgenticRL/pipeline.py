"""End-to-end Agentic RL training pipeline example."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .tool import RLTrainingTool


class AgenticRLPipeline:
    """Run data preparation, SFT, GRPO, evaluation, and result persistence."""

    def __init__(self, config_path: str = "config.json"):
        self.rl_tool = RLTrainingTool()
        self.config_path = Path(config_path)
        self.config = self.load_config(self.config_path)
        self.results: dict[str, Any] = {}

    def load_config(self, config_path: Path) -> dict[str, Any]:
        with config_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _run_json(self, params: dict[str, Any]) -> dict[str, Any]:
        result = json.loads(self.rl_tool.run(params))
        if result.get("status") == "error":
            raise RuntimeError(result["error"])
        return result

    def stage1_prepare_data(self) -> dict[str, Any]:
        self.log("阶段1: 数据准备")
        data = self.config["data"]
        result = self._run_json(
            {
                "action": "load_dataset",
                "format": "sft",
                "max_samples": data.get("max_samples"),
                "model_name": self.config["model"]["base_model"],
            }
        )
        self.results["data"] = result
        self.log(f"数据集加载完成: {result['dataset_size']} 个样本")
        return result

    def stage2_sft_training(self) -> str:
        self.log("阶段2: SFT训练")
        sft = self.config["sft"]
        result = self._run_json(
            {
                "action": "train",
                "algorithm": "sft",
                "model_name": self.config["model"]["base_model"],
                "output_dir": sft["output_dir"],
                "max_samples": self.config["data"].get("max_samples"),
                "num_epochs": sft.get("num_epochs", 1),
                "batch_size": sft.get("batch_size", 2),
                "learning_rate": sft.get("learning_rate", 5e-5),
                "use_lora": sft.get("use_lora", True),
                **self.config.get("monitoring", {}),
            }
        )
        self.results["sft_training"] = result
        self.log(f"SFT训练完成: {result['output_dir']}")
        return result["output_dir"]

    def stage3_sft_evaluation(self, model_path: str) -> dict[str, Any]:
        self.log("阶段3: SFT评估")
        result = self._run_json(
            {
                "action": "evaluate",
                "model_path": model_path,
                "max_samples": self.config["eval"].get("max_samples", 100),
                "use_lora": True,
            }
        )
        self.results["sft_evaluation"] = result
        self.log(f"SFT准确率: {result['accuracy']}")
        return result

    def stage4_grpo_training(self, sft_model_path: str) -> str:
        self.log("阶段4: GRPO训练")
        grpo = self.config["grpo"]
        result = self._run_json(
            {
                "action": "train",
                "algorithm": "grpo",
                "model_name": sft_model_path,
                "output_dir": grpo["output_dir"],
                "max_samples": self.config["data"].get("max_samples"),
                "num_epochs": grpo.get("num_epochs", 1),
                "batch_size": grpo.get("batch_size", 2),
                "learning_rate": grpo.get("learning_rate", 1e-5),
                "num_generations": grpo.get("num_generations", 4),
                "kl_coef": grpo.get("kl_coef", 0.05),
                "use_lora": grpo.get("use_lora", True),
                "reward_type": grpo.get("reward_type", "accuracy"),
                "reward_config": grpo.get("reward_config", {}),
                **self.config.get("monitoring", {}),
            }
        )
        self.results["grpo_training"] = result
        self.log(f"GRPO训练完成: {result['output_dir']}")
        return result["output_dir"]

    def stage5_grpo_evaluation(self, model_path: str) -> dict[str, Any]:
        self.log("阶段5: GRPO评估")
        result = self._run_json(
            {
                "action": "evaluate",
                "model_path": model_path,
                "max_samples": self.config["eval"].get("max_samples", 100),
                "use_lora": True,
            }
        )
        self.results["grpo_evaluation"] = result
        self.log(f"GRPO准确率: {result['accuracy']}")
        return result

    def stage6_export_model(self, model_path: str) -> dict[str, Any] | None:
        export_config = self.config.get("export", {})
        if not export_config.get("enabled", False):
            return None

        self.log("阶段6: 导出生产模型")
        result = self._run_json(
            {
                "action": "export_model",
                "base_model": self.config["model"]["base_model"],
                "model_path": model_path,
                "output_dir": export_config.get("output_dir", "./models/merged_model"),
            }
        )
        self.results["model_export"] = result
        self.log(f"生产模型已导出: {result['merged_model_path']}")
        return result

    def stage7_save_results(self) -> None:
        path = Path(self.config.get("results_path", "training_results.json"))
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.results, file, ensure_ascii=False, indent=2)
        self.log(f"结果已保存到: {path}")

    def run(self) -> dict[str, Any]:
        self.stage1_prepare_data()
        sft_model_path = self.stage2_sft_training()
        self.stage3_sft_evaluation(sft_model_path)
        grpo_model_path = self.stage4_grpo_training(sft_model_path)
        self.stage5_grpo_evaluation(grpo_model_path)
        self.stage6_export_model(grpo_model_path)
        self.stage7_save_results()
        self.log("训练流程完成")
        return self.results


def default_config() -> dict[str, Any]:
    return {
        "model": {"base_model": "Qwen/Qwen3-0.6B"},
        "data": {"max_samples": 100},
        "sft": {
            "output_dir": "./models/sft_model",
            "num_epochs": 1,
            "batch_size": 2,
            "learning_rate": 5e-5,
            "use_lora": True,
        },
        "grpo": {
            "output_dir": "./models/grpo_model",
            "num_epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-5,
            "num_generations": 4,
            "kl_coef": 0.05,
            "use_lora": True,
            "reward_type": "accuracy",
        },
        "eval": {"max_samples": 50},
        "export": {
            "enabled": True,
            "output_dir": "./models/merged_model",
        },
        "monitoring": {"use_wandb": False, "use_tensorboard": True},
        "results_path": "training_results.json",
    }


if __name__ == "__main__":
    config_path = Path("config.json")
    if not config_path.exists():
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(default_config(), file, ensure_ascii=False, indent=2)
    AgenticRLPipeline(str(config_path)).run()
