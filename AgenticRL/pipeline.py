"""End-to-end Agentic RL training pipeline example."""

from __future__ import annotations

import copy
import itertools
import json
import math
import random
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
                "base_model": self.config["model"]["base_model"],
                "max_samples": self.config["eval"].get("max_samples", 100),
                "use_lora": True,
            }
        )
        self.results["sft_evaluation"] = result
        self.log(f"SFT准确率: {result['accuracy']}")
        return result

    def stage4_grpo_training(self, sft_model_path: str) -> str:
        self.log("阶段4: GRPO训练")
        optimization = self.config.get("optimization", {})
        if optimization.get("enabled", False):
            return self.stage4_grpo_optimization(sft_model_path)

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

    def stage4_grpo_optimization(self, sft_model_path: str) -> str:
        optimization = self.config.get("optimization", {})
        target = optimization.get("target", "grpo")
        if target != "grpo":
            raise ValueError("当前仅支持对 `grpo` 阶段做参数搜索。")

        method = optimization.get("method", "random").lower()
        metric = optimization.get("metric", "accuracy")
        maximize = bool(optimization.get("maximize", True))
        self.log(f"阶段4: GRPO参数搜索 ({method})")

        if method == "grid":
            return self._run_grid_search(
                sft_model_path=sft_model_path,
                metric=metric,
                maximize=maximize,
            )
        if method == "random":
            return self._run_random_search(
                sft_model_path=sft_model_path,
                metric=metric,
                maximize=maximize,
            )
        if method == "bayesian":
            return self._run_bayesian_optimization(
                sft_model_path=sft_model_path,
                metric=metric,
                maximize=maximize,
            )
        raise ValueError(f"Unsupported optimization method: {method}")

    def _grpo_train_and_evaluate(
        self,
        sft_model_path: str,
        trial_params: dict[str, Any],
        trial_index: int,
        metric: str,
    ) -> dict[str, Any]:
        grpo = copy.deepcopy(self.config["grpo"])
        grpo.update(trial_params)
        base_output_dir = Path(grpo["output_dir"])
        trial_output_dir = base_output_dir.parent / f"{base_output_dir.name}_trial_{trial_index:03d}"

        train_result = self._run_json(
            {
                "action": "train",
                "algorithm": "grpo",
                "model_name": sft_model_path,
                "output_dir": str(trial_output_dir),
                "max_samples": self.config["data"].get("max_samples"),
                "num_epochs": grpo.get("num_epochs", 1),
                "batch_size": grpo.get("batch_size", 2),
                "learning_rate": grpo.get("learning_rate", 1e-5),
                "num_generations": grpo.get("num_generations", 4),
                "kl_coef": grpo.get("kl_coef", 0.05),
                "temperature": grpo.get("temperature", 0.8),
                "use_lora": grpo.get("use_lora", True),
                "reward_type": grpo.get("reward_type", "accuracy"),
                "reward_config": grpo.get("reward_config", {}),
                **self.config.get("monitoring", {}),
            }
        )
        eval_result = self._run_json(
            {
                "action": "evaluate",
                "model_path": train_result["output_dir"],
                "base_model": sft_model_path,
                "max_samples": self.config["eval"].get("max_samples", 100),
                "use_lora": True,
                "reward_type": grpo.get("reward_type", "accuracy"),
                "reward_config": grpo.get("reward_config", {}),
            }
        )
        if metric not in eval_result:
            raise ValueError(f"评估结果中不存在指标 `{metric}`。")

        return {
            "trial_index": trial_index,
            "params": trial_params,
            "train_result": train_result,
            "eval_result": eval_result,
            "score": eval_result[metric],
            "model_path": train_result["output_dir"],
        }

    def _run_grid_search(
        self,
        sft_model_path: str,
        metric: str,
        maximize: bool,
    ) -> str:
        optimization = self.config.get("optimization", {})
        search_space = optimization.get("search_space", {})
        if not search_space:
            raise ValueError("Grid search requires a non-empty `optimization.search_space`.")

        combinations = list(self._iter_grid_combinations(search_space))
        max_trials = optimization.get("max_trials")
        if max_trials is not None:
            combinations = combinations[: int(max_trials)]

        if not combinations:
            raise ValueError("Grid search generated zero parameter combinations.")

        trials = []
        best_trial = None
        for index, params in enumerate(combinations, start=1):
            self.log(f"开始 grid trial {index}/{len(combinations)}: {params}")
            trial = self._grpo_train_and_evaluate(sft_model_path, params, index, metric)
            trials.append(trial)
            best_trial = self._pick_better_trial(best_trial, trial, maximize=maximize)
            self.log(f"trial {index} 完成, {metric}={trial['score']}")

        return self._finalize_optimization_result(
            method="grid",
            metric=metric,
            maximize=maximize,
            trials=trials,
            best_trial=best_trial,
        )

    def _run_random_search(
        self,
        sft_model_path: str,
        metric: str,
        maximize: bool,
    ) -> str:
        optimization = self.config.get("optimization", {})
        search_space = optimization.get("search_space", {})
        max_trials = int(optimization.get("max_trials", 5))
        rng = random.Random(optimization.get("random_seed", 42))

        if not search_space:
            raise ValueError("Random search requires a non-empty `optimization.search_space`.")

        trials = []
        best_trial = None
        for index in range(1, max_trials + 1):
            params = {
                name: self._sample_random_value(name, spec, rng)
                for name, spec in search_space.items()
            }
            self.log(f"开始 random trial {index}/{max_trials}: {params}")
            trial = self._grpo_train_and_evaluate(sft_model_path, params, index, metric)
            trials.append(trial)
            best_trial = self._pick_better_trial(best_trial, trial, maximize=maximize)
            self.log(f"trial {index} 完成, {metric}={trial['score']}")

        return self._finalize_optimization_result(
            method="random",
            metric=metric,
            maximize=maximize,
            trials=trials,
            best_trial=best_trial,
        )

    def _run_bayesian_optimization(
        self,
        sft_model_path: str,
        metric: str,
        maximize: bool,
    ) -> str:
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "贝叶斯优化需要 `optuna`，请先安装：`uv add --optional rl optuna` "
                "或 `pip install optuna`。"
            ) from exc

        optimization = self.config.get("optimization", {})
        search_space = optimization.get("search_space", {})
        max_trials = int(optimization.get("max_trials", 10))
        if not search_space:
            raise ValueError("Bayesian optimization requires a non-empty `optimization.search_space`.")

        direction = "maximize" if maximize else "minimize"
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=optimization.get("random_seed", 42)),
        )
        trials: list[dict[str, Any]] = []

        def objective(trial: Any) -> float:
            trial_params = {
                name: self._sample_optuna_value(trial, name, spec)
                for name, spec in search_space.items()
            }
            current = self._grpo_train_and_evaluate(
                sft_model_path=sft_model_path,
                trial_params=trial_params,
                trial_index=len(trials) + 1,
                metric=metric,
            )
            trials.append(current)
            return float(current["score"])

        study.optimize(objective, n_trials=max_trials)
        best_trial = max(trials, key=lambda item: item["score"]) if maximize else min(
            trials,
            key=lambda item: item["score"],
        )

        self.results["optimization_study"] = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "direction": direction,
        }
        return self._finalize_optimization_result(
            method="bayesian",
            metric=metric,
            maximize=maximize,
            trials=trials,
            best_trial=best_trial,
        )

    def _finalize_optimization_result(
        self,
        method: str,
        metric: str,
        maximize: bool,
        trials: list[dict[str, Any]],
        best_trial: dict[str, Any] | None,
    ) -> str:
        if best_trial is None:
            raise ValueError("参数搜索未产生任何 trial。")

        self.results["optimization"] = {
            "method": method,
            "metric": metric,
            "maximize": maximize,
            "num_trials": len(trials),
            "best_trial": best_trial,
            "trials": trials,
        }
        self.results["grpo_training"] = best_trial["train_result"]
        self.results["grpo_optimization_evaluation"] = best_trial["eval_result"]
        self.log(
            "参数搜索完成: "
            f"best trial={best_trial['trial_index']}, "
            f"{metric}={best_trial['score']}, params={best_trial['params']}"
        )
        return best_trial["model_path"]

    def _pick_better_trial(
        self,
        current_best: dict[str, Any] | None,
        candidate: dict[str, Any],
        maximize: bool,
    ) -> dict[str, Any]:
        if current_best is None:
            return candidate
        if maximize and candidate["score"] > current_best["score"]:
            return candidate
        if not maximize and candidate["score"] < current_best["score"]:
            return candidate
        return current_best

    def _iter_grid_combinations(
        self,
        search_space: dict[str, Any],
    ):
        names = list(search_space.keys())
        values = [self._extract_grid_values(name, spec) for name, spec in search_space.items()]
        for combination in itertools.product(*values):
            yield dict(zip(names, combination))

    def _extract_grid_values(self, name: str, spec: Any) -> list[Any]:
        if isinstance(spec, list):
            values = spec
        elif isinstance(spec, dict) and "values" in spec:
            values = spec["values"]
        else:
            raise ValueError(
                f"Grid search parameter `{name}` must be a list or a dict with `values`."
            )
        if not values:
            raise ValueError(f"Grid search parameter `{name}` has no candidate values.")
        return list(values)

    def _sample_random_value(self, name: str, spec: Any, rng: random.Random) -> Any:
        if isinstance(spec, list):
            return rng.choice(spec)
        if not isinstance(spec, dict):
            raise ValueError(f"Random search parameter `{name}` 配置无效。")

        param_type = spec.get("type", "categorical")
        if param_type == "categorical":
            values = spec.get("values", [])
            if not values:
                raise ValueError(f"Random search parameter `{name}` has no candidate values.")
            return rng.choice(values)
        if param_type == "int":
            return rng.randint(int(spec["low"]), int(spec["high"]))
        if param_type == "float":
            low = float(spec["low"])
            high = float(spec["high"])
            if spec.get("log", False):
                return 10 ** rng.uniform(_safe_log10(low), _safe_log10(high))
            return rng.uniform(low, high)
        raise ValueError(f"Unsupported random search parameter type: {param_type}")

    def _sample_optuna_value(self, trial: Any, name: str, spec: Any) -> Any:
        if isinstance(spec, list):
            return trial.suggest_categorical(name, spec)
        if not isinstance(spec, dict):
            raise ValueError(f"Bayesian optimization parameter `{name}` 配置无效。")

        param_type = spec.get("type", "categorical")
        if param_type == "categorical":
            return trial.suggest_categorical(name, spec["values"])
        if param_type == "int":
            return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        if param_type == "float":
            return trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        raise ValueError(f"Unsupported Bayesian optimization parameter type: {param_type}")

    def stage5_grpo_evaluation(
        self,
        model_path: str,
        base_model: str,
    ) -> dict[str, Any]:
        self.log("阶段5: GRPO评估")
        result = self._run_json(
            {
                "action": "evaluate",
                "model_path": model_path,
                "base_model": base_model,
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
        self.stage5_grpo_evaluation(grpo_model_path, base_model=sft_model_path)
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
        "optimization": {
            "enabled": False,
            "target": "grpo",
            "method": "random",
            "metric": "accuracy",
            "maximize": True,
            "max_trials": 4,
            "random_seed": 42,
            "search_space": {
                "learning_rate": {
                    "type": "float",
                    "low": 5e-6,
                    "high": 5e-5,
                    "log": True,
                },
                "batch_size": {"type": "categorical", "values": [1, 2, 4]},
                "num_generations": {"type": "categorical", "values": [2, 4, 8]},
                "kl_coef": {"type": "float", "low": 0.01, "high": 0.1},
                "temperature": {"type": "float", "low": 0.6, "high": 1.0},
            },
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


def _safe_log10(value: float) -> float:
    if value <= 0:
        raise ValueError("对数采样要求 `low` 和 `high` 都大于 0。")
    return math.log10(value)
