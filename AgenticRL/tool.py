"""Unified Agentic RL training tool."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from .datasets import (
    create_rl_dataset,
    create_sft_dataset,
    dataset_summary,
    format_math_dataset,
)
from .rewards import (
    answers_equal,
    create_reward_function,
    count_reasoning_steps,
    extract_answer,
)
from .trainers import GRPOTrainerWrapper, SFTTrainerWrapper, TrainingConfig


class RLTrainingTool:
    """A small unified interface for dataset loading, rewards, training, and eval."""

    def __init__(self):
        self.datasets: dict[str, Any] = {}
        self.reward_functions: dict[str, Callable[..., list[float]]] = {}

    def register_dataset(self, name: str, dataset: Any) -> None:
        self.datasets[name] = dataset

    def register_reward_function(
        self,
        name: str,
        reward_function: Callable[..., list[float]],
    ) -> None:
        self.reward_functions[name] = reward_function

    def run(self, params: dict[str, Any]) -> str:
        action = params.get("action")
        try:
            if action == "load_dataset":
                result = self._load_dataset(params)
            elif action == "create_reward":
                result = self._create_reward(params)
            elif action == "train":
                result = self._train(params)
            elif action == "evaluate":
                result = self._evaluate(params)
            else:
                raise ValueError(
                    "Unknown action. Use one of: load_dataset, create_reward, train, evaluate."
                )
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as exc:
            return json.dumps(
                {
                    "status": "error",
                    "action": action,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                ensure_ascii=False,
                indent=2,
            )

    def _load_dataset(self, params: dict[str, Any]) -> dict[str, Any]:
        format_type = params.get("format", "sft")
        max_samples = params.get("max_samples")
        split = params.get("split", "train")
        model_name = params.get("model_name", "Qwen/Qwen3-0.6B")
        custom_dataset = params.get("custom_dataset")

        if custom_dataset is not None:
            dataset = format_math_dataset(
                custom_dataset,
                format_type=format_type,
                model_name=model_name,
            )
        elif format_type == "sft":
            dataset = create_sft_dataset(
                max_samples=max_samples,
                split=split,
                model_name=model_name,
            )
        elif format_type == "rl":
            dataset = create_rl_dataset(
                max_samples=max_samples,
                split=split,
                model_name=model_name,
            )
        else:
            raise ValueError(f"Unsupported dataset format: {format_type}")

        name = params.get("name")
        if name:
            self.register_dataset(name, dataset)

        summary = dataset_summary(dataset, format_type=format_type)
        summary["status"] = "success"
        return summary

    def _create_reward(self, params: dict[str, Any]) -> dict[str, Any]:
        reward_type = params.get("reward_type", "accuracy")
        reward_config = {
            key: value
            for key, value in params.items()
            if key not in {"action", "reward_type", "name"}
        }
        reward_fn = create_reward_function(reward_type, **reward_config)

        name = params.get("name")
        if name:
            self.register_reward_function(name, reward_fn)

        result = {
            "status": "success",
            "reward_type": reward_type,
            "description": reward_fn.__class__.__name__,
        }
        result.update(reward_config)
        return result

    def _resolve_dataset(self, params: dict[str, Any], algorithm: str):
        if params.get("custom_dataset") is not None:
            return params["custom_dataset"]

        dataset_name = params.get("dataset")
        if dataset_name:
            if dataset_name not in self.datasets:
                raise ValueError(f"Dataset is not registered: {dataset_name}")
            return self.datasets[dataset_name]

        max_samples = params.get("max_samples")
        split = params.get("split", "train")
        model_name = params.get("model_name", "Qwen/Qwen3-0.6B")
        if algorithm == "sft":
            return create_sft_dataset(max_samples=max_samples, split=split, model_name=model_name)
        return create_rl_dataset(max_samples=max_samples, split=split, model_name=model_name)

    def _resolve_reward(self, params: dict[str, Any]):
        custom_reward = params.get("custom_reward")
        if custom_reward is not None:
            return custom_reward

        reward_name = params.get("reward")
        if reward_name:
            if reward_name not in self.reward_functions:
                raise ValueError(f"Reward function is not registered: {reward_name}")
            return self.reward_functions[reward_name]

        dataset_name = params.get("dataset")
        if dataset_name and dataset_name in self.reward_functions:
            return self.reward_functions[dataset_name]

        reward_type = params.get("reward_type", "accuracy")
        reward_config = params.get("reward_config", {})
        return create_reward_function(reward_type, **reward_config)

    def _train(self, params: dict[str, Any]) -> dict[str, Any]:
        algorithm = params.get("algorithm", "sft").lower()
        dataset = self._resolve_dataset(params, algorithm=algorithm)
        config = TrainingConfig.from_dict({**params, "algorithm": algorithm})

        if algorithm == "sft":
            return SFTTrainerWrapper(config).train(dataset)
        if algorithm == "grpo":
            reward_fn = self._resolve_reward(params)
            return GRPOTrainerWrapper(config, reward_function=reward_fn).train(dataset)
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    def _evaluate(self, params: dict[str, Any]) -> dict[str, Any]:
        if "predictions" in params:
            return self._evaluate_predictions(params)
        return self._evaluate_model(params)

    def _evaluate_predictions(self, params: dict[str, Any]) -> dict[str, Any]:
        predictions = list(params["predictions"])
        ground_truths = list(params.get("ground_truth", params.get("ground_truths", [])))
        if len(predictions) != len(ground_truths):
            raise ValueError("`predictions` and `ground_truth` must have the same length.")

        reward_fn = self._resolve_reward(params)
        rewards = reward_fn(predictions, ground_truth=ground_truths)
        details = _build_eval_details(predictions, ground_truths, rewards)
        return _summarize_eval(details, params=params)

    def _evaluate_model(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Model evaluation requires torch, transformers, and peft. "
                "For a dependency-light check, pass `predictions` and `ground_truth`."
            ) from exc

        model_path = params.get("model_path")
        if not model_path:
            raise ValueError("`model_path` is required for model evaluation.")

        base_model = params.get("base_model", params.get("model_name", model_path))
        use_lora = bool(params.get("use_lora", False))
        max_samples = params.get("max_samples", 100)
        split = params.get("split", "test")
        dataset = params.get("custom_dataset") or create_rl_dataset(
            max_samples=max_samples,
            split=split,
            model_name=params.get("model_name", "Qwen/Qwen3-0.6B"),
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if use_lora:
            model = PeftModel.from_pretrained(model, model_path)
        elif model_path != base_model:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )

        reward_fn = self._resolve_reward(params)
        predictions: list[str] = []
        ground_truths: list[str] = []
        durations: list[float] = []
        for sample in dataset:
            prompt = sample["prompt"]
            start = time.time()
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(params.get("max_new_tokens", 256)),
                temperature=float(params.get("temperature", 0.7)),
                do_sample=bool(params.get("do_sample", True)),
            )
            durations.append(time.time() - start)
            text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            predictions.append(text[len(prompt) :] if text.startswith(prompt) else text)
            ground_truths.append(sample["ground_truth"])

        rewards = reward_fn(predictions, ground_truth=ground_truths)
        details = _build_eval_details(predictions, ground_truths, rewards, durations)
        return _summarize_eval(details, params=params)


def _build_eval_details(
    predictions: list[str],
    ground_truths: list[Any],
    rewards: list[float],
    durations: list[float] | None = None,
) -> list[dict[str, Any]]:
    details = []
    for index, (prediction, truth, reward) in enumerate(
        zip(predictions, ground_truths, rewards)
    ):
        pred_answer = extract_answer(prediction)
        correct = answers_equal(pred_answer, truth)
        item = {
            "index": index,
            "prediction": prediction,
            "predicted_answer": pred_answer,
            "ground_truth": truth,
            "reward": float(reward),
            "correct": correct,
            "length": len(prediction),
            "steps": count_reasoning_steps(prediction),
            "format_correct": "Final Answer" in prediction or "####" in prediction,
        }
        if durations is not None:
            item["duration_seconds"] = durations[index]
        details.append(item)
    return details


def _summarize_eval(details: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
    total = len(details)
    accuracy = sum(1 for item in details if item["correct"]) / total if total else 0.0
    average_reward = (
        sum(float(item["reward"]) for item in details) / total if total else 0.0
    )
    average_length = sum(item["length"] for item in details) / total if total else 0.0
    average_steps = sum(item["steps"] for item in details) / total if total else 0.0
    format_correctness = (
        sum(1 for item in details if item["format_correct"]) / total if total else 0.0
    )

    result = {
        "status": "success",
        "accuracy": accuracy,
        "average_reward": average_reward,
        "num_samples": total,
        "average_length": average_length,
        "average_steps": average_steps,
        "format_correctness": format_correctness,
    }

    if params.get("return_details"):
        result["details"] = details
        result["errors"] = [item for item in details if not item["correct"]]
    return result
