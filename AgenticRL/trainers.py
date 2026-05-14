"""Trainer wrappers around Hugging Face Transformers, TRL, and PEFT."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def _missing_rl_deps_message() -> str:
    return (
        "Training requires optional AgenticRL dependencies: torch, transformers, "
        "datasets, trl, peft, and accelerate. Install them with "
        "`uv sync --extra rl` or `pip install torch transformers datasets trl peft accelerate`."
    )


def _require_training_deps():
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import GRPOConfig, GRPOTrainer, SFTTrainer
    except ImportError as exc:
        raise ImportError(_missing_rl_deps_message()) from exc

    return {
        "torch": torch,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "TrainingArguments": TrainingArguments,
        "SFTTrainer": SFTTrainer,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
    }


@dataclass
class TrainingConfig:
    algorithm: str
    model_name: str = "Qwen/Qwen3-0.6B"
    output_dir: str = "./models/agentic_rl"
    num_epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 5e-5
    max_length: int = 1024
    max_new_tokens: int = 256
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    save_steps: int = 500
    logging_steps: int = 50
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    num_generations: int = 4
    temperature: float = 0.8
    kl_coef: float = 0.05
    clip_range: float = 0.2
    use_wandb: bool = False
    use_tensorboard: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        allowed = cls.__dataclass_fields__.keys()
        payload = {key: value for key, value in data.items() if key in allowed}
        if "algorithm" not in payload:
            payload["algorithm"] = data.get("algorithm", "sft")
        return cls(**payload)


class BaseTrainerWrapper:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.deps = _require_training_deps()

    def _load_model_and_tokenizer(self):
        torch = self.deps["torch"]
        AutoModelForCausalLM = self.deps["AutoModelForCausalLM"]
        AutoTokenizer = self.deps["AutoTokenizer"]

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        if self.config.use_lora:
            LoraConfig = self.deps["LoraConfig"]
            get_peft_model = self.deps["get_peft_model"]
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        return model, tokenizer

    def _report_to(self) -> list[str]:
        targets: list[str] = []
        if self.config.use_wandb:
            targets.append("wandb")
        if self.config.use_tensorboard:
            targets.append("tensorboard")
        return targets


def export_merged_model(
    base_model_name: str,
    lora_model_path: str,
    output_dir: str,
) -> dict[str, Any]:
    deps = _require_training_deps()
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    torch = deps["torch"]

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(_missing_rl_deps_message()) from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_path)

    return {
        "status": "success",
        "base_model": base_model_name,
        "lora_model_path": lora_model_path,
        "merged_model_path": str(output_path),
    }


class SFTTrainerWrapper(BaseTrainerWrapper):
    """Supervised fine-tuning wrapper."""

    def train(self, dataset: Any) -> dict[str, Any]:
        model, tokenizer = self._load_model_and_tokenizer()
        TrainingArguments = self.deps["TrainingArguments"]
        SFTTrainer = self.deps["SFTTrainer"]

        args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            report_to=self._report_to(),
        )

        try:
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=self.config.max_length,
                args=args,
            )
        except TypeError:
            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=dataset,
                args=args,
            )

        train_result = trainer.train()
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        final_loss = getattr(train_result, "training_loss", None)
        return {
            "status": "success",
            "algorithm": "sft",
            "output_dir": self.config.output_dir,
            "model_path": self.config.output_dir,
            "num_samples": len(dataset),
            "num_epochs": self.config.num_epochs,
            "final_loss": final_loss,
        }


class GRPOTrainerWrapper(BaseTrainerWrapper):
    """GRPO trainer wrapper."""

    def __init__(
        self,
        config: TrainingConfig,
        reward_function: Callable[[list[str]], list[float]],
    ):
        super().__init__(config)
        self.reward_function = reward_function

    def train(self, dataset: Any) -> dict[str, Any]:
        model, tokenizer = self._load_model_and_tokenizer()
        GRPOConfig = self.deps["GRPOConfig"]
        GRPOTrainer = self.deps["GRPOTrainer"]

        args = GRPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            report_to=self._report_to(),
            num_generations=self.config.num_generations,
            max_completion_length=self.config.max_new_tokens,
            temperature=self.config.temperature,
            beta=self.config.kl_coef,
            epsilon=self.config.clip_range,
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=self.reward_function,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        train_result = trainer.train()
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        average_reward = None
        metrics = getattr(train_result, "metrics", {}) or {}
        for key in ("train/reward", "reward", "rewards/mean"):
            if key in metrics:
                average_reward = metrics[key]
                break
        return {
            "status": "success",
            "algorithm": "grpo",
            "output_dir": self.config.output_dir,
            "model_path": self.config.output_dir,
            "num_samples": len(dataset),
            "num_epochs": self.config.num_epochs,
            "average_reward": average_reward,
        }
