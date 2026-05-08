"""Dataset helpers for Agentic RL math reasoning.

The functions here intentionally keep a small dependency surface. They accept
plain Python lists and, when the optional ``datasets`` package is installed,
also return Hugging Face ``Dataset`` objects for direct use with TRL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal

FormatType = Literal["sft", "rl"]


DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"


def _require_datasets():
    try:
        from datasets import Dataset, load_dataset
    except ImportError as exc:
        raise ImportError(
            "AgenticRL dataset loading requires the optional dependency "
            "`datasets`. Install it with `uv sync --extra rl` or "
            "`pip install datasets`."
        ) from exc
    return Dataset, load_dataset


def extract_gsm8k_answer(answer: str) -> str:
    """Extract the final answer from a GSM8K-style answer string."""

    if answer is None:
        return ""

    marker_match = re.search(r"####\s*([^\n]+)", str(answer))
    if marker_match:
        return _clean_answer(marker_match.group(1))

    final_match = re.search(
        r"(?:final\s*answer|answer)\s*[:：]\s*([^\n<|]+)",
        str(answer),
        flags=re.IGNORECASE,
    )
    if final_match:
        return _clean_answer(final_match.group(1))

    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(answer))
    return _clean_answer(numbers[-1]) if numbers else str(answer).strip()


def _clean_answer(value: Any) -> str:
    text = str(value).strip()
    text = text.replace(",", "")
    text = re.sub(r"<\|.*$", "", text).strip()
    return text.rstrip(".。")


def build_chat_prompt(question: str, model_name: str = DEFAULT_MODEL_NAME) -> str:
    """Build a Qwen-style chat prompt for a math question."""

    del model_name
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def build_sft_completion(answer: str) -> str:
    """Build a structured SFT completion from a raw GSM8K answer."""

    ground_truth = extract_gsm8k_answer(answer)
    reasoning = str(answer).split("####", maxsplit=1)[0].strip()
    if reasoning:
        completion = (
            "Let me solve this step by step.\n\n"
            f"{reasoning}\n\n"
            f"Final Answer: {ground_truth}<|im_end|>"
        )
    else:
        completion = f"Final Answer: {ground_truth}<|im_end|>"
    return completion


def _sample_to_record(
    sample: dict[str, Any],
    format_type: FormatType,
    model_name: str,
) -> dict[str, Any]:
    question = str(sample.get("question", "")).strip()
    answer = str(sample.get("answer", sample.get("full_answer", ""))).strip()
    prompt = build_chat_prompt(question, model_name=model_name)
    ground_truth = extract_gsm8k_answer(answer)

    if format_type == "sft":
        completion = build_sft_completion(answer)
        return {
            "question": question,
            "prompt": prompt,
            "completion": completion,
            "text": f"{prompt}{completion}",
            "ground_truth": ground_truth,
            "full_answer": answer,
        }

    if format_type == "rl":
        return {
            "question": question,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "full_answer": answer,
        }

    raise ValueError(f"Unsupported format_type: {format_type}")


def _iter_records(dataset: Any) -> Iterable[dict[str, Any]]:
    if isinstance(dataset, list):
        yield from dataset
        return
    for sample in dataset:
        yield dict(sample)


def _to_dataset_or_list(records: list[dict[str, Any]], prefer_hf: bool):
    if not prefer_hf:
        return records
    try:
        from datasets import Dataset
    except ImportError:
        return records
    return Dataset.from_list(records)


def format_math_dataset(
    dataset: Any,
    format_type: FormatType = "sft",
    model_name: str = DEFAULT_MODEL_NAME,
    prefer_hf: bool = True,
):
    """Convert math samples with ``question`` and ``answer`` into SFT/RL form."""

    records = [
        _sample_to_record(sample, format_type=format_type, model_name=model_name)
        for sample in _iter_records(dataset)
    ]
    return _to_dataset_or_list(records, prefer_hf=prefer_hf)


@dataclass
class GSM8KDataset:
    """Loader and formatter for GSM8K."""

    model_name: str = DEFAULT_MODEL_NAME
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"

    def load_raw(self, split: str = "train", max_samples: int | None = None):
        _, load_dataset = _require_datasets()
        dataset = load_dataset(self.dataset_name, self.dataset_config, split=split)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset

    def load(
        self,
        split: str = "train",
        format_type: FormatType = "sft",
        max_samples: int | None = None,
    ):
        raw_dataset = self.load_raw(split=split, max_samples=max_samples)
        return format_math_dataset(
            raw_dataset,
            format_type=format_type,
            model_name=self.model_name,
            prefer_hf=True,
        )


def create_sft_dataset(
    max_samples: int | None = None,
    split: str = "train",
    model_name: str = DEFAULT_MODEL_NAME,
):
    return GSM8KDataset(model_name=model_name).load(
        split=split,
        format_type="sft",
        max_samples=max_samples,
    )


def create_rl_dataset(
    max_samples: int | None = None,
    split: str = "train",
    model_name: str = DEFAULT_MODEL_NAME,
):
    return GSM8KDataset(model_name=model_name).load(
        split=split,
        format_type="rl",
        max_samples=max_samples,
    )


def dataset_summary(dataset: Any, format_type: str) -> dict[str, Any]:
    """Return a JSON-serializable summary for a formatted dataset."""

    size = len(dataset)
    sample = dict(dataset[0]) if size else {}
    return {
        "dataset_size": size,
        "format": format_type,
        "sample_keys": list(sample.keys()),
        "sample": sample,
    }
