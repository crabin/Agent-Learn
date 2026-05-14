"""FastAPI inference service for exported AgenticRL models."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def _missing_api_deps_message() -> str:
    return (
        "API serving requires optional AgenticRL dependencies: fastapi, uvicorn, "
        "torch, and transformers. Install them with "
        "`uv sync --extra rl`."
    )


try:
    import torch
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    raise ImportError(_missing_api_deps_message()) from exc


DEFAULT_MODEL_DIR = Path("./models/merged_model")

app = FastAPI(title="AgenticRL Inference API", version="0.1.0")


class Question(BaseModel):
    text: str = Field(..., description="User question or prompt")
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    do_sample: bool = True


class Answer(BaseModel):
    text: str
    model_path: str


@lru_cache(maxsize=1)
def _load_model_bundle() -> tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    model_dir = DEFAULT_MODEL_DIR
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory does not exist: {model_dir}. "
            "Run the training pipeline with export enabled first."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    return model, tokenizer, str(model_dir)


@app.get("/health")
def health() -> dict[str, str]:
    _, _, model_path = _load_model_bundle()
    return {"status": "ok", "model_path": model_path}


@app.post("/generate", response_model=Answer)
def generate(question: Question) -> Answer:
    model, tokenizer, model_path = _load_model_bundle()
    prompt = f"<|im_start|>user\n{question.text}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=question.max_tokens,
        temperature=question.temperature,
        do_sample=question.do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response_text = full_text[len(prompt) :] if full_text.startswith(prompt) else full_text
    return Answer(text=response_text, model_path=model_path)
