"""Local LLM wrapper with OpenAI-compatible and deterministic fallback modes."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable
from typing import Any


class HelloAgentsLLM:
    """Small OpenAI-compatible LLM wrapper.

    If ``OPENAI_API_KEY`` is not configured, the wrapper falls back to a
    deterministic function-call generator so local evaluation examples can run
    without network access.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        mock_responses: dict[str, str] | list[str] | Callable[[str], str] | None = None,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "local-rule-llm")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.base_url = self._normalize_base_url(base_url or os.getenv("OPENAI_API_BASE"))
        self.mock_responses = mock_responses
        self._mock_iter: Iterable[str] | None = iter(mock_responses) if isinstance(mock_responses, list) else None

    def generate(self, prompt: str, tools: list[Any] | None = None) -> str:
        mocked = self._generate_mock(prompt)
        if mocked is not None:
            return mocked
        if self.api_key:
            return self._generate_openai(prompt, tools=tools)
        return self._generate_rule_based(prompt)

    def _generate_mock(self, prompt: str) -> str | None:
        if self.mock_responses is None:
            return None
        if callable(self.mock_responses):
            return self.mock_responses(prompt)
        if isinstance(self.mock_responses, dict):
            for needle, response in self.mock_responses.items():
                if needle in prompt:
                    return response
            return next(iter(self.mock_responses.values()), "")
        if self._mock_iter is not None:
            try:
                return next(self._mock_iter)
            except StopIteration:
                return ""
        return None

    def _generate_openai(self, prompt: str, tools: list[Any] | None = None) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required when OPENAI_API_KEY is set.") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            tools=self._format_tools(tools or []) or None,
        )
        message = response.choices[0].message
        if getattr(message, "tool_calls", None):
            calls = []
            for tool_call in message.tool_calls:
                calls.append(
                    {
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments or "{}"),
                    }
                )
            return json.dumps(calls, ensure_ascii=False)
        return message.content or ""

    def _generate_rule_based(self, prompt: str) -> str:
        functions = self._extract_functions(prompt)
        if functions:
            function = functions[0]
            name = function.get("name") or function.get("function", {}).get("name")
            parameters = function.get("parameters") or function.get("function", {}).get("parameters", {})
            arguments = self._infer_arguments(prompt, parameters)
            return json.dumps([{"name": name, "arguments": arguments}], ensure_ascii=False)

        if "FINAL ANSWER" in prompt:
            return "FINAL ANSWER: unknown"
        return "我已经收到问题，但本地规则 LLM 无法生成可靠答案。"

    def _extract_functions(self, prompt: str) -> list[dict[str, Any]]:
        marker = "可用函数:"
        if marker not in prompt:
            return []
        raw = prompt.split(marker, 1)[1].strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"(\[.*\])", raw, flags=re.DOTALL)
            if not match:
                return []
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                return []
        return parsed if isinstance(parsed, list) else []

    def _infer_arguments(self, prompt: str, parameters: dict[str, Any]) -> dict[str, Any]:
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        required = parameters.get("required", list(properties)) if isinstance(parameters, dict) else []
        arguments: dict[str, Any] = {}
        for name in required:
            lowered = name.lower()
            if lowered in {"location", "city"}:
                arguments[name] = self._first_city(prompt)
            elif lowered == "unit":
                arguments[name] = "celsius" if "celsius" in prompt.lower() else "fahrenheit"
            elif lowered in {"number", "n"}:
                arguments[name] = self._first_number(prompt)
            elif lowered in {"base", "height", "width", "length"}:
                arguments[name] = self._first_number(prompt)
            else:
                arguments[name] = self._default_value(properties.get(name, {}))
        return arguments

    def _first_city(self, prompt: str) -> str:
        for city in ("Beijing", "Shanghai", "Guangzhou", "Shenzhen", "California"):
            if city.lower() in prompt.lower():
                return city
        return "Beijing"

    def _first_number(self, prompt: str) -> int:
        match = re.search(r"-?\d+", prompt)
        return int(match.group(0)) if match else 1

    def _default_value(self, schema: dict[str, Any]) -> Any:
        schema_type = schema.get("type")
        if schema_type in {"integer", "number"}:
            return 1
        if schema_type == "boolean":
            return True
        if schema_type == "array":
            return []
        return ""

    def _format_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        formatted = []
        for tool in tools:
            if isinstance(tool, dict):
                formatted.append(tool)
                continue
            name = getattr(tool, "name", tool.__class__.__name__)
            description = getattr(tool, "description", "")
            parameters = getattr(tool, "parameters", {"type": "object", "properties": {}})
            if hasattr(parameters, "model_json_schema"):
                parameters = parameters.model_json_schema()
            formatted.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )
        return formatted

    def _normalize_base_url(self, base_url: str | None) -> str | None:
        if not base_url:
            return None
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        return normalized
