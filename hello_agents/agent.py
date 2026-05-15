"""Local SimpleAgent implementation used by evaluation examples."""

from __future__ import annotations

from typing import Any

from .llm import HelloAgentsLLM


class SimpleAgent:
    """A minimal synchronous agent with the ``run(prompt)`` protocol."""

    def __init__(
        self,
        name: str = "SimpleAgent",
        llm: HelloAgentsLLM | None = None,
        system_prompt: str = "",
    ):
        self.name = name
        self.llm = llm or HelloAgentsLLM()
        self.system_prompt = system_prompt
        self.tools: list[Any] = []

    def add_tool(self, tool: Any) -> None:
        self.tools.append(tool)

    def run(self, prompt: str) -> str:
        full_prompt = prompt
        if self.system_prompt:
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
        return self.llm.generate(full_prompt, tools=self.tools)
