import json
import re

import type
from ChatOpenAI import ChatOpenAI


class Agent:
    def __init__(self, tools: list[type.Tool]):
        self.tools = tools
        self.llm = ChatOpenAI(self.get_tool_calls())

    async def query(self, query: str):
        step = 1
        while True:
            response = await self.llm.chat(query)
            content = response.message.content or ""
            print(f"\n=== Step {step} ===")
            print("Reasoning/Response:", content)

            observation = ""
            if response.message.tool_calls:
                tool_calls = response.message.tool_calls
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    observation += await self._execute_tool(tool_name, tool_args)
            else:
                parsed_action = self._parse_action(content)
                if parsed_action is None:
                    return content

                tool_name, tool_args = parsed_action
                observation += await self._execute_tool(tool_name, tool_args)

            query = observation
            step += 1

    async def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        print(f"Action: {tool_name}({tool_args})")

        target_tool = next((tool for tool in self.tools if tool.name == tool_name), None)
        if target_tool is None:
            raise ValueError(f"Tool {tool_name} not found")

        params = target_tool.parameters.model_validate(tool_args)
        tool_response = await target_tool.execute(params)

        print(f"Observation: {tool_response}")
        return f"Observation from {tool_name}: {tool_response}\n"

    def _parse_action(self, content: str):
        match = re.search(
            r"Action:\s*call_[^:]+:(?P<tool>[A-Za-z_][A-Za-z0-9_]*)\{(?P<args>.*)\}",
            content,
            re.DOTALL,
        )
        if not match:
            return None

        tool_name = match.group("tool")
        raw_args = match.group("args").strip()
        parts = [part.strip() for part in raw_args.split(",") if part.strip()]
        tool_args = {}
        for part in parts:
            key, value = part.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value.replace(".", "", 1).isdigit():
                tool_args[key] = float(value) if "." in value else int(value)
            else:
                tool_args[key] = value

        return tool_name, tool_args

    def get_tool_calls(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters.model_json_schema(),
                },
            } for tool in self.tools
        ]
