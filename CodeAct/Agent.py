import re
import os

from SandBox import Sandbox
from ChatOpenAI import ChatOpenAI

class Agent:
    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self.llm = ChatOpenAI()
        self.debug = os.getenv("DEBUG", "1") == "1"

    def _debug_print(self, title: str, content):
        if not self.debug:
            return
        print(f"\n=== {title} ===")
        print(content)

    async def query(self, user_query: str):
        response = await self.llm.chat(user_query)
        content = response.message.content
        self._debug_print("Agent Received Content", content)

        if "```" not in content:
            self._debug_print("Agent Decision", "No code block found, returning plain text.")
            return content

        # 提取第一个 Markdown fenced code block，语言标签可选。
        match = re.search(r"```[^\n]*\n(.*?)```", content, re.DOTALL)
        if not match:
            self._debug_print("Agent Decision", "Code fence marker found but no valid fenced block matched.")
            return content

        code = match.group(1).strip()
        self._debug_print("Extracted Code", code)
        logs = self.sandbox.run(code)
        self._debug_print("Sandbox Output", logs)
        return logs
        
