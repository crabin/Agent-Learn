import os

from openai import OpenAI
from dotenv import load_dotenv
import prompt

load_dotenv()


class ChatOpenAI:
    def __init__(self, tool_calls=None):
        self.model = os.getenv("OPENAI_MODEL")
        self.debug = os.getenv("DEBUG", "1") == "1"
        self.llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        self.messages = [{
            "role": "system",
            "content": prompt.SYSTEM_PROMPT
        }]
        self.tool_calls = tool_calls or []

    def _debug_print(self, title: str, content):
        if not self.debug:
            return
        print(f"\n=== {title} ===")
        print(content)

    async def chat(self, query: str):
        self.messages.append({
            "role": "user",
            "content": query
        })
        self._debug_print("User Query", query)
        self._debug_print("Request Messages", self.messages)

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tool_calls
        )
        message = response.choices[0].message
        self._debug_print("Raw Model Message", message)
        self._debug_print("Raw Model Content", message.content)

        self.messages.append(message)
        return response.choices[0]
