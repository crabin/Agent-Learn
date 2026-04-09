import os

from openai import OpenAI
from dotenv import load_dotenv
import prompt

load_dotenv()


class ChatOpenAI:
    def __init__(self, tool_calls=None):
        self.model = os.getenv("OPENAI_MODEL")
        self.llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        self.messages = [{
            "role": "system",
            "content": prompt.SYSTEM_PROMPT
        }]
        self.tool_calls = tool_calls or []

    async def chat(self, query: str):
        self.messages.append({
            "role": "user",
            "content": query
        })
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tool_calls
        )
        self.messages.append(response.choices[0].message)
        return response.choices[0]
