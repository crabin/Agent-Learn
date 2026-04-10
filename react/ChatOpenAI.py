import os

from openai import NotFoundError, OpenAI
from dotenv import load_dotenv
import prompt

load_dotenv()


class ChatOpenAI:
    def __init__(self, tool_calls=None):
        self.model = os.getenv("OPENAI_MODEL")
        self.base_url = self._normalize_base_url(os.getenv("OPENAI_API_BASE"))
        self.llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=self.base_url
        )
        self.messages = [{
            "role": "system",
            "content": prompt.SYSTEM_PROMPT
        }]
        self.tool_calls = tool_calls or []

    def _normalize_base_url(self, base_url: str | None) -> str | None:
        if not base_url:
            return None

        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        return normalized

    async def chat(self, query: str):
        self.messages.append({
            "role": "user",
            "content": query
        })
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tool_calls
            )
        except NotFoundError as exc:
            raise RuntimeError(
                "OpenAI-compatible endpoint returned 404. "
                f"Resolved base URL: {self.base_url}. "
                "This usually means the proxy expects a /v1 base path or the requested model does not exist."
            ) from exc
        self.messages.append(response.choices[0].message)
        return response.choices[0]
