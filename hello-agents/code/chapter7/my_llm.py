from pathlib import Path
import sys
from typing import Optional

from openai import OpenAI
from hello_agents import HelloAgentsLLM


def _find_code_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if candidate.name == "code" and candidate.parent.name == "hello-agents":
            return candidate
    raise ValueError("Unable to locate hello-agents/code root")


CODE_ROOT = _find_code_root(Path(__file__).resolve().parent)
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shared.env_config import get_llm_config


class MyLLM(HelloAgentsLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs,
    ):
        if provider == "modelscope":
            print("正在使用自定义的 ModelScope Provider")
            self.provider = "modelscope"

            llm_config = get_llm_config(code_root=CODE_ROOT)
            self.api_key = api_key or llm_config.api_key
            self.base_url = base_url or llm_config.base_url

            if not self.api_key:
                raise ValueError("Shared .env is missing API_KEY for the modelscope provider.")

            self.model = model or llm_config.model_id or "Qwen/Qwen2.5-VL-72B-Instruct"
            self.temperature = kwargs.get("temperature", 0.7)
            self.max_tokens = kwargs.get("max_tokens")
            self.timeout = kwargs.get("timeout", llm_config.timeout)
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
            return

        super().__init__(model=model, api_key=api_key, base_url=base_url, provider=provider, **kwargs)
