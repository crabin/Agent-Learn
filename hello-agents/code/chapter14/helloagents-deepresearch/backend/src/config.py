from pathlib import Path
import sys
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _find_code_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if candidate.name == "code" and candidate.parent.name == "hello-agents":
            return candidate
    raise ValueError("Unable to locate hello-agents/code root")


CODE_ROOT = _find_code_root(Path(__file__).resolve().parent)
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shared.env_config import get_env_value


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"
    ADVANCED = "advanced"


class Configuration(BaseModel):
    """Configuration options for the deep research assistant."""

    max_web_research_loops: int = Field(
        default=3,
        title="Research Depth",
        description="Number of research iterations to perform",
    )
    local_llm: str = Field(
        default="llama3.2",
        title="Local Model Name",
        description="Name of the locally hosted LLM (Ollama/LMStudio)",
    )
    llm_provider: str = Field(
        default="ollama",
        title="LLM Provider",
        description="Provider identifier (ollama, lmstudio, or custom)",
    )
    search_api: SearchAPI = Field(
        default=SearchAPI.DUCKDUCKGO,
        title="Search API",
        description="Web search API to use",
    )
    enable_notes: bool = Field(
        default=True,
        title="Enable Notes",
        description="Whether to store task progress in NoteTool",
    )
    notes_workspace: str = Field(
        default="./notes",
        title="Notes Workspace",
        description="Directory for NoteTool to persist task notes",
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        title="Ollama Base URL",
        description="Base URL for Ollama API (without /v1 suffix)",
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        title="LMStudio Base URL",
        description="Base URL for LMStudio OpenAI-compatible API",
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses",
    )
    use_tool_calling: bool = Field(
        default=False,
        title="Use Tool Calling",
        description="Use tool calling instead of JSON mode for structured output",
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        title="LLM API Key",
        description="Optional API key when using custom OpenAI-compatible services",
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        title="LLM Base URL",
        description="Optional base URL when using custom OpenAI-compatible services",
    )
    llm_model_id: Optional[str] = Field(
        default=None,
        title="LLM Model ID",
        description="Optional model identifier for custom OpenAI-compatible services",
    )

    @classmethod
    def from_env(cls, overrides: Optional[dict[str, Any]] = None) -> "Configuration":
        """Create a configuration object using shared env values and overrides."""

        raw_values: dict[str, Any] = {}
        shared_values = {
            "local_llm": get_env_value("LOCAL_LLM", default=None, code_root=CODE_ROOT),
            "llm_provider": get_env_value("LLM_PROVIDER", default=None, code_root=CODE_ROOT),
            "llm_api_key": get_env_value("API_KEY", default=None, code_root=CODE_ROOT),
            "llm_model_id": get_env_value("MODEL_ID", default=None, code_root=CODE_ROOT),
            "llm_base_url": get_env_value("BASE_URL", default=None, code_root=CODE_ROOT),
            "lmstudio_base_url": get_env_value(
                "LMSTUDIO_BASE_URL", default=None, code_root=CODE_ROOT
            ),
            "ollama_base_url": get_env_value(
                "OLLAMA_BASE_URL", default=None, code_root=CODE_ROOT
            ),
            "max_web_research_loops": get_env_value(
                "MAX_WEB_RESEARCH_LOOPS", default=None, code_root=CODE_ROOT
            ),
            "fetch_full_page": get_env_value(
                "FETCH_FULL_PAGE", default=None, code_root=CODE_ROOT
            ),
            "strip_thinking_tokens": get_env_value(
                "STRIP_THINKING_TOKENS", default=None, code_root=CODE_ROOT
            ),
            "use_tool_calling": get_env_value(
                "USE_TOOL_CALLING", default=None, code_root=CODE_ROOT
            ),
            "search_api": get_env_value("SEARCH_API", default=None, code_root=CODE_ROOT),
            "enable_notes": get_env_value("ENABLE_NOTES", default=None, code_root=CODE_ROOT),
            "notes_workspace": get_env_value(
                "NOTES_WORKSPACE", default=None, code_root=CODE_ROOT
            ),
        }

        for key, value in shared_values.items():
            if value is not None:
                raw_values.setdefault(key, value)

        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    raw_values[key] = value

        return cls(**raw_values)

    def sanitized_ollama_url(self) -> str:
        """Ensure Ollama base URL includes the /v1 suffix required by OpenAI clients."""

        base = self.ollama_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    def resolved_model(self) -> Optional[str]:
        """Best-effort resolution of the model identifier to use."""

        return self.llm_model_id or self.local_llm
