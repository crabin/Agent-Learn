# Hello Agents Shared Env Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `hello-agents/code` use one shared configuration-loading approach centered on `hello-agents/code/.env`, with `API_KEY` / `BASE_URL` / `MODEL_ID` read from that file and `TAVILY_API_KEY` / `GITHUB_TOKEN` / `WANDB_API_KEY` read only from the process environment.

**Architecture:** Add a focused shared module at `hello-agents/code/shared/env_config.py` that locates `hello-agents/code`, loads `hello-agents/code/.env` exactly once, and exposes small helpers for common LLM config, generic `.env` keys, and required system-only keys. Migrate direct-run chapter scripts to import this module via a consistent `Path(__file__)` + `sys.path` bootstrap snippet, while standalone app configs in chapters 13–15 keep their own config classes but delegate raw environment access to the shared helper.

**Tech Stack:** Python 3.10+, `python-dotenv`, `pytest`, `pydantic-settings` (existing chapter 13 app), JSON notebook format for `chapter1/FirstAgentTest.ipynb`.

---

## File Structure

**Create:**
- `hello-agents/code/shared/__init__.py` — package marker and narrow public exports.
- `hello-agents/code/shared/env_config.py` — shared root discovery, `.env` loading, `.env` readers, system-env readers, LLM config dataclass.
- `hello-agents/code/.env.example` — single shared example file for all chapters under `hello-agents/code`.
- `tests/hello_agents_code/shared/test_env_config.py` — unit tests for root discovery, `.env` loading, required/optional env readers, system-only env behavior, and migration smoke checks.

**Modify:**
- `hello-agents/code/chapter1/FirstAgentTest.py`
- `hello-agents/code/chapter1/FirstAgentTest.ipynb`
- `hello-agents/code/chapter4/llm_client.py`
- `hello-agents/code/chapter4/Plan_and_solve.py`
- `hello-agents/code/chapter4/tools.py`
- `hello-agents/code/chapter6/Langgraph/Dialogue_System.py`
- `hello-agents/code/chapter6/AutoGenDemo/autogen_software_team.py`
- `hello-agents/code/chapter6/CAMEL/DigitalBookWriting.py`
- `hello-agents/code/chapter6/AgentScopeDemo/main_cn.py`
- `hello-agents/code/chapter7/my_llm.py`
- `hello-agents/code/chapter7/my_main.py`
- `hello-agents/code/chapter7/my_advanced_search.py`
- `hello-agents/code/chapter7/test_advanced_search.py`
- `hello-agents/code/chapter7/test_plan_solve_agent.py`
- `hello-agents/code/chapter7/test_react_agent.py`
- `hello-agents/code/chapter7/test_reflection_agent.py`
- `hello-agents/code/chapter7/test_simple_agent.py`
- `hello-agents/code/chapter7/test_my_calculator.py`
- `hello-agents/code/chapter8/01_MemoryTool_Basic_Operations.py`
- `hello-agents/code/chapter8/02_MemoryTool_Architecture.py`
- `hello-agents/code/chapter8/03_WorkingMemory_Implementation.py`
- `hello-agents/code/chapter8/04_RAGTool_MarkItDown_Pipeline.py`
- `hello-agents/code/chapter8/05_RAGTool_Advanced_Search.py`
- `hello-agents/code/chapter8/06_Memory_Consolidation_Demo.py`
- `hello-agents/code/chapter8/07_RAGTool_Intelligent_QA.py`
- `hello-agents/code/chapter8/08_Agent_Tool_Integration.py`
- `hello-agents/code/chapter8/09_Memory_Types_Deep_Dive.py`
- `hello-agents/code/chapter8/10_RAG_Pipeline_Complete.py`
- `hello-agents/code/chapter8/11_Q&A_Assistant.py`
- `hello-agents/code/chapter9/01_context_builder_basic.py`
- `hello-agents/code/chapter9/02_context_builder_with_agent.py`
- `hello-agents/code/chapter9/04_note_tool_integration.py`
- `hello-agents/code/chapter9/06_three_day_workflow.py`
- `hello-agents/code/chapter10/06_MultiAgentDocumentAssist.py`
- `hello-agents/code/chapter10/10_A2ATool_Simple.py`
- `hello-agents/code/chapter10/10_CustomerService.py`
- `hello-agents/code/chapter10/12_ANPTaskDistribution.py`
- `hello-agents/code/chapter10/14_weather_agent.py`
- `hello-agents/code/chapter10/weather-mcp-server/server.py`
- `hello-agents/code/chapter11/08_distributed_training.py`
- `hello-agents/code/chapter12/05_gaia_quick_start.py`
- `hello-agents/code/chapter13/helloagents-trip-planner/backend/app/config.py`
- `hello-agents/code/chapter14/helloagents-deepresearch/backend/src/config.py`
- `hello-agents/code/chapter15/Helloagents-AI-Town/backend/config.py`
- chapter-local `.env.example` files that currently duplicate root config guidance:
  - `hello-agents/code/chapter7/.env.example`
  - `hello-agents/code/chapter8/.env.example`
  - `hello-agents/code/chapter9/.env.example`
  - `hello-agents/code/chapter10/.env.example`
  - `hello-agents/code/chapter11/.env.example`
  - `hello-agents/code/chapter12/.env.example`
  - `hello-agents/code/chapter13/helloagents-trip-planner/backend/.env.example`
  - `hello-agents/code/chapter15/Helloagents-AI-Town/backend/.env.example`

**Keep unchanged:**
- `hello-agents/code/chapter11/config.json` — remains JSON config; only env-backed values around it are standardized.
- `hello-agents/code/chapter12/04_run_bfcl_evaluation.py` — `PYTHONUTF8` process tweak is not part of external credential loading.

---

### Task 1: Build the shared env module first

**Files:**
- Create: `hello-agents/code/shared/__init__.py`
- Create: `hello-agents/code/shared/env_config.py`
- Test: `tests/hello_agents_code/shared/test_env_config.py`

- [ ] **Step 1: Write the failing tests for shared env behavior**

```python
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest


def load_env_config_module(code_root: Path | None = None) -> ModuleType:
    root = code_root or Path("hello-agents/code")
    spec = spec_from_file_location("shared.env_config", root / "shared" / "env_config.py")
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_find_code_root_returns_hello_agents_code_dir(tmp_path: Path) -> None:
    code_root = tmp_path / "hello-agents" / "code"
    nested = code_root / "chapter7"
    nested.mkdir(parents=True)

    env_config = load_env_config_module(code_root)

    assert env_config.find_code_root(nested) == code_root


def test_get_llm_config_reads_shared_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code_root = tmp_path / "hello-agents" / "code"
    code_root.mkdir(parents=True)
    (code_root / ".env").write_text(
        "API_KEY=test-key\nBASE_URL=https://example.com/v1\nMODEL_ID=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_ID", raising=False)

    env_config = load_env_config_module(code_root)
    llm = env_config.get_llm_config(code_root=code_root)

    assert llm.api_key == "test-key"
    assert llm.base_url == "https://example.com/v1"
    assert llm.model_id == "test-model"


def test_get_system_env_rejects_missing_required_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    env_config = load_env_config_module()

    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        env_config.get_system_env("TAVILY_API_KEY")
```

- [ ] **Step 2: Run the tests and verify they fail because the module does not exist yet**

Run: `pytest tests/hello_agents_code/shared/test_env_config.py -q`
Expected: FAIL with import or file-not-found errors for `shared/env_config.py`

- [ ] **Step 3: Create `hello-agents/code/shared/__init__.py` with the public exports**

```python
from .env_config import LLMConfig, get_env_value, get_llm_config, get_system_env, load_shared_dotenv

__all__ = [
    "LLMConfig",
    "get_env_value",
    "get_llm_config",
    "get_system_env",
    "load_shared_dotenv",
]
```

- [ ] **Step 5: Create `hello-agents/code/shared/env_config.py` with the minimal implementation**

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


SYSTEM_ONLY_KEYS = {"TAVILY_API_KEY", "GITHUB_TOKEN", "WANDB_API_KEY"}


@dataclass(frozen=True)
class LLMConfig:
    model_id: str
    api_key: str
    base_url: str
    timeout: int


def find_code_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    candidates = [current] + list(current.parents)
    for candidate in candidates:
        if candidate.name == "code" and candidate.parent.name == "hello-agents":
            return candidate
    raise ValueError("Unable to locate hello-agents/code root")


def load_shared_dotenv(code_root: Path | None = None) -> Path:
    root = code_root or find_code_root()
    env_path = root / ".env"
    load_dotenv(env_path, override=False)
    return env_path


def get_env_value(name: str, *, default: str | None = None, required: bool = False, code_root: Path | None = None) -> str | None:
    if name in SYSTEM_ONLY_KEYS:
        raise ValueError(f"{name} must be read with get_system_env()")
    load_shared_dotenv(code_root=code_root)
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"Missing required shared .env value: {name}")
    return value


def get_system_env(name: str) -> str:
    if name not in SYSTEM_ONLY_KEYS:
        raise ValueError(f"{name} is not a system-only key")
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing required system environment variable: {name}")
    return value


def get_llm_config(*, code_root: Path | None = None) -> LLMConfig:
    return LLMConfig(
        model_id=get_env_value("MODEL_ID", required=True, code_root=code_root) or "",
        api_key=get_env_value("API_KEY", required=True, code_root=code_root) or "",
        base_url=get_env_value("BASE_URL", required=True, code_root=code_root) or "",
        timeout=int(get_env_value("LLM_TIMEOUT", default="60", code_root=code_root) or "60"),
    )
```

- [ ] **Step 6: Run the focused tests again**

Run: `pytest tests/hello_agents_code/shared/test_env_config.py -q`
Expected: PASS

- [ ] **Step 7: Commit the shared env module**

```bash
git add hello-agents/code/shared/__init__.py hello-agents/code/shared/env_config.py tests/hello_agents_code/shared/test_env_config.py
git commit -m "feat: add shared hello-agents env loader"
```

### Task 2: Add the single shared `.env` example and root-import bootstrap pattern

**Files:**
- Create: `hello-agents/code/.env.example`
- Modify: `hello-agents/code/chapter7/.env.example`
- Modify: `hello-agents/code/chapter8/.env.example`
- Modify: `hello-agents/code/chapter9/.env.example`
- Modify: `hello-agents/code/chapter10/.env.example`
- Modify: `hello-agents/code/chapter11/.env.example`
- Modify: `hello-agents/code/chapter12/.env.example`
- Modify: `hello-agents/code/chapter13/helloagents-trip-planner/backend/.env.example`
- Modify: `hello-agents/code/chapter15/Helloagents-AI-Town/backend/.env.example`

- [ ] **Step 1: Write the failing test that documents the root `.env.example` contract**

```python
from pathlib import Path


def test_root_env_example_lists_shared_and_system_only_keys() -> None:
    content = Path("hello-agents/code/.env.example").read_text(encoding="utf-8")

    assert "API_KEY=" in content
    assert "BASE_URL=" in content
    assert "MODEL_ID=" in content
    assert "TAVILY_API_KEY" not in content
    assert "GITHUB_TOKEN" not in content
    assert "WANDB_API_KEY" not in content
```

- [ ] **Step 2: Run the focused test to verify it fails before the file exists**

Run: `pytest tests/hello_agents_code/shared/test_env_config.py -q`
Expected: FAIL because `hello-agents/code/.env.example` is missing

- [ ] **Step 3: Create the shared root `.env.example`**

```dotenv
# Shared Hello Agents runtime config
API_KEY=your-api-key
BASE_URL=https://your-openai-compatible-endpoint/v1
MODEL_ID=your-model-id
LLM_TIMEOUT=60

# Optional shared config for specific chapters/apps
AMAP_API_KEY=
UNSPLASH_ACCESS_KEY=
UNSPLASH_SECRET_KEY=
OPENWEATHER_API_KEY=
LOCAL_LLM=
LLM_PROVIDER=
OLLAMA_BASE_URL=http://localhost:11434
LMSTUDIO_BASE_URL=http://localhost:1234/v1
HF_TOKEN=
HF_ENDPOINT=
```

- [ ] **Step 4: Reduce chapter-local `.env.example` files to pointers instead of duplicate env definitions**

```dotenv
# This chapter now uses the shared config file at hello-agents/code/.env
# Copy hello-agents/code/.env.example to hello-agents/code/.env and fill in values there.
# System-only secrets are NOT read from .env:
# - TAVILY_API_KEY
# - GITHUB_TOKEN
# - WANDB_API_KEY
```

- [ ] **Step 5: Run a grep-based verification sweep**

Run: `rg -n "TAVILY_API_KEY=|GITHUB_TOKEN=|WANDB_API_KEY=" hello-agents/code/.env.example hello-agents/code/chapter{7,8,9,10,11,12}/.env.example hello-agents/code/chapter13/helloagents-trip-planner/backend/.env.example hello-agents/code/chapter15/Helloagents-AI-Town/backend/.env.example`
Expected: no matches

- [ ] **Step 6: Commit the shared `.env.example` cleanup**

```bash
git add hello-agents/code/.env.example hello-agents/code/chapter7/.env.example hello-agents/code/chapter8/.env.example hello-agents/code/chapter9/.env.example hello-agents/code/chapter10/.env.example hello-agents/code/chapter11/.env.example hello-agents/code/chapter12/.env.example hello-agents/code/chapter13/helloagents-trip-planner/backend/.env.example hello-agents/code/chapter15/Helloagents-AI-Town/backend/.env.example
git commit -m "docs: unify hello-agents env examples"
```

### Task 3: Migrate the direct LLM example scripts in chapters 1, 4, 6, and 7

**Files:**
- Modify: `hello-agents/code/chapter1/FirstAgentTest.py`
- Modify: `hello-agents/code/chapter1/FirstAgentTest.ipynb`
- Modify: `hello-agents/code/chapter4/llm_client.py`
- Modify: `hello-agents/code/chapter4/Plan_and_solve.py`
- Modify: `hello-agents/code/chapter4/tools.py`
- Modify: `hello-agents/code/chapter6/Langgraph/Dialogue_System.py`
- Modify: `hello-agents/code/chapter6/AutoGenDemo/autogen_software_team.py`
- Modify: `hello-agents/code/chapter6/CAMEL/DigitalBookWriting.py`
- Modify: `hello-agents/code/chapter6/AgentScopeDemo/main_cn.py`
- Modify: `hello-agents/code/chapter7/my_llm.py`
- Modify: `hello-agents/code/chapter7/my_main.py`
- Modify: `hello-agents/code/chapter7/my_advanced_search.py`
- Modify: `hello-agents/code/chapter7/test_advanced_search.py`
- Modify: `hello-agents/code/chapter7/test_plan_solve_agent.py`
- Modify: `hello-agents/code/chapter7/test_react_agent.py`
- Modify: `hello-agents/code/chapter7/test_reflection_agent.py`
- Modify: `hello-agents/code/chapter7/test_simple_agent.py`
- Modify: `hello-agents/code/chapter7/test_my_calculator.py`

- [ ] **Step 1: Write a failing smoke test for the direct script bootstrap snippet**

```python
from pathlib import Path


def test_first_agent_script_imports_shared_env_config() -> None:
    content = Path("hello-agents/code/chapter1/FirstAgentTest.py").read_text(encoding="utf-8")

    assert "from shared.env_config import get_llm_config, get_system_env" in content
    assert 'API_KEY = "YOUR_API_KEY"' not in content
    assert 'os.environ[\'TAVILY_API_KEY\'] = "YOUR_TAVILY_API_KEY"' not in content
```

- [ ] **Step 2: Run the smoke test to capture the existing hardcoded config failure**

Run: `pytest tests/hello_agents_code/shared/test_env_config.py -q`
Expected: FAIL because `FirstAgentTest.py` still contains hardcoded placeholders

- [ ] **Step 3: Add the consistent bootstrap snippet to direct-run scripts**

```python
from pathlib import Path
import sys


def _find_code_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if candidate.name == "code" and candidate.parent.name == "hello-agents":
            return candidate
    raise ValueError("Unable to locate hello-agents/code root")


CODE_ROOT = _find_code_root(Path(__file__).resolve().parent)
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shared.env_config import get_llm_config, get_system_env
```

- [ ] **Step 4: Replace direct LLM env access with the shared helper in `chapter4/llm_client.py`**

```python
llm_config = get_llm_config(code_root=CODE_ROOT)
self.model = model or llm_config.model_id
api_key = apiKey or llm_config.api_key
base_url = baseUrl or llm_config.base_url
timeout = timeout or llm_config.timeout
```

- [ ] **Step 5: Remove hardcoded placeholders from `chapter1/FirstAgentTest.py` and switch Tavily to system env**

```python
llm_config = get_llm_config(code_root=CODE_ROOT)
llm = OpenAICompatibleClient(
    model=llm_config.model_id,
    api_key=llm_config.api_key,
    base_url=llm_config.base_url,
)

api_key = get_system_env("TAVILY_API_KEY")
```

- [ ] **Step 6: Update `chapter7/my_llm.py` to stop reading `MODELSCOPE_API_KEY` directly**

```python
llm_config = get_llm_config(code_root=CODE_ROOT)
self.api_key = api_key or llm_config.api_key
self.base_url = base_url or llm_config.base_url
self.model = model or llm_config.model_id
```

- [ ] **Step 7: Update `chapter7/my_advanced_search.py` and related tests to use `get_system_env("TAVILY_API_KEY")`**

```python
from shared.env_config import get_system_env


def tavily_search(query: str) -> str:
    client = TavilyClient(api_key=get_system_env("TAVILY_API_KEY"))
    response = client.search(query=query, search_depth="basic", include_answer=True)
    if response.get("answer"):
        return response["answer"]
    results = response.get("results", [])
    return "\n".join(item.get("content", "") for item in results if item.get("content"))
```

- [ ] **Step 8: Update the notebook source cell in `chapter1/FirstAgentTest.ipynb` to match the script behavior**

```python
from pathlib import Path
import sys

CODE_ROOT = next(
    candidate
    for candidate in [Path.cwd(), *Path.cwd().parents]
    if candidate.name == "code" and candidate.parent.name == "hello-agents"
)
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shared.env_config import get_llm_config, get_system_env
llm_config = get_llm_config(code_root=CODE_ROOT)
```

- [ ] **Step 9: Run focused validation for the first migration wave**

Run: `python -m py_compile hello-agents/code/chapter1/FirstAgentTest.py hello-agents/code/chapter4/llm_client.py hello-agents/code/chapter6/Langgraph/Dialogue_System.py hello-agents/code/chapter6/AutoGenDemo/autogen_software_team.py hello-agents/code/chapter6/CAMEL/DigitalBookWriting.py hello-agents/code/chapter7/my_llm.py hello-agents/code/chapter7/my_main.py hello-agents/code/chapter7/my_advanced_search.py`
Expected: no output

- [ ] **Step 10: Commit the first migration wave**

```bash
git add hello-agents/code/chapter1/FirstAgentTest.py hello-agents/code/chapter1/FirstAgentTest.ipynb hello-agents/code/chapter4/llm_client.py hello-agents/code/chapter4/Plan_and_solve.py hello-agents/code/chapter4/tools.py hello-agents/code/chapter6/Langgraph/Dialogue_System.py hello-agents/code/chapter6/AutoGenDemo/autogen_software_team.py hello-agents/code/chapter6/CAMEL/DigitalBookWriting.py hello-agents/code/chapter6/AgentScopeDemo/main_cn.py hello-agents/code/chapter7/my_llm.py hello-agents/code/chapter7/my_main.py hello-agents/code/chapter7/my_advanced_search.py hello-agents/code/chapter7/test_advanced_search.py hello-agents/code/chapter7/test_plan_solve_agent.py hello-agents/code/chapter7/test_react_agent.py hello-agents/code/chapter7/test_reflection_agent.py hello-agents/code/chapter7/test_simple_agent.py hello-agents/code/chapter7/test_my_calculator.py
git commit -m "refactor: unify env loading in early hello-agents chapters"
```

### Task 4: Migrate chapters 8–10 and the remaining script-style examples

**Files:**
- Modify: `hello-agents/code/chapter8/01_MemoryTool_Basic_Operations.py`
- Modify: `hello-agents/code/chapter8/02_MemoryTool_Architecture.py`
- Modify: `hello-agents/code/chapter8/03_WorkingMemory_Implementation.py`
- Modify: `hello-agents/code/chapter8/04_RAGTool_MarkItDown_Pipeline.py`
- Modify: `hello-agents/code/chapter8/05_RAGTool_Advanced_Search.py`
- Modify: `hello-agents/code/chapter8/06_Memory_Consolidation_Demo.py`
- Modify: `hello-agents/code/chapter8/07_RAGTool_Intelligent_QA.py`
- Modify: `hello-agents/code/chapter8/08_Agent_Tool_Integration.py`
- Modify: `hello-agents/code/chapter8/09_Memory_Types_Deep_Dive.py`
- Modify: `hello-agents/code/chapter8/10_RAG_Pipeline_Complete.py`
- Modify: `hello-agents/code/chapter8/11_Q&A_Assistant.py`
- Modify: `hello-agents/code/chapter9/01_context_builder_basic.py`
- Modify: `hello-agents/code/chapter9/02_context_builder_with_agent.py`
- Modify: `hello-agents/code/chapter9/04_note_tool_integration.py`
- Modify: `hello-agents/code/chapter9/06_three_day_workflow.py`
- Modify: `hello-agents/code/chapter10/06_MultiAgentDocumentAssist.py`
- Modify: `hello-agents/code/chapter10/10_A2ATool_Simple.py`
- Modify: `hello-agents/code/chapter10/10_CustomerService.py`
- Modify: `hello-agents/code/chapter10/12_ANPTaskDistribution.py`
- Modify: `hello-agents/code/chapter10/14_weather_agent.py`
- Modify: `hello-agents/code/chapter10/weather-mcp-server/server.py`
- Modify: `hello-agents/code/chapter12/05_gaia_quick_start.py`

- [ ] **Step 1: Write the failing sweep test for stale `load_dotenv()` calls and nonlocal env paths**

```python
from pathlib import Path


MIGRATED_FILES = [
    "hello-agents/code/chapter8/01_MemoryTool_Basic_Operations.py",
    "hello-agents/code/chapter9/01_context_builder_basic.py",
    "hello-agents/code/chapter10/06_MultiAgentDocumentAssist.py",
    "hello-agents/code/chapter10/14_weather_agent.py",
]


def test_migrated_wave_two_files_stop_calling_load_dotenv_directly() -> None:
    for file_name in MIGRATED_FILES:
        content = Path(file_name).read_text(encoding="utf-8")
        assert "load_dotenv(" not in content
        assert "HelloAgents/.env" not in content
```

- [ ] **Step 2: Run the failing sweep before editing**

Run: `pytest tests/hello_agents_code/shared/test_env_config.py -q`
Expected: FAIL because these files still contain `load_dotenv()` or nonlocal env paths

- [ ] **Step 3: Apply the bootstrap snippet and replace direct `.env` loading in chapters 8–10**

```python
CODE_ROOT = _find_code_root(Path(__file__).resolve().parent)
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shared.env_config import get_llm_config, load_shared_dotenv

load_shared_dotenv(code_root=CODE_ROOT)
llm_config = get_llm_config(code_root=CODE_ROOT)
```

- [ ] **Step 4: Replace the bad custom path in `chapter10/06_MultiAgentDocumentAssist.py`**

```python
load_shared_dotenv(code_root=CODE_ROOT)
github_searcher = SimpleAgent(name="GitHub搜索专家", llm=HelloAgentsLLM())
```

- [ ] **Step 5: Keep non-credential runtime values local, but move true external config into the shared module contract**

```python
hf_endpoint = get_env_value("HF_ENDPOINT", default="https://hf-mirror.com", code_root=CODE_ROOT)
if hf_endpoint:
    os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
```

- [ ] **Step 6: Validate the second migration wave**

Run: `python -m py_compile hello-agents/code/chapter8/*.py hello-agents/code/chapter9/01_context_builder_basic.py hello-agents/code/chapter9/02_context_builder_with_agent.py hello-agents/code/chapter9/04_note_tool_integration.py hello-agents/code/chapter9/06_three_day_workflow.py hello-agents/code/chapter10/06_MultiAgentDocumentAssist.py hello-agents/code/chapter10/10_A2ATool_Simple.py hello-agents/code/chapter10/10_CustomerService.py hello-agents/code/chapter10/12_ANPTaskDistribution.py hello-agents/code/chapter10/14_weather_agent.py hello-agents/code/chapter10/weather-mcp-server/server.py hello-agents/code/chapter12/05_gaia_quick_start.py`
Expected: no output

- [ ] **Step 7: Commit the second migration wave**

```bash
git add hello-agents/code/chapter8/01_MemoryTool_Basic_Operations.py hello-agents/code/chapter8/02_MemoryTool_Architecture.py hello-agents/code/chapter8/03_WorkingMemory_Implementation.py hello-agents/code/chapter8/04_RAGTool_MarkItDown_Pipeline.py hello-agents/code/chapter8/05_RAGTool_Advanced_Search.py hello-agents/code/chapter8/06_Memory_Consolidation_Demo.py hello-agents/code/chapter8/07_RAGTool_Intelligent_QA.py hello-agents/code/chapter8/08_Agent_Tool_Integration.py hello-agents/code/chapter8/09_Memory_Types_Deep_Dive.py hello-agents/code/chapter8/10_RAG_Pipeline_Complete.py hello-agents/code/chapter8/11_Q&A_Assistant.py hello-agents/code/chapter9/01_context_builder_basic.py hello-agents/code/chapter9/02_context_builder_with_agent.py hello-agents/code/chapter9/04_note_tool_integration.py hello-agents/code/chapter9/06_three_day_workflow.py hello-agents/code/chapter10/06_MultiAgentDocumentAssist.py hello-agents/code/chapter10/10_A2ATool_Simple.py hello-agents/code/chapter10/10_CustomerService.py hello-agents/code/chapter10/12_ANPTaskDistribution.py hello-agents/code/chapter10/14_weather_agent.py hello-agents/code/chapter10/weather-mcp-server/server.py hello-agents/code/chapter12/05_gaia_quick_start.py
git commit -m "refactor: unify env loading in hello-agents tools chapters"
```

### Task 5: Adapt the app-style configs in chapters 11, 13, 14, and 15 without breaking their local abstractions

**Files:**
- Modify: `hello-agents/code/chapter11/08_distributed_training.py`
- Modify: `hello-agents/code/chapter13/helloagents-trip-planner/backend/app/config.py`
- Modify: `hello-agents/code/chapter14/helloagents-deepresearch/backend/src/config.py`
- Modify: `hello-agents/code/chapter15/Helloagents-AI-Town/backend/config.py`

- [ ] **Step 1: Write the failing test that these config files stop reading LLM secrets directly from env**

```python
from pathlib import Path


def test_app_config_files_delegate_llm_values_to_shared_env_module() -> None:
    config_files = [
        "hello-agents/code/chapter13/helloagents-trip-planner/backend/app/config.py",
        "hello-agents/code/chapter14/helloagents-deepresearch/backend/src/config.py",
        "hello-agents/code/chapter15/Helloagents-AI-Town/backend/config.py",
    ]
    for file_name in config_files:
        content = Path(file_name).read_text(encoding="utf-8")
        assert "from shared.env_config import" in content
        assert 'os.getenv("LLM_API_KEY")' not in content
        assert 'os.getenv("LLM_MODEL_ID")' not in content
        assert 'os.getenv("LLM_BASE_URL")' not in content
```

- [ ] **Step 2: Run the test and confirm the current config modules still read env directly**

Run: `pytest tests/hello_agents_code/shared/test_env_config.py -q`
Expected: FAIL because the app config files still use `os.getenv(...)`

- [ ] **Step 3: Keep chapter 13 `Settings` but source the LLM defaults from `get_llm_config()`**

```python
LLM = get_llm_config(code_root=CODE_ROOT)

class Settings(BaseSettings):
    openai_api_key: str = LLM.api_key
    openai_base_url: str = LLM.base_url
    openai_model: str = LLM.model_id
```

- [ ] **Step 4: Keep chapter 14 flexibility for local providers, but fall back to shared `.env` for custom-provider credentials**

```python
llm = get_llm_config(code_root=CODE_ROOT)
raw_values.setdefault("llm_api_key", llm.api_key)
raw_values.setdefault("llm_base_url", llm.base_url)
raw_values.setdefault("llm_model_id", llm.model_id)
```

- [ ] **Step 5: Replace chapter 15 class attributes with shared helper values and preserve validation output**

```python
LLM = get_llm_config(code_root=CODE_ROOT)

class Settings:
    LLM_MODEL_ID: str = LLM.model_id
    LLM_API_KEY: Optional[str] = LLM.api_key
    LLM_BASE_URL: str = LLM.base_url
```

- [ ] **Step 6: Move system-only secrets out of `.env` examples and into runtime validation paths where needed**

```python
try:
    tavily_api_key = get_system_env("TAVILY_API_KEY")
except ValueError:
    tavily_api_key = None
```

- [ ] **Step 7: Validate the app config wave**

Run: `python -m py_compile hello-agents/code/chapter11/08_distributed_training.py hello-agents/code/chapter13/helloagents-trip-planner/backend/app/config.py hello-agents/code/chapter14/helloagents-deepresearch/backend/src/config.py hello-agents/code/chapter15/Helloagents-AI-Town/backend/config.py`
Expected: no output

- [ ] **Step 8: Commit the app config migration wave**

```bash
git add hello-agents/code/chapter11/08_distributed_training.py hello-agents/code/chapter13/helloagents-trip-planner/backend/app/config.py hello-agents/code/chapter14/helloagents-deepresearch/backend/src/config.py hello-agents/code/chapter15/Helloagents-AI-Town/backend/config.py
git commit -m "refactor: route hello-agents app configs through shared env"
```

### Task 6: Run the final verification sweep and remove stale config patterns

**Files:**
- Modify: every file changed in Tasks 1–5 if cleanup is needed after verification.
- Test: `tests/hello_agents_code/shared/test_env_config.py`

- [ ] **Step 1: Run the shared env unit tests**

Run: `pytest tests/hello_agents_code/shared/test_env_config.py -q`
Expected: PASS

- [ ] **Step 2: Run a repository-wide stale-pattern grep**

Run: `rg -n "load_dotenv\(|HelloAgents/\.env|YOUR_API_KEY|YOUR_BASE_URL|YOUR_MODEL_ID|YOUR_TAVILY_API_KEY|os\.getenv\(\"LLM_API_KEY\"|os\.getenv\(\"LLM_MODEL_ID\"|os\.getenv\(\"LLM_BASE_URL\"" hello-agents/code`
Expected: no matches in migrated files; any remaining matches must be intentionally documented before merge

- [ ] **Step 3: Run a compile sweep across all changed Python files**

Run: `python -m compileall hello-agents/code/shared hello-agents/code/chapter1 hello-agents/code/chapter4 hello-agents/code/chapter6 hello-agents/code/chapter7 hello-agents/code/chapter8 hello-agents/code/chapter9 hello-agents/code/chapter10 hello-agents/code/chapter11 hello-agents/code/chapter12 hello-agents/code/chapter13/helloagents-trip-planner/backend/app hello-agents/code/chapter14/helloagents-deepresearch/backend/src hello-agents/code/chapter15/Helloagents-AI-Town/backend`
Expected: `Compiling ...` lines with no syntax errors

- [ ] **Step 4: Review the diff for accidental scope creep**

Run: `git diff -- hello-agents/code tests/hello_agents_code docs/superpowers/plans/2026-05-15-hello-agents-shared-env-config.md`
Expected: only env-loading, example-file, and test updates

- [ ] **Step 5: Commit the verification cleanup**

```bash
git add hello-agents/code tests/hello_agents_code docs/superpowers/plans/2026-05-15-hello-agents-shared-env-config.md
git commit -m "test: verify shared hello-agents env migration"
```

## Notes for the implementer

- Do not rename the user-approved keys. The shared `.env` contract is `API_KEY`, `BASE_URL`, `MODEL_ID`, and optional `LLM_TIMEOUT`.
- `TAVILY_API_KEY`, `GITHUB_TOKEN`, and `WANDB_API_KEY` must never be read from `hello-agents/code/.env`. They must come from `os.environ` only.
- For standalone scripts, avoid reintroducing raw `load_dotenv()` calls. Always go through `shared.env_config`.
- For the app configs in chapters 13–15, preserve their current class structure and public API; only change how raw values are sourced.
- `chapter11/config.json` remains in place. Do not fold it into `.env`.
- If a migrated file still needs a non-secret runtime env like `HF_ENDPOINT`, read it through `get_env_value()` from the shared `.env`, then write it into `os.environ` only when the downstream library requires process-level environment variables.
