from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = REPO_ROOT / "hello-agents" / "code"


def load_env_config_module() -> ModuleType:
    spec = spec_from_file_location("shared.env_config", CODE_ROOT / "shared" / "env_config.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load shared.env_config module")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_find_code_root_returns_hello_agents_code_dir(tmp_path: Path) -> None:
    code_root = tmp_path / "hello-agents" / "code"
    nested = code_root / "chapter7"
    nested.mkdir(parents=True)

    env_config = load_env_config_module()

    assert env_config.find_code_root(nested) == code_root


def test_get_llm_config_reads_shared_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code_root = tmp_path / "hello-agents" / "code"
    code_root.mkdir(parents=True)
    (code_root / ".env").write_text(
        "API_KEY=test-key\nBASE_URL=https://example.com/v1\nMODEL_ID=test-model\nLLM_TIMEOUT=45\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_ID", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT", raising=False)

    env_config = load_env_config_module()
    llm = env_config.get_llm_config(code_root=code_root)

    assert llm.api_key == "test-key"
    assert llm.base_url == "https://example.com/v1"
    assert llm.model_id == "test-model"
    assert llm.timeout == 45


def test_shared_dotenv_values_override_same_named_process_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code_root = tmp_path / "hello-agents" / "code"
    code_root.mkdir(parents=True)
    (code_root / ".env").write_text(
        "API_KEY=dotenv-key\nBASE_URL=https://dotenv.example/v1\nMODEL_ID=dotenv-model\nLLM_TIMEOUT=75\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("API_KEY", "process-key")
    monkeypatch.setenv("BASE_URL", "https://process.example/v1")
    monkeypatch.setenv("MODEL_ID", "process-model")
    monkeypatch.setenv("LLM_TIMEOUT", "15")

    env_config = load_env_config_module()
    llm = env_config.get_llm_config(code_root=code_root)

    assert llm.api_key == "dotenv-key"
    assert llm.base_url == "https://dotenv.example/v1"
    assert llm.model_id == "dotenv-model"
    assert llm.timeout == 75


@pytest.mark.parametrize("name", ["TAVILY_API_KEY", "GITHUB_TOKEN", "WANDB_API_KEY"])
def test_get_system_env_rejects_missing_required_secret(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(name, raising=False)
    env_config = load_env_config_module()

    with pytest.raises(ValueError, match=name):
        env_config.get_system_env(name)


def test_get_env_value_rejects_system_only_keys() -> None:
    env_config = load_env_config_module()

    with pytest.raises(ValueError, match="must be read with get_system_env"):
        env_config.get_env_value("TAVILY_API_KEY")


def test_root_env_example_lists_shared_and_system_only_keys() -> None:
    content = (CODE_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "API_KEY=" in content
    assert "BASE_URL=" in content
    assert "MODEL_ID=" in content
    assert "TAVILY_API_KEY" not in content
    assert "GITHUB_TOKEN" not in content
    assert "WANDB_API_KEY" not in content


def test_first_agent_script_imports_shared_env_config() -> None:
    content = (CODE_ROOT / "chapter1" / "FirstAgentTest.py").read_text(encoding="utf-8")

    assert "from shared.env_config import get_llm_config, get_system_env" in content
    assert 'API_KEY = "YOUR_API_KEY"' not in content
    assert 'os.environ[\'TAVILY_API_KEY\'] = "YOUR_TAVILY_API_KEY"' not in content


def test_wave_one_scripts_stop_calling_load_dotenv_directly() -> None:
    files = [
        CODE_ROOT / "chapter4" / "Plan_and_solve.py",
        CODE_ROOT / "chapter4" / "tools.py",
        CODE_ROOT / "chapter6" / "Langgraph" / "Dialogue_System.py",
        CODE_ROOT / "chapter6" / "AutoGenDemo" / "autogen_software_team.py",
        CODE_ROOT / "chapter6" / "CAMEL" / "DigitalBookWriting.py",
        CODE_ROOT / "chapter6" / "AgentScopeDemo" / "main_cn.py",
        CODE_ROOT / "chapter7" / "my_main.py",
        CODE_ROOT / "chapter7" / "test_advanced_search.py",
        CODE_ROOT / "chapter7" / "test_plan_solve_agent.py",
        CODE_ROOT / "chapter7" / "test_react_agent.py",
        CODE_ROOT / "chapter7" / "test_reflection_agent.py",
        CODE_ROOT / "chapter7" / "test_simple_agent.py",
        CODE_ROOT / "chapter7" / "test_my_calculator.py",
    ]

    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        assert "load_dotenv(" not in content, str(file_path)


def test_wave_one_modules_use_shared_env_helpers() -> None:
    llm_client_content = (CODE_ROOT / "chapter4" / "llm_client.py").read_text(encoding="utf-8")
    advanced_search_content = (CODE_ROOT / "chapter7" / "my_advanced_search.py").read_text(encoding="utf-8")
    my_llm_content = (CODE_ROOT / "chapter7" / "my_llm.py").read_text(encoding="utf-8")
    agentscope_content = (CODE_ROOT / "chapter6" / "AgentScopeDemo" / "main_cn.py").read_text(encoding="utf-8")

    assert "from shared.env_config import get_llm_config" in llm_client_content
    assert 'os.getenv("LLM_API_KEY")' not in llm_client_content
    assert 'os.getenv("LLM_BASE_URL")' not in llm_client_content
    assert 'os.getenv("LLM_MODEL_ID")' not in llm_client_content

    assert 'get_system_env("TAVILY_API_KEY")' in advanced_search_content
    assert 'os.getenv("TAVILY_API_KEY")' not in advanced_search_content

    assert "from shared.env_config import get_llm_config" in my_llm_content
    assert 'os.getenv("MODELSCOPE_API_KEY")' not in my_llm_content

    assert "from shared.env_config import get_env_value" in agentscope_content
    assert 'os.environ["DASHSCOPE_API_KEY"]' not in agentscope_content


def test_first_agent_notebook_uses_shared_env_helpers() -> None:
    content = (CODE_ROOT / "chapter1" / "FirstAgentTest.ipynb").read_text(encoding="utf-8")

    assert "from shared.env_config import get_llm_config, get_system_env" in content
    assert 'load_dotenv()' not in content
    assert 'os.environ[\'TAVILY_API_KEY\'] = TAVILY_API_KEY' not in content


def test_wave_two_scripts_stop_calling_load_dotenv_directly() -> None:
    files = [
        CODE_ROOT / "chapter8" / "01_MemoryTool_Basic_Operations.py",
        CODE_ROOT / "chapter8" / "02_MemoryTool_Architecture.py",
        CODE_ROOT / "chapter8" / "03_WorkingMemory_Implementation.py",
        CODE_ROOT / "chapter8" / "04_RAGTool_MarkItDown_Pipeline.py",
        CODE_ROOT / "chapter8" / "05_RAGTool_Advanced_Search.py",
        CODE_ROOT / "chapter8" / "06_Memory_Consolidation_Demo.py",
        CODE_ROOT / "chapter8" / "07_RAGTool_Intelligent_QA.py",
        CODE_ROOT / "chapter8" / "08_Agent_Tool_Integration.py",
        CODE_ROOT / "chapter8" / "09_Memory_Types_Deep_Dive.py",
        CODE_ROOT / "chapter8" / "10_RAG_Pipeline_Complete.py",
        CODE_ROOT / "chapter8" / "11_Q&A_Assistant.py",
        CODE_ROOT / "chapter9" / "01_context_builder_basic.py",
        CODE_ROOT / "chapter9" / "02_context_builder_with_agent.py",
        CODE_ROOT / "chapter9" / "04_note_tool_integration.py",
        CODE_ROOT / "chapter9" / "06_three_day_workflow.py",
        CODE_ROOT / "chapter10" / "06_MultiAgentDocumentAssist.py",
        CODE_ROOT / "chapter10" / "10_A2ATool_Simple.py",
        CODE_ROOT / "chapter10" / "10_CustomerService.py",
        CODE_ROOT / "chapter10" / "12_ANPTaskDistribution.py",
        CODE_ROOT / "chapter10" / "14_weather_agent.py",
    ]

    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        assert "load_dotenv(" not in content, str(file_path)
        assert "HelloAgents/.env" not in content, str(file_path)


def test_wave_two_scripts_use_shared_env_bootstrap() -> None:
    files = [
        CODE_ROOT / "chapter8" / "01_MemoryTool_Basic_Operations.py",
        CODE_ROOT / "chapter9" / "01_context_builder_basic.py",
        CODE_ROOT / "chapter10" / "06_MultiAgentDocumentAssist.py",
        CODE_ROOT / "chapter10" / "14_weather_agent.py",
    ]

    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        assert "from shared.env_config import load_shared_dotenv" in content, str(file_path)
        assert "load_shared_dotenv(code_root=CODE_ROOT)" in content, str(file_path)



def test_wave_two_special_cases_keep_intended_env_behavior() -> None:
    workflow_content = (CODE_ROOT / "chapter9" / "06_three_day_workflow.py").read_text(encoding="utf-8")
    weather_server_content = (CODE_ROOT / "chapter10" / "weather-mcp-server" / "server.py").read_text(encoding="utf-8")
    gaia_quick_start_content = (CODE_ROOT / "chapter12" / "05_gaia_quick_start.py").read_text(encoding="utf-8")

    assert 'get_env_value("HF_ENDPOINT", default="https://hf-mirror.com", code_root=CODE_ROOT)' in workflow_content
    assert 'os.environ.setdefault("HF_ENDPOINT", hf_endpoint)' in workflow_content
    assert 'os.getenv("PORT", 8081)' in weather_server_content
    assert 'os.getenv("HOST", "0.0.0.0")' in weather_server_content
    assert 'os.environ["HF_TOKEN"] = "your_huggingface_token_here"' not in gaia_quick_start_content



def test_app_config_files_delegate_llm_values_to_shared_env_module() -> None:
    config_files = [
        CODE_ROOT / "chapter13" / "helloagents-trip-planner" / "backend" / "app" / "config.py",
        CODE_ROOT / "chapter14" / "helloagents-deepresearch" / "backend" / "src" / "config.py",
        CODE_ROOT / "chapter15" / "Helloagents-AI-Town" / "backend" / "config.py",
    ]

    for file_path in config_files:
        content = file_path.read_text(encoding="utf-8")
        assert "from shared.env_config import" in content, str(file_path)
        assert 'os.getenv("LLM_API_KEY")' not in content, str(file_path)
        assert 'os.getenv("LLM_MODEL_ID")' not in content, str(file_path)
        assert 'os.getenv("LLM_BASE_URL")' not in content, str(file_path)



def test_trip_planner_config_stops_loading_local_dotenv_files() -> None:
    content = (
        CODE_ROOT / "chapter13" / "helloagents-trip-planner" / "backend" / "app" / "config.py"
    ).read_text(encoding="utf-8")

    assert "load_dotenv(" not in content
    assert 'env_file = ".env"' not in content
    assert 'HelloAgents" / ".env"' not in content
