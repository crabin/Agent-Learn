"""配置文件"""

from pathlib import Path
import sys
from typing import Optional


def _find_code_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if candidate.name == "code" and candidate.parent.name == "hello-agents":
            return candidate
    raise ValueError("Unable to locate hello-agents/code root")


CODE_ROOT = _find_code_root(Path(__file__).resolve().parent)
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shared.env_config import get_env_value


class Settings:
    """应用配置"""

    API_TITLE = "赛博小镇 API"
    API_VERSION = "1.0.0"
    API_HOST = "0.0.0.0"
    API_PORT = 8000

    NPC_UPDATE_INTERVAL = 30

    LLM_MODEL_ID: str = (
        get_env_value("MODEL_ID", default="Qwen/Qwen2.5-72B-Instruct", code_root=CODE_ROOT)
        or "Qwen/Qwen2.5-72B-Instruct"
    )
    LLM_API_KEY: Optional[str] = get_env_value("API_KEY", default=None, code_root=CODE_ROOT)
    LLM_BASE_URL: str = (
        get_env_value(
            "BASE_URL",
            default="https://api-inference.modelscope.cn/v1/",
            code_root=CODE_ROOT,
        )
        or "https://api-inference.modelscope.cn/v1/"
    )

    CORS_ORIGINS = ["*"]

    @classmethod
    def validate(cls):
        """验证配置"""
        if not cls.LLM_API_KEY:
            print("⚠️  警告: 未在共享 .env 中设置 API_KEY")
            print("   请在 hello-agents/code/.env 中配置 API_KEY")
            print('   示例: API_KEY="your-api-key"')
            return False

        print("✅ LLM配置:")
        print(f"   模型: {cls.LLM_MODEL_ID}")
        print(f"   服务地址: {cls.LLM_BASE_URL}")
        return True


settings = Settings()
