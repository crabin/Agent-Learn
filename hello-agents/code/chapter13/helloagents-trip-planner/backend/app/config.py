"""配置管理模块"""

from pathlib import Path
import sys
from typing import List

from pydantic_settings import BaseSettings


def _find_code_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if candidate.name == "code" and candidate.parent.name == "hello-agents":
            return candidate
    raise ValueError("Unable to locate hello-agents/code root")


CODE_ROOT = _find_code_root(Path(__file__).resolve().parent)
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shared.env_config import get_env_value


class Settings(BaseSettings):
    """应用配置"""

    app_name: str = "HelloAgents智能旅行助手"
    app_version: str = "1.0.0"
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = 8000

    cors_origins: str = (
        "http://localhost:5173,http://localhost:3000,"
        "http://127.0.0.1:5173,http://127.0.0.1:3000"
    )

    amap_api_key: str = get_env_value("AMAP_API_KEY", default="", code_root=CODE_ROOT) or ""
    unsplash_access_key: str = (
        get_env_value("UNSPLASH_ACCESS_KEY", default="", code_root=CODE_ROOT) or ""
    )
    unsplash_secret_key: str = (
        get_env_value("UNSPLASH_SECRET_KEY", default="", code_root=CODE_ROOT) or ""
    )

    openai_api_key: str = get_env_value("API_KEY", default="", code_root=CODE_ROOT) or ""
    openai_base_url: str = (
        get_env_value("BASE_URL", default="https://api.openai.com/v1", code_root=CODE_ROOT)
        or "https://api.openai.com/v1"
    )
    openai_model: str = get_env_value("MODEL_ID", default="gpt-4", code_root=CODE_ROOT) or "gpt-4"

    log_level: str = "INFO"

    class Config:
        case_sensitive = False
        extra = "ignore"

    def get_cors_origins_list(self) -> List[str]:
        """获取 CORS origins 列表"""
        return [origin.strip() for origin in self.cors_origins.split(",")]


settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings


def validate_config() -> bool:
    """验证配置是否完整"""
    errors = []
    warnings = []

    if not settings.amap_api_key:
        errors.append("AMAP_API_KEY未配置")

    if not settings.openai_api_key:
        warnings.append("共享 .env 中的 API_KEY 未配置,LLM功能可能无法使用")

    if errors:
        error_msg = "配置错误:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    if warnings:
        print("\n⚠️  配置警告:")
        for warning in warnings:
            print(f"  - {warning}")

    return True


def print_config() -> None:
    """打印当前配置(隐藏敏感信息)"""
    print(f"应用名称: {settings.app_name}")
    print(f"版本: {settings.app_version}")
    print(f"服务器: {settings.host}:{settings.port}")
    print(f"高德地图API Key: {'已配置' if settings.amap_api_key else '未配置'}")
    print(f"LLM API Key: {'已配置' if settings.openai_api_key else '未配置'}")
    print(f"LLM Base URL: {settings.openai_base_url}")
    print(f"LLM Model: {settings.openai_model}")
    print(f"日志级别: {settings.log_level}")
