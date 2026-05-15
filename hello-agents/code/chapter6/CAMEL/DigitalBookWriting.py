from pathlib import Path
import sys

from colorama import Fore
from camel.models import ModelFactory
from camel.societies import RolePlaying
from camel.types import ModelPlatformType
from camel.utils import print_text_animated


def _find_code_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if candidate.name == "code" and candidate.parent.name == "hello-agents":
            return candidate
    raise ValueError("Unable to locate hello-agents/code root")


CODE_ROOT = _find_code_root(Path(__file__).resolve().parent)
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shared.env_config import get_llm_config

LLM_CONFIG = get_llm_config(code_root=CODE_ROOT)
LLM_API_KEY = LLM_CONFIG.api_key
LLM_BASE_URL = LLM_CONFIG.base_url
LLM_MODEL = LLM_CONFIG.model_id

#创建模型,在这里以Qwen为例,调用的百炼大模型平台API
model = ModelFactory.create(
    model_platform=ModelPlatformType.QWEN,
    model_type=LLM_MODEL,
    url=LLM_BASE_URL,
    api_key=LLM_API_KEY
)

# 定义协作任务
task_prompt = """
创作一本关于"拖延症心理学"的短篇电子书，目标读者是对心理学感兴趣的普通大众。
要求：
1. 内容科学严谨，基于实证研究
2. 语言通俗易懂，避免过多专业术语
3. 包含实用的改善建议和案例分析
4. 篇幅控制在8000-10000字
5. 结构清晰，包含引言、核心章节和总结
"""

print(Fore.YELLOW + f"协作任务:\n{task_prompt}\n")

# 初始化角色扮演会话
role_play_session = RolePlaying(
    assistant_role_name="心理学家", 
    user_role_name="作家", 
    task_prompt=task_prompt,
    model=model
)

print(Fore.CYAN + f"具体任务描述:\n{role_play_session.task_prompt}\n")

# 开始协作对话
chat_turn_limit, n = 30, 0
input_msg = role_play_session.init_chat()

while n < chat_turn_limit:
    n += 1
    assistant_response, user_response = role_play_session.step(input_msg)
    
    print_text_animated(Fore.BLUE + f"作家:\n\n{user_response.msg.content}\n")
    print_text_animated(Fore.GREEN + f"心理学家:\n\n{assistant_response.msg.content}\n")
    
    # 检查任务完成标志
    if "CAMEL_TASK_DONE" in user_response.msg.content:
        print(Fore.MAGENTA + "✅ 电子书创作完成！")
        break
    
    input_msg = assistant_response.msg

print(Fore.YELLOW + f"总共进行了 {n} 轮协作对话")