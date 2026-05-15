"""
10.3.4 在智能体中使用A2A工具
（1）使用A2ATool包装器
"""

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

from shared.env_config import load_shared_dotenv

load_shared_dotenv(code_root=CODE_ROOT)

from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import A2ATool

llm = HelloAgentsLLM()

# 假设已经有一个研究员Agent服务运行在 http://localhost:5000

# 创建协调者Agent
coordinator = SimpleAgent(name="协调者", llm=llm)

# 添加A2A工具，连接到研究员Agent
researcher_tool = A2ATool(agent_url="http://localhost:5000")
coordinator.add_tool(researcher_tool)

# 协调者可以调用研究员Agent
# 使用 action="ask" 向 Agent 提问
response = coordinator.run("使用a2a工具，向Agent提问：请研究AI在教育领域的应用")
print(response)

