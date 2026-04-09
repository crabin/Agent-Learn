from typing import Any, Protocol, Type
from pydantic import BaseModel

class Tool(Protocol):
    name: str
    description: str
    parameters: Type[BaseModel] 

    async def execute(self, params: Any) -> str:
        ...