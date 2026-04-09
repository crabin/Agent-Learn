from typing import Any, Protocol, Type
from pydantic import BaseModel

class Sandbox(Protocol):
    def execute(self, code: str) -> Any:
        ...