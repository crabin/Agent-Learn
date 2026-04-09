from pydantic import BaseModel

from type import Tool
from enum import Enum

class Operation(str, Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"

class CalculatorParameters(BaseModel):
    num1: float
    num2: float
    operation: Operation


class Calculator(Tool):
    name = "Calculator"
    description = "A tool for performing basic arithmetic operations."
    parameters = CalculatorParameters

    async def execute(self, params: CalculatorParameters) -> str:
        
        match params.operation:
            case Operation.ADD:
                result = params.num1 + params.num2
            case Operation.SUBTRACT:
                result = params.num1 - params.num2
            case Operation.MULTIPLY:
                result = params.num1 * params.num2
            case Operation.DIVIDE:
                result = params.num1 / params.num2
            case _:
                return "Invalid operation"
        return f"The result of {params.operation}ing {params.num1} and {params.num2} is {result}."