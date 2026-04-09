import asyncio

from SandBox import Sandbox
from Agent import Agent

async def main():
    agent = Agent(Sandbox())
    # user_query = "请计算 10 + 5 的结果，并打印出来。"
    user_query = "(48/6+7)*(15-9)-5*(3+2)"
    response = await agent.query(user_query)
    print(f"Agent 响应:\n{response}")

if __name__ == "__main__":  
    asyncio.run(main())



"""
 python main.py

=== User Query ===
(48/6+7)*(15-9)-5*(3+2)

=== Request Messages ===
[{'role': 'system', 'content': '\nYou will be given a task to perform. You should output either\n- a Python code snippet that provides the solution to the task, or a step towards the solution. Any output you want to extract from the code should be printed with print(). Code should be output in a fenced code block.\n- text to be shown directly to the user, if you want to ask for more information or provide the final answer.\n- Only Python code block is valid\n\nImportant rules:\n- The runtime executes Python code with exec(), not JavaScript.\n- Do not use JavaScript syntax such as const, let, function, or console.log.\n- When writing code blocks, prefer ```python as the fence language.\n- If the task is solvable with code, output only a single Python fenced code block.\n\nIn addition to Python builtins, you can use the following functions:\n\nsum(a, b)  # a + b\nsub(a, b)  # a - b\nmul(a, b)  # a * b\ndiv(a, b)  # a / b\n'}, {'role': 'user', 'content': '(48/6+7)*(15-9)-5*(3+2)'}]

=== Raw Model Message ===
ChatCompletionMessage(content='```python\nresult = (48/6+7)*(15-9)-5*(3+2)\nprint(result)\n```', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning="The user wants me to evaluate this mathematical expression: (48/6+7)*(15-9)-5*(3+2)\n\nLet me break it down step by step:\n\n1. 48/6 + 7 = 8 + 7 = 15\n2. 15 - 9 = 6\n3. 3 + 2 = 5\n4. 15 * 6 = 90\n5. 5 * 5 = 25\n6. 90 - 25 = 65\n\nSo the answer is 65.\n\nLet me write this in Python code:\n\n```python\nresult = (48/6+7)*(15-9)-5*(3+2)\nprint(result)\n```\n\nOr I could calculate it step by step:\n\n```python\nstep1 = 48/6 + 7\nstep2 = 15 - 9\nstep3 = 3 + 2\nresult = step1 * step2 - 5 * step3\nprint(result)\n```\n\nBoth approaches will give the same answer. Let me use the first one since it's more direct.")

=== Raw Model Content ===
```python
result = (48/6+7)*(15-9)-5*(3+2)
print(result)
```

=== Agent Received Content ===
```python
result = (48/6+7)*(15-9)-5*(3+2)
print(result)
```

=== Extracted Code ===
result = (48/6+7)*(15-9)-5*(3+2)
print(result)

=== Sandbox Output ===
65.0

Agent 响应:
65.0
"""