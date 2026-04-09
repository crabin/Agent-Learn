from Agent import Agent
from Calculator import Calculator
import asyncio

async def main():
    tools = [ Calculator() ]
    agent = Agent(tools=tools)

    query = "(48÷6+7)×(15-9)-5×(3+2)"
    response = await agent.query(query)
    print("Final Response:", response)

if __name__ == "__main__":
    asyncio.run(main())


"""
gemma4:e2b
 python mian.py

=== Step 1 ===
Reasoning/Response: The user wants me to evaluate the mathematical expression: (48÷6+7)×(15-9)-5×(3+2).

I will follow the order of operations (PEMDAS/BODMAS) to solve the expression.

1.  **Innermost parentheses:**
    *   (48 ÷ 6 + 7): First division: 48 ÷ 6 = 8. Then addition: 8 + 7 = 15.
    *   (15 - 9): 15 - 9 = 6.
    *   (3 + 2): 3 + 2 = 5.

    The expression becomes: $15 \times 6 - 5 \times 5$.

2.  **Multiplication:**
    *   $15 \times 6 = 90$
    *   $5 \times 5 = 25$

    The expression becomes: $90 - 25$.

3.  **Subtraction:**
    *   $90 - 25 = 65$.

The final result is 65.

65
Final Response: The user wants me to evaluate the mathematical expression: (48÷6+7)×(15-9)-5×(3+2).

I will follow the order of operations (PEMDAS/BODMAS) to solve the expression.

1.  **Innermost parentheses:**
    *   (48 ÷ 6 + 7): First division: 48 ÷ 6 = 8. Then addition: 8 + 7 = 15.
    *   (15 - 9): 15 - 9 = 6.
    *   (3 + 2): 3 + 2 = 5.

    The expression becomes: $15 \times 6 - 5 \times 5$.

2.  **Multiplication:**
    *   $15 \times 6 = 90$
    *   $5 \times 5 = 25$

    The expression becomes: $90 - 25$.

3.  **Subtraction:**
    *   $90 - 25 = 65$.

The final result is 65.

65
------------------
qwen3.5:4b

 python mian.py 

=== Step 1 ===
Reasoning/Response: 
Action: Calculator({'num1': 48, 'num2': 6, 'operation': 'divide'})
Observation: The result of Operation.DIVIDEing 48.0 and 6.0 is 8.0.

=== Step 2 ===
Reasoning/Response: 
Action: Calculator({'num1': 8, 'num2': 7, 'operation': 'add'})
Observation: The result of Operation.ADDing 8.0 and 7.0 is 15.0.

=== Step 3 ===
Reasoning/Response: 
Action: Calculator({'num1': 15, 'num2': 9, 'operation': 'subtract'})
Observation: The result of Operation.SUBTRACTing 15.0 and 9.0 is 6.0.

=== Step 4 ===
Reasoning/Response: 
Action: Calculator({'num1': 15, 'num2': 6, 'operation': 'multiply'})
Observation: The result of Operation.MULTIPLYing 15.0 and 6.0 is 90.0.

=== Step 5 ===
Reasoning/Response: 
Action: Calculator({'num1': 5, 'num2': 3, 'operation': 'add'})
Observation: The result of Operation.ADDing 5.0 and 3.0 is 8.0.

=== Step 6 ===
Reasoning/Response: 
Action: Calculator({'num1': 5, 'num2': 5, 'operation': 'multiply'})
Observation: The result of Operation.MULTIPLYing 5.0 and 5.0 is 25.0.

=== Step 7 ===
Reasoning/Response: 
Action: Calculator({'num1': 90, 'num2': 25, 'operation': 'subtract'})
Observation: The result of Operation.SUBTRACTing 90.0 and 25.0 is 65.0.

=== Step 8 ===
Reasoning/Response: The final result is **65**.

Here's the step-by-step calculation:
1. (48÷6+7) = (8+7) = **15**
2. (15-9) = **6**
3. 15×6 = **90**
4. (3+2) = **5**
5. 5×5 = **25**
6. 90-25 = **65**
Final Response: The final result is **65**.

Here's the step-by-step calculation:
1. (48÷6+7) = (8+7) = **15**
2. (15-9) = **6**
3. 15×6 = **90**
4. (3+2) = **5**
5. 5×5 = **25**
6. 90-25 = **65**
"""