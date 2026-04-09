import sys
import io

class _Console:
    def log(self, *args, **kwargs):
        print(*args, **kwargs)

class Sandbox:
    def __init__(self):
        # 1. 定义全局可用函数 (类似 TS 中的 global.sum)
        self.globals_env = {
            "sum": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: a / b,
            "console": _Console(),
            "__builtins__": __builtins__ # 允许使用内置函数如 print
        }

    def run(self, code: str):
        # 2. 捕获标准输出 (类似重写 console.log)
        output_buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = output_buffer

        try:
            # 3. 执行代码块
            exec(code, self.globals_env)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # 4. 恢复标准输出
            sys.stdout = old_stdout

        # 返回捕获到的日志字符串
        return output_buffer.getvalue()




if __name__ == "__main__":
    import textwrap
    
    sandbox = Sandbox()
    user_code = textwrap.dedent("""
        result = sum(10, 5)
        print(f"计算结果: {result}")
    """)
    logs = sandbox.run(user_code)
    print(f"沙箱输出:\n{logs}")
