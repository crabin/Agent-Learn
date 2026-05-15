from my_llm import MyLLM

llm = MyLLM(provider="modelscope")

messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]

response_stream = llm.think(messages)

print("ModelScope Response:")
for chunk in response_stream:
    pass
