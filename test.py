from langchain_ollama import ChatOllama

llm=ChatOllama(
    model='gemma:2b',
    temperature=0.1
)

response=llm.invoke("Hello, how are you?")
print(response.content)