import os
os.environ["OPENAI_API_KEY"] = "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"

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Say hello from LangChain!")
print(response.content)
