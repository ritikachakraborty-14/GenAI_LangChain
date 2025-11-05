from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
Access_Token="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"
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=Access_Token,
    temperature=0,
    max_new_tokens=256
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("National Flower of India?")

poem = model.invoke("write a 5 line poem on indian flower in bengali language")
print(result.content)
print(poem.content)