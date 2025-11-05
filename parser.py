from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ✅ Initialize endpoint (no chat wrapper used)
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    max_new_tokens=200
)

template1 = PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=["topic"]
)

topic = "Artificial Intelligence in Education"
prompt_text = template1.format(topic=topic)

# ✅ Call directly
response = llm.invoke(prompt_text)
print(response)
