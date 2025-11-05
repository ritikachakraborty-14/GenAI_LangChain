# Ollamatesting.py
# Ollama + LangChain integration for PDF querying with Gemma3:4b

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Step 1: Load the PDF
pdf_path = "C:/Users/ritik/Downloads/example.pdf"
try:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    if not documents:
        raise ValueError("No content loaded from PDF. Check file path or content.")
except Exception as e:
    print(f"Error loading PDF: {e}")
    exit(1)

# Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Adjust as needed
    chunk_overlap=400,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)

# Step 3: Create embeddings and vector store
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"  # Saves to disk for reuse
    )
    vectorstore.persist()
except Exception as e:
    print(f"Error creating vector store: {e}")
    exit(1)

# Step 4: Set up retriever and LLM
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # Top 4 chunks
#llm = Ollama(model="gemma3:4b")  # Using gemma3:4b from your ollama list
Access_Token="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"
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=Access_Token,
    temperature=0,
    max_new_tokens=256
)

model = ChatHuggingFace(llm=llm)
# Step 5: Define prompt and RAG chain
prompt_template = """
Use the following context from the PDF to answer the question. If you don't know the answer, say so.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Step 6: Interactive querying
print("PDF processed! Enter questions (type 'quit' to exit):")
while True:
    question = input()
    if question.lower() == "quit":
        break
    try:
        response = chain.invoke(question)
        print(response)
    except Exception as e:
        print(f"Error processing question: {e}")
    print("\n")