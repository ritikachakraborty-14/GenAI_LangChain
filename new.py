import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings


# -------- Step 1: Download PDF from API URL --------
pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"  # replace with your PDF API
response = requests.get(pdf_url)
if response.status_code == 200:
    with open("api_pdf.pdf", "wb") as file:
        file.write(response.content)
else:
    raise Exception(f"❌ Failed to download PDF. Status {response.status_code}")

# -------- Step 2: Load PDF --------
loader = PyPDFLoader("api_pdf.pdf")
documents = loader.load()

# -------- Step 3: Split into chunks --------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# -------- Step 4: Create embeddings + Chroma DB --------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(chunks, embeddings)

# -------- Step 5: Load Hugging Face LLM --------




llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",  # ✅ Free model
    task="text2text-generation",
    temperature=0.3,
    max_new_tokens=256,
    huggingfacehub_api_token="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"
)



# -------- Step 6: Create RAG QA chain --------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# -------- Step 7: Ask questions --------
while True:
    query = input("\nAsk a question about the PDF (or type 'exit'): ")
    if query.lower() == "exit":
        break
    result = qa.invoke({"query": query})
    print("\nAnswer:", result["result"])
