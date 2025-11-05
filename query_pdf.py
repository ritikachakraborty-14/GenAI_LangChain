from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Step 1: Load the PDF
pdf_path = "ALGO_SUGGESTION micro.pdf"  # Change to your PDF file name
loader = PyPDFLoader("c:\Users\ritik\Downloads")
pages = loader.load()

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(pages)

# Step 3: Create embeddings & vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 4: Create a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 5: Create an LLM (ChatGPT model)
llm = ChatOpenAI(model="gpt-4o-mini")  # You can use gpt-4o or gpt-3.5-turbo

# Step 6: Create a RetrievalQA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 7: Ask a question
query = "Summarize the main topic of this PDF."
response = qa.run(query)

print("📘 Question:", query)
print("💬 Answer:", response)
