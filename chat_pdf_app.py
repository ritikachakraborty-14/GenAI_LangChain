import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

st.set_page_config(page_title="📘 Chat with your PDF", layout="wide")

st.title("💬 Chat with your PDF using LangChain + OpenAI")

# Step 1: File uploader
pdf_file = st.file_uploader("📂 Upload your PDF file", type=["ALGO_SUGGESTION micro.pdf"])

if pdf_file is not None:
    with st.spinner("Loading and processing PDF..."):
        # Save uploaded file temporarily
        with open("ALGO_SUGGESTION micro.pdf", "wb") as f:
            f.write(ALGO_SUGGESTION micro.pdf.read())

        # Load and split PDF
        loader = PyPDFLoader("ALGO_SUGGESTION micro.pdf")
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(pages)

        # Create embeddings & vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Create LLM and chain
        llm = ChatOpenAI(model="gpt-4o-mini")
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("✅ PDF loaded successfully! You can now ask questions below:")

        # Chat input
        query = st.text_input("💬 Ask a question about your PDF:")
        if query:
            with st.spinner("Thinking..."):
                answer = qa.run(query)
                st.markdown(f"**Answer:** {answer}")
