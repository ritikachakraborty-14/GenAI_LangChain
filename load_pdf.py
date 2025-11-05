from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
os.environ["Test_Key"] = "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"


pdf_path = r"C:\Users\ritik\Downloads\8.4_IdentityIQ_System_Configuration.pdf"

loader = PyPDFLoader(pdf_path)
pages = loader.load()

print(f"✅ Pages loaded: {len(pages)}")
if len(pages) == 0:
    print("❌ PDF has no readable pages or is encrypted.")
    exit()

# ✅ Show first page text to check if it's readable
print("\n🔍 Sample page content preview:\n")
print(pages[5].page_content[:500])  # print first 500 characters

# ✅ Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)

print(f"\n✅ Total chunks created: {len(chunks)}")
if len(chunks) > 0:
    print("\n📚 First chunk preview:\n", chunks[100].page_content[:1000])
else:
    print("⚠️ No text chunks created. PDF may be scanned or image-based.")
