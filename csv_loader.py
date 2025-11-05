#rom langchain_community.document_loaders import CSVLoader

from langchain_community.document_loaders import CSVLoader



Loader = CSVLoader(file_path ='student_data.csv')
docs=Loader.load()

print(len(docs))
print("Row 1 content:", docs[0].page_content)
print("Row 1 metadata:", docs[0].metadata)

