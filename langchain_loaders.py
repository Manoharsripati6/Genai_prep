from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader

#pdf loader
loader = PyPDFLoader("sample.pdf")
pages = loader.load_and_split()

print(f"Number of pages: {len(pages)}")
print(f"Content of the first page:\n{pages[0].page_content}")
print(f"Metadata of the first page:\n{pages[0].metadata}")

#docx loader
docx_loader = Docx2txtLoader("sample.docx")
docx_docs = docx_loader.load()
print(f"Number of documents: {len(docx_docs)}")
print(f"Content of the first document:\n{docx_docs[0].page_content}")


#text loader
text_loader = TextLoader("sample.txt")
text_docs = text_loader.load()
print(f"Number of documents: {len(text_docs)}")
print(f"Content of the first document:\n{text_docs[0].page_content}")