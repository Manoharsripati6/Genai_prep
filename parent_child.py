from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import chromadb

load_dotenv()

# initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# load pdf document
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# parent splitter for large context
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

# child splitter for semantic search
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

# create persistent chroma client
client = chromadb.PersistentClient(
    path="./chroma_parent_child"
)

# create vector store
vectorstore = Chroma(
    client=client,
    collection_name="parent_child_demo",
    embedding_function=embeddings
)

# store parent documents
store = InMemoryStore()

# initialize parent-child retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    parent_splitter=parent_splitter,
    child_splitter=child_splitter,
)

# add documents into retriever
retriever.add_documents(documents)

# interactive query loop
while True:

    question = input("\nAsk Question: ")

    docs = retriever.invoke(question)

    print("\n" + "=" * 60)
    print("RETRIEVED PARENT DOCUMENTS")
    print("=" * 60)

    for i, doc in enumerate(docs, start=1):
        print(f"\nDocument {i}")
        print("-" * 40)
        print(doc.page_content[:500])