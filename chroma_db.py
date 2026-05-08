from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

data = [
    Document(
        page_content="The Refund Policy allows customers to return products within 30 days of purchase",
        metadata={"source": "refund_policy.pdf", "page_number": 1, "department": "sales"},
        id="refund_doc_1"
    ),
    Document(
        page_content="All employees get only 10 sick leaves and 5 casual leaves per year.",
        metadata={"source": "hr_policy.pdf", "page_number": 1, "department": "hr"},
        id="hr_doc_1"
    ),
    Document(
        page_content="Every Developer must use python 3.13 version for development.",
        metadata={"source": "dev_policy.pdf", "page_number": 1, "department": "development"},
        id="dev_doc_1"
    )
]

# CREATE + AUTO PERSIST
vector_store = Chroma.from_documents(
    documents=data,
    embedding=embedding_model,
    collection_name="company_policies_persistent",
    persist_directory="./vector_store_persistent"
)


"""
# ADD DOCUMENT
new_doc = Document(
    page_content="All employees must submit their timesheets by the end of the week.",
    metadata={"source": "hr_policy.pdf", "page_number": 2, "department": "hr"},
    id="hr_doc_2"
)

vector_store.add_documents([new_doc])
"""


"""
# DELETE DOCUMENT
vector_store.delete(ids=["hr_doc_2"])
"""


# LOAD EXISTING DB
load_vector_db = Chroma(
    collection_name="company_policies_persistent",
    embedding_function=embedding_model,
    persist_directory="./vector_store_persistent"
)

print("\n--- Accessing Existing Document ---")
res = load_vector_db.get(ids=["dev_doc_1"])
print(res)

retriever = load_vector_db.as_retriever(
    search_kwargs={"k": 1}
)

res = retriever.invoke("deadline for returning the product ?")

print(res[0].page_content)
print(res[0].metadata)