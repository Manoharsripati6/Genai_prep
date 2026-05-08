#ephemeral memory=local
#persistent memory=chroma 
#collection=collection of documents
#document=single document with metadata
#embedding=vector representation of a document
#metadata=additional information about a document
#id=unique identifier for a document
#query=search for documents based on metadata or content
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

#embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)
# Document -> data
data = [
    Document(
        page_content="The Refund Policy allows customers to return products within 30 days of purchase",
        metadata={
            "source": "refund_policy.pdf",
            "page_number": 1,
            "department":"sales"
        }
    ),
    Document(
        page_content="All employess get only 10 sick Leaves and 5 casual leaves per year.",
        metadata={
            "source": "hr_policy.pdf",
            "page_number": 1,
            "department":"hr"
        }
    ),
    Document(
        page_content="Every Developer must use python 3.13 version for development.",
        metadata={
            "source": "dev_policy.pdf",
            "page_number": 1,
            "department":"development"
        }
    )
]

vector_store = Chroma.from_documents(
    documents=data,
    embedding=embedding_model,
    collection_name="company_policies"
)

test_query = "can I use Python?"
#filtering by metadata
results = vector_store.similarity_search(
    query=test_query,
    filter={"department": "development"},
    k=2
)
for key, val in enumerate(results):
    print(key, val.page_content)
    print(key, val.metadata)



    