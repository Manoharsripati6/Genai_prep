from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
import chromadb


load_dotenv()


llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="data/vector_store_rag"
)

vectorstore = Chroma(

    client=client,

    collection_name="my_collection",

    embedding_function=embeddings
)

base_retriever = vectorstore.as_retriever(

    search_type="mmr",

    search_kwargs={

        "k": 5,

        "fetch_k": 20,

        "lambda_mult": 0.5
    }
)


MULTI_QUERY_PROMPT = PromptTemplate(

    input_variables=["question"],

    template="""
You are an AI retrieval assistant.

Generate 4 different semantic variations
of the user's question for better retrieval.

Rules:
- Preserve original meaning
- Use different wording
- Use related terminology
- Keep queries concise

Return only the queries.
One query per line.

Question:
{question}
"""
)

multi_query_retriever = MultiQueryRetriever.from_llm(

    retriever=base_retriever,

    llm=llm,

    prompt=MULTI_QUERY_PROMPT,

    include_original=True
)


question = input("\nEnter Question: ")

standard_docs = base_retriever.invoke(question)

print("\n" + "=" * 60)
print("STANDARD RETRIEVAL")
print("=" * 60)

for i, doc in enumerate(standard_docs, start=1):

    print(f"\nDocument {i}")

    print("-" * 40)

    print(doc.page_content[:300])
mq_docs = multi_query_retriever.invoke(question)

print("\n" + "=" * 60)
print("MULTI QUERY RETRIEVAL")
print("=" * 60)

seen = set()

unique_docs = []

for doc in mq_docs:

    if doc.page_content not in seen:

        unique_docs.append(doc)

        seen.add(doc.page_content)

for i, doc in enumerate(unique_docs, start=1):

    print(f"\nDocument {i}")

    print("-" * 40)

    print(doc.page_content[:300])