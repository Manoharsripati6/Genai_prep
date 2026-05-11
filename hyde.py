from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
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
    path="./chroma_db_rules"
)

vectorstore = Chroma(
    client=client,
    collection_name="service_rules",
    embedding_function=embeddings
)
hyde_prompt = PromptTemplate(

    input_variables=["question"],

    template="""
You are an expert technical writer.

Write a detailed hypothetical answer
for the user's question.

The answer should:
- sound like real documentation
- contain technical terminology
- directly answer the question
- be information rich

Question:
{question}

Hypothetical Answer:
"""
)

hyde_chain = (

    hyde_prompt

    | llm

    | StrOutputParser()
)

question = input("\nEnter Question: ")


hypothetical_doc = hyde_chain.invoke({
    "question": question
})

print("\n" + "=" * 60)
print("HYPOTHETICAL DOCUMENT")
print("=" * 60)

print(hypothetical_doc)


retriever = vectorstore.as_retriever(

    search_type="mmr",

    search_kwargs={
        "k": 4,
        "fetch_k": 20
    }
)

docs = retriever.invoke(hypothetical_doc)

print("\n" + "=" * 60)
print("HyDE RETRIEVAL RESULTS")
print("=" * 60)

for i, doc in enumerate(docs, start=1):

    print(f"\nDocument {i}")

    print("-" * 40)

    source = doc.metadata.get(
        "source",
        "Unknown"
    )

    print(f"Source: {source}\n")

    print(doc.page_content[:400])