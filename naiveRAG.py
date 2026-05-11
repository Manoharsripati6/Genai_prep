from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


#loader
loader = PyPDFLoader("data/ragfeed.pdf")
pages = loader.load()
#print(pages[0].page_content)

#splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)
#print(f"Number of chunks: {len(chunks)}")
#print(f"Content of the first chunk:\n{chunks[0].page_content}")
 
#embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#storing in chroma db
vector_store = Chroma.from_documents(
    chunks, 
    embeddings, 
    collection_name="my_collection", 
    persist_directory="data/vector_store_rag"
)


# Add a simple document manually
docs_simple = [
    Document(
        page_content="The Principal is the Boss and name is : Dr. Smith.",
        metadata={"title": "The Boss", "source": "manual"}
    )
]
vector_store.add_documents(docs_simple)

print("Vector Store Updated Successfully!!!")

#creating a retriever
retriever = vector_store.as_retriever(
    search_kwargs = {"k" : 3}
)

#llm
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    # max_tokens=None,
    # top_p=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """\
You are a really helpful College staff assistant.
You're going to help the College staff with the relevant
information that has been provided with you. If the question
is not related to the College staff,
just say that you don't have enough information regarding this question.
You have to answer the questions based on the following context.

Context: {context}
"""),
    ("human", "{question}")
])
# document formatting
def format_docs(docs):
    doc_string = "\n\n".join(doc.page_content for doc in docs)
    return doc_string

# Rag Chain
rag_chain = (
    RunnableParallel(
        context=retriever,
        question=RunnablePassthrough()
    )
    | prompt
    | llm
    | StrOutputParser()
)
# User Query

print(rag_chain.invoke(input("Ask a Question : ")))
