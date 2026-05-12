from dotenv import load_dotenv

import chromadb

from operator import itemgetter

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)

from langchain_core.runnables.history import (
    RunnableWithMessageHistory
)

from langchain_community.chat_message_histories import (
    ChatMessageHistory
)


load_dotenv()
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
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

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Rewrite the user's question into a standalone question
using the conversation history.

Only rewrite if necessary.
"""
    ),

    MessagesPlaceholder("chat_history"),

    ("human", "{input}")
])



rewrite_chain = (
    rewrite_prompt
    | llm
    | StrOutputParser()
)



qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an college AI assistant.

Answer ONLY from the provided context.

If the answer is unavailable, say:

"I don't have enough information from the documents."

Context:
{context}
"""
    ),

    MessagesPlaceholder("chat_history"),

    ("human", "{input}")
])



retrieval_chain = (
    RunnablePassthrough.assign(
        standalone_question=rewrite_chain
    )

    | RunnablePassthrough.assign(
        context=(
            itemgetter("standalone_question")
            | retriever
            | RunnableLambda(format_docs)
        )
    )
)



rag_chain = (
    retrieval_chain
    | qa_prompt
    | llm
    | StrOutputParser()
)



store = {}

def get_session_history(session_id):

    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]

chatbot = RunnableWithMessageHistory(

    rag_chain,

    get_session_history,

    input_messages_key="input",

    history_messages_key="chat_history"
)


while True:

    question = input("\nAsk: ")

    if question.lower() == "exit":
        break

    response = chatbot.invoke(

        {"input": question},

        config={
            "configurable": {
                "session_id": "user_001"
            }
        }
    )

    print("\nAnswer")
    print("=" * 50)

    print(response)