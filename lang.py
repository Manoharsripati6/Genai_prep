import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm=ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.5,
    max_tokens=4000,
    max_retries=3,
)
prompt=ChatPromptTemplate.from_messages([
    {"role": "system", "content": "You are Stephen Hawking, the world-renowned theoretical physicist and cosmologist. You have a deep understanding of the universe, black holes, and the nature of time. You are known for your ability to explain complex scientific concepts in a way that is accessible to everyone. You have a dry sense of humor and a passion for sharing knowledge with others."}
    ,{"role": "user", "content": "{question}"}
])

llm=prompt | llm | StrOutputParser()

response=llm.invoke({"question":"why are you in epstien files?"})

for i in response.split("\n"):
    for j in range(len(i)):
        print(i[j], end="", flush=True)
        time.sleep(0.025)
    print("\n")
    