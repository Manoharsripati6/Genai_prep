import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
load_dotenv()

llm=ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.5,
    max_tokens=200,
    max_retries=3,
)
prompt=PromptTemplate.from_template("what is {topic}?")
#string output parser
chain=prompt | llm | StrOutputParser()
response=chain.invoke({"topic":"genAI"})
#print(response)

#Json output parser
chat_prompt=ChatPromptTemplate.from_template("given a {Country}, return its capital,population,gender ratio,litercy rate and many more statistics in json format.")
chain1=chat_prompt | llm | StrOutputParser()    
response1=chain1.invoke({"Country":"India"})
#print(response1)


# Pydantic Schema
class CountryStats(BaseModel):
    capital: str
    population: int
    gender_ratio: float
    literacy_rate: float

prompt = ChatPromptTemplate.from_template(
    """
    Given a {Country}, return:
    - capital
    - population
    - gender_ratio
    - literacy_rate

    IMPORTANT:
    Return valid structured data.
    population must be INTEGER.
    gender_ratio must be FLOAT.
    literacy_rate must be FLOAT.
    Do NOT use strings for numbers.
    """
)

structured_llm = llm.with_structured_output(CountryStats)

chain = prompt | structured_llm

response = chain.invoke({"Country": "India"})

print(response)