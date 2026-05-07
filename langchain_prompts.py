import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate,FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
load_dotenv()

llm=ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=4000,
    max_retries=3
)

"""
#prompt template
prompt=PromptTemplate.from_template("how to become {profession}?")

chain=prompt | llm | StrOutputParser()
response=chain.invoke({"profession":"a data scientist"})
print(response)


#chat prompt template
chat_prompt=ChatPromptTemplate.from_messages([
    {"role":"system", "content":"You are a rude and sarcastic assistant in SBI agent."},
    {"role":"user", "content":"{question}"},
    {"role":"system", "content":"Memory: you are from vizag and its lunch time."}
])
chain1=chat_prompt | llm | StrOutputParser()
response=chain1.invoke({"question":"my bank account is blocked, what should i do?"})
print(response)
"""
#fewshot prompt template
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="input: {input}\noutput: {output}"
)

# Few-shot prompt
fewshot_prompt = FewShotPromptTemplate(
    examples=[
        {"input": "vizag", "output": "city of destiny"},
        {"input": "hyderabad", "output": "city of biriyani"},
        {"input": "delhi", "output": "city of politics"},
        {"input": "mumbai", "output": "city of dreams"},
    ],
    example_prompt=example_prompt,    
    prefix="Answer using the pattern shown below.",
    suffix="input: {question}\noutput:",
    input_variables=["question"],       
)

chain2 = fewshot_prompt | llm | StrOutputParser()

response = chain2.invoke({"question": "bangalore"})
print(response)
