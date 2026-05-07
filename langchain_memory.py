import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory

load_dotenv()

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=1.4)

# Buffer Memory
memory = ConversationBufferMemory()
conversation = ConversationChain(llm = llm, memory= memory)

# responses = [
#     conversation.predict(input="hi my name is manohar"),
#     conversation.predict(input="what is my name"),
# ]

# for res in responses:
#     print(res)

# new_response = conversation.invoke({"input":"Write a Small Poem based on with name in english like a sri sri"})
# print(new_response["response"])

# Window Memory
convo_window = ConversationBufferWindowMemory(k=3)
conversation_window = ConversationChain(llm = llm, memory= convo_window)
# res1 = conversation_window.invoke({"input":"hi my favourite sport is Cricket"})
# res = conversation_window.invoke({"input":"hi My name is Manohar"})
# res3 = conversation_window.invoke({"input":"I am a data scientist"})

# res2 = conversation_window.invoke({"input":"tell me what is my name and what i do ? and what is my favourite sport ?"})
# print(res2["response"])

# Summary Memory
convo_summary = ConversationSummaryMemory(llm=llm)
conversation_summary = ConversationChain(llm=llm, memory=convo_summary)
res1 = conversation_summary.invoke({"input":"hi my favourite sport is Badminton"})
res = conversation_summary.invoke({"input":"hi My name is Manohar"})
res3 = conversation_summary.invoke({"input":"I am a data scientist"})

res2 = conversation_summary.invoke({"input":"tell me what is my name and what i do ? and what is my favourite sport ?"})
print(res2["response"])
