from openai import OpenAI
import time
from dotenv import load_dotenv
import os
load_dotenv()
client=OpenAI(
    api_key=os.getenv("GROQ_API"),
    base_url="https://api.groq.com/openai/v1",
)

def chat_with_memory():
    messages=[
        {"role":"system","content":"Answer in 2-3 lines. you are a PHD expert"}
    ]
    while True:
        print("\npress exit if you want to exit")
        inp=input("You: ").strip()
        if inp=="exit":
            print("\nbyeee")
            break            
        messages.append({"role":"user","content":inp})
        
        responses=client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.5,
            max_tokens=200
        )
        print("AI: ",end=" ")
        r=responses.choices[0].message.content
        messages.append({"role":"assistant","content":r})
        for i in r.split("\n"):
            print(i)
            time.sleep(0.7)

if __name__=="__main__":
    chat_with_memory()