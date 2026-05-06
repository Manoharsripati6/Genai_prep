from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
client=OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
print("text")
response=client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {"role": "system", "content": "You are a PHD researcher."},
        {"role": "user", "content": "explain Deep Learning"}
    ]
    ,temperature=0.5,
    max_tokens=200,
    
)
print("message")
print(response.choices[0].message.content)
print("end")