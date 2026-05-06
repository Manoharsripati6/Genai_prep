import os

from openai import OpenAI
import time
client=OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
print("text")
response=client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {"role": "system", "content": "You are a PHD researcher.give me answer in briefpoints."},
        {"role": "user", "content": "i wanna learn about agenticAi? give 5 points how i can learn genAi?"}
    ]
    ,temperature=0.5,
    max_tokens=200,
    stream=True
    
)
print("message")
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
        time.sleep(0.2)
print("end")