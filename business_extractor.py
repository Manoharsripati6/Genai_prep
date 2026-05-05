import os
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
from typing import Optional 
from openai import OpenAI
client=OpenAI(
    api_key=os.getenv("GROQ_API"),
    base_url="https://api.groq.com/openai/v1",
)
class BusinessModel(BaseModel):
    company_name:str
    revenue:Optional[int]=None
    employee_count:Optional[int]=None
    established_year:Optional[int]=None
def extract(inp:str)->BusinessModel:
    msg=[{
        "role":"system","content":"you are a business analyst expert. extract the company name , revenue, established year and employee count if mentioned from the given text and give answer in json format with keys name, company_name and revenue. if anything is missing use your knowledge to fill the data. if still you are not able to find the data fill it with null"
    },{"role":"user", "content": inp}]
    responses=client.chat.completions.parse(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=msg,
            temperature=0.5,
            max_tokens=200,
            response_format=BusinessModel 
    )
    return responses.choices[0].message.parsed
if __name__=="__main__":
    text="about Microsoft"
    business_info=extract(text)
    print(business_info)
    