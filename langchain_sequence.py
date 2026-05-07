import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
load_dotenv()

llm=ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct"
    ,temperature=0.7,
)

parser=StrOutputParser()

#chain 1 
idea_validation_prompt=ChatPromptTemplate.from_template(
    """
You are a top business analyst with 20+ years of experience.

You specialize in:
- First principles thinking
- Business model analysis
- Market validation
- Strategic positioning

Analyze the following business idea deeply.

Business Idea:
{business_idea}

Provide:
1. Idea Summary
2. Market Potential
3. Strengths
4. Weaknesses
5. Feasibility
6. Unique Advantage
7. Final Verdict
""")
idea_validation_chain=idea_validation_prompt | llm | parser

#chain 2 painpoint analysis
painpoint_analysis_prompt=ChatPromptTemplate.from_template(
    """
You are an elite business strategist.

Based on the business idea and validation report below,
identify the major pain points, risks, and operational challenges.

Business Idea:
{business_idea}

Idea Validation Report:
{idea_validation_output}

Provide:
1. Customer Pain Points
2. Operational Problems
3. Manufacturing Challenges
4. Scaling Risks
5. Competition Risks
6. Financial Risks
7. Market Adoption Challenges
"""
)
pain_points_chain=painpoint_analysis_prompt | llm | parser

#chain 3 Solutions & Monetization
solution_prompt=ChatPromptTemplate.from_template(
    """
You are a world-class business consultant.

Based on:
1. Business idea
2. Validation report
3. Pain points report

Create strategic solutions and monetization plans.

Business Idea:
{business_idea}

Idea Validation:
{idea_validation_output}

Pain Points:
{pain_points_output}

Provide:
1. Strategic Solutions
2. Revenue Model
3. Luxury Branding Strategy
4. Customer Acquisition Strategy
5. Profit Opportunities
6. Scaling Strategy
7. Long-Term Business Vision
"""
)
solution_chain=solution_prompt | llm | parser

#full pipeline
full_pipeline = (
    {"business_idea": RunnablePassthrough()}
    | RunnablePassthrough.assign(idea_validation_output=idea_validation_chain)
    | RunnablePassthrough.assign(pain_points_output=pain_points_chain)
    | RunnablePassthrough.assign(solution_output=solution_chain)
)

#text
sample_text = """
Building a luxury watch customization company
where customers can customize:
- watch straps
- dial
- crown
- case
- movement aesthetics

We are positioning it as a luxury mechanical watch brand.

We will use premium materials like:
- titanium
- sapphire crystal
- high-grade steel

The focus is handcrafted personalization and premium craftsmanship.
"""

#Invoke the full pipeline
result=full_pipeline.invoke({"business_idea": sample_text})

#results
print("\n========== IDEA VALIDATION ==========\n")
print(result["idea_validation_output"])

print("\n========== PAIN POINTS ==========\n")
print(result["pain_points_output"])

print("\n========== SOLUTIONS ==========\n")
print(result["solution_output"])