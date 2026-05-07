import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
load_dotenv()

llm=ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct"
    ,temperature=0.7
)
parser=StrOutputParser()
# 2 Chains one Bussiness idea optimistic view and another non optimistic

optimist_prompt = ChatPromptTemplate.from_template(
    """
    You world class business analyst with 20+ years of experience.
    
    Give optimistic feedback for the following business idea.
    
    Business Idea:
    {business_idea}
    
    Provide:
    1. Business idea summary
    2. Market potential
    3. Strengths
    4. Feasibility 
    5. Unique advantage
    6. Final verdict

    output should be in the form of list

    """
)
# Chain 2 -> Critic Prompt
critic_prompt = ChatPromptTemplate.from_template(
    """
    You world class business analyst with 20+ years of experience.
    
    Give pessimistic feedback for the following business idea.
    
    Business Idea:
    {business_idea}
    
    Provide:
    1. Weaknesses
    2. major risks in this business
    3. what can make this business unsuccessful 
    4. competetors, problems why this idea does not work when this does not work etc.
    
    output should be in the form of list

    """
)

# Chain3 Final decesion maker
decison_maker_prompt = ChatPromptTemplate.from_template(
    """
    You world class business analyst with 20+ years of experience.
    Based on the optimistic and pessimistic feedback below, make a final decision on whether to proceed with the business idea.
    
    Business Idea:
    {business_idea}
    
    Optimistic Feedback:
    {optimistic_feedback}
    
    Pessimistic Feedback:
    {pessimistic_feedback}
    
    Provide:
    1. Final Decision
    2. Justification
    3. Recommendations
    """
)
optimistic_chain = optimist_prompt | llm | parser
critic_chain = critic_prompt | llm | parser
decison_maker_chain = decison_maker_prompt | llm | StrOutputParser()

parallel_analysis = RunnableParallel({
    "optimistic_feedback": optimistic_chain,
    "pessimistic_feedback": critic_chain,
})
# Final Analysis -> Full Pipeline
full_pipeline = (
    {
        "business_idea": RunnablePassthrough(),
        "parallel_analysis": parallel_analysis
    }
    | RunnablePassthrough.assign(
        optimistic_feedback=lambda x: x["parallel_analysis"]["optimistic_feedback"],
        pessimistic_feedback=lambda x: x["parallel_analysis"]["pessimistic_feedback"]
    )
    | RunnablePassthrough.assign(
        final_decision=decison_maker_chain
    )
)

# Idea 
idea = """
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
res = full_pipeline.invoke(idea)
print("\n========== OPTIMISTIC VIEW ==========\n")
print(res["optimistic_feedback"])

print("\n========== CRITIC VIEW ==========\n")
print(res["pessimistic_feedback"])

print("\n========== FINAL DECISION ==========\n")
print(res["final_decision"])