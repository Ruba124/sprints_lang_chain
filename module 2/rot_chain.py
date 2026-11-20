from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from typing import Literal
import os
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

# 2. Initialize the model via OpenRouter
llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

# 3. Define Pydantic models matching your school/education domain
class RouterInput(BaseModel):
    role_type: Literal["student", "teacher"] = Field(
        ..., description="Type of person in the school ('student' or 'teacher')"
    )

class TeachingPlan(BaseModel):
    tech1: str = Field(..., description="First practical technique for helping in teaching")
    tech2: str = Field(..., description="Second practical technique for helping in teaching")
    search: str = Field(..., description="Recommended search query or resource for the best teaching technique")

class StudyingPlan(BaseModel):
    tech1: str = Field(..., description="First practical technique for helping in studying")
    tech2: str = Field(..., description="Second practical technique for helping in studying")
    search: str = Field(..., description="Recommended search query or resource for the best studying technique")

# 4. Create JsonOutputParsers for each schema
router_input_parser = JsonOutputParser(pydantic_object=RouterInput)
teaching_parser = JsonOutputParser(pydantic_object=TeachingPlan)
studying_parser = JsonOutputParser(pydantic_object=StudyingPlan)

# 5. Prompts aligned to your Pydantic models and {helping} input variable
router_prompt = ChatPromptTemplate.from_template(
    "A user in a school setting needs assistance. Determine whether the user is a 'student' or a 'teacher' "
    "based on their request: {helping}. Use the format instructions.\n\n{format_instructions}"
).partial(format_instructions=router_input_parser.get_format_instructions())

teaching_prompt = ChatPromptTemplate.from_template(
    "A teacher needs help with the following request: {helping}. "
    "Provide two effective teaching techniques and a helpful search query/resource for mastering this topic. "
    "Use the format instructions.\n\n{format_instructions}"
).partial(format_instructions=teaching_parser.get_format_instructions())

studying_prompt = ChatPromptTemplate.from_template(
    "A student needs help with the following request: {helping}. "
    "Provide two effective studying techniques and a helpful search query/resource for mastering this topic. "
    "Use the format instructions.\n\n{format_instructions}"
).partial(format_instructions=studying_parser.get_format_instructions())

# 6. Build the Router Chain
router_chain = router_prompt | llm | router_input_parser

# 7. Handler functions for each branch
def handle_teaching(helping: str) -> dict:
    teaching_chain = teaching_prompt | llm | teaching_parser
    return teaching_chain.invoke({"helping": helping})

def handle_studying(helping: str) -> dict:
    studying_chain = studying_prompt | llm | studying_parser
    return studying_chain.invoke({"helping": helping})

# 8. Routing logic function
def route(output):
    role_type = output["decision"].get("role_type", "")
    helping = output.get("helping", "")
    
    if role_type == "teacher":
        return handle_teaching(helping)
    elif role_type == "student":
        return handle_studying(helping)
    else:
        return "Sorry, I can only assist students or teachers."

# 9. Assemble and run the router app
router_app = {
    "decision": router_chain,
    "helping": lambda x: x["helping"]
} | RunnableLambda(route)

# Test with teacher input
print(router_app.invoke({"helping": "i want to teach math"}))

# Test with student input
# print(router_app.invoke({"helping": "i need help preparing for my physics exam"}))