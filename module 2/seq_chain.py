import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import LLMChain, SequentialChain
from pydantic import BaseModel, Field

# 1. Load the OPENAI_API_KEY from your .env file
load_dotenv()

# 2. Initialize the model (automatically uses OPENAI_API_KEY from environment)
llm = ChatOpenAI(model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1")

class CitySuggestion(BaseModel):
    city: str

class Outline(BaseModel):
    day1: str = Field(description="Description of activities for day 1")
    day2: str = Field(description="Description of activities for day 2")
    day3: str = Field(description="Description of activities for day 3")

city_parser = PydanticOutputParser(pydantic_object=CitySuggestion)
outline_parser = PydanticOutputParser(pydantic_object=Outline)

# 1. Suggest a city for the travel goal
city_prompt = ChatPromptTemplate.from_template(
    "Recommend a city for a traveler seeking: {travel_goal}.\n\n{format_instructions}"
).partial(format_instructions=city_parser.get_format_instructions())

city_chain = LLMChain(
    llm=llm,
    prompt=city_prompt,
    output_key="city"  # This is generated here and passed on
)

# 2. Build an itinerary for the chosen city
outline_prompt = ChatPromptTemplate.from_template(
    "Give a short 3-day outline for visiting {city} for this travel goal: {travel_goal}.\n\n{format_instructions}"
).partial(format_instructions=outline_parser.get_format_instructions())

outline_chain = LLMChain(
    llm=llm,
    prompt=outline_prompt,
    output_key="outline"
)

# Create the SequentialChain
chain = SequentialChain(
    chains=[city_chain, outline_chain],
    input_variables=["travel_goal"],        # Only need this at the start
    output_variables=["city", "outline"],   # All final outputs
    verbose=True
)

# Example run!
result = chain.invoke({"travel_goal": "foootball"})
print(result["city"])
print(result["outline"] )