import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# Initialize LLM with OpenRouter free model
llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# 1. Define the shape of the model's answer
class playerReport(BaseModel):
    h_name: str = Field(description="player name")
    current_team: str = Field(description="Current team")
    position: str = Field(description="player position")

parser = PydanticOutputParser(pydantic_object=playerReport)


# 2. Build a prompt that instructs the LLM to follow that schema
prompt1 = ChatPromptTemplate.from_template(
    "Give me information about the player {h_name} in JSON.\n{format_instructions}"
)
prompt2= prompt1.partial(format_instructions=parser.get_format_instructions())

# 3. Run & parse
chain = prompt2 | llm | parser

h_name = input("Enter the player's name: ")
report: playerReport = chain.invoke({"h_name": h_name})

# 4. Print the result
print(report)
# Because it's a validated Pydantic object, you can access fields directly:
print(f"Player: {report.h_name} | Team: {report.current_team} | Position: {report.position}")