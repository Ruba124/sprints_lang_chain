import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. Load the OPENAI_API_KEY from your .env file
load_dotenv()

# 2. Initialize the model (automatically uses OPENAI_API_KEY from environment)
llm = ChatOpenAI(model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1")

# 3. Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant."),
    ("user", "I want to visit {destination} for {days} days. Suggest me best offers.")
])

# 4. Build the chain using LangChain Expression Language (LCEL)
chain = prompt | llm

# 5. Get user inputs
destination = input("Enter your preferred destination: ")
days = input("Enter number of days for your trip: ")

# 6. Invoke the chain and print the response
response = chain.invoke({
    "destination": destination,
    "days": days
})

print("\n--- Travel Suggestions ---")
print(response.content)