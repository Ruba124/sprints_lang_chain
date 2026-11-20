import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Load the OPENAI_API_KEY from your .env file
load_dotenv()

# 2. Initialize the model (automatically uses OPENAI_API_KEY from environment)
llm = ChatOpenAI(model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1")


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

chain = prompt | llm

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input"
)
s = "hello"
while (s != "exit"):
    s = input("message: ")
    response = chatbot.invoke(
        {"input": s},
        config={"configurable": {"session_id": "user1"}}
    )

    print(response.content) 