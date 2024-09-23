from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

# load environment variables from .env files
load_dotenv()

# chat messages
chat_messages = []

# create a chat model
model = ChatOpenAI(model="gpt-4o")
system_message = SystemMessage(content="You are a helpful automotive assistant to give me an latest about new automobiles?")

chat_messages.append(system_message)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    chat_messages.append(HumanMessage(content=user_input))
    response = model.invoke(chat_messages)
    chat_messages.append(AIMessage(content=response.content))
    print(f"Assistant: {response.content}")

print('----chat-history--------')
print(chat_messages)