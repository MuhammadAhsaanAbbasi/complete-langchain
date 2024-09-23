from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# load environment variables from .env files
load_dotenv()

# Create a chat model
model = ChatOpenAI(model="gpt-4o")

# System message
#    Message for Primary AI Behavior, usually passed in as the first of a sequence of Input Messages.
# Human Message
#    Message for Human to AI Model
# AI Message
#    Message for Primary AI Behavior, usually passed in as the third of a sequence of Input Messages.

messages = [
    SystemMessage(content="You are a helpful automotive assistant to give me an latest about new automobiles?"),
    HumanMessage(content="what is the latest & limited model of lamborghini?"),
]

# Invoke the model
response = model.invoke(messages)
print(f"Full A.I REsult: {response.content}")


# AI message
#    Message from an AI Model


aiMessage = AIMessage(content=response.content)
next_question = HumanMessage(content="What about HURACÁN EVO SPYDER")

messages.append(aiMessage)
messages.append(next_question)

# Invoke the model with AI Messages
result = model.invoke(messages)
print(f"Full A.I REsult: {result.content}")