# Chat Model Documentation: https://python.langchain.com/v0.2/docs/integerations/chat/
# Openai Chat Model Documentation: https://python.langchain.com/v0.2/docs/integrations/chat/openai

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# load environment variables from .env files
load_dotenv()

# create a chat model
model = ChatOpenAI(model="gpt-4o")

# get a response from the chat model by invoking a model
response = model.invoke("How many states are in Pakistan?")

print(f"Full REsult: {response}")
print(f'content only: {response.content}')