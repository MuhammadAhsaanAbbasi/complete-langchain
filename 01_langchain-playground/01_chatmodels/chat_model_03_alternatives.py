from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# load environment variables from .env files
load_dotenv()

messages = [
    SystemMessage(content="You are a helpful automotive assistant to give me an latest about new automobiles?"),
    HumanMessage(content="what is the latest & limited model of lamborghini?"),
]

# create a Openai chat model
model = ChatOpenAI(model="gpt-4o")

# invoke the model
response = model.invoke(messages)
print(f"Full OpenA.I REsult: {response.content}")

# Create an Gemini  chat model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# invoke the model
response = model.invoke(messages)
print(f"Full Gemini REsult: {response.content}")

# Create a HuggingFaceEndpoint chat model
model = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    max_new_tokens=100,
    do_sample=True,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN"),
)

# invoke the model
response = model.invoke(messages)
print(f"Full HuggingFace REsult: {response}")