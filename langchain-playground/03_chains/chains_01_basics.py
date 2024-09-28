# Chain Docs
# https://python.langchain.com/v0.1/docs/modules/chains/

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# load environment variables from .env files
load_dotenv()

# create a model
model = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    do_sample=True,
    max_new_tokens=100,
)

# create a prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful automotive assistant to give me an latest about new {topic}?"),
        ("human", "what is the latest & limited model of {car}?"),
    ]
)

# create a chain
chain = prompt | model | StrOutputParser()

# Inovke the chain
response = chain.invoke({"topic": "cars", "car": "lamborghini"})
print(response)