from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser

# load environment variables from .env files
load_dotenv()


# create a model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# create a prompt for chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful automotive assistant to give me an latest about new {topic}?"),
        ("human", "what is the latest & limited model of {car}?"),
    ]
)

# # Define additional processing steps using RunnableLambda
uppercase_output = RunnableLambda(lambda x: x.upper())
word_count = RunnableLambda(lambda x: f"Words:{len(x.split())}\n {x}")


# Create a Combined Chain using Langchain Expression Language (LCEL)
chain = prompt | model | StrOutputParser()  | uppercase_output | word_count

# run the chain
response = chain.invoke({"topic": "automobiles", "car": "lamborghini"})
# Output
print(response)