from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence

# load environment variables from .env files
load_dotenv()

# create a model
model = ChatOpenAI(model="gpt-4o-mini")

# create a prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful automotive assistant to give me an latest about new {topic}?"),
        ("human", "what is the latest & limited model of {car}?"),
    ]
)

# Create individual runnables (steps in the chain)
prompt_parser = RunnableLambda(lambda x: prompt.format_prompt(**x))
model_parser = RunnableLambda(lambda x: model.invoke(x.to_messages()))
output_parser = RunnableLambda(lambda x: x.content)

# Create the RunnableSequence (equivalent to the LCEL Chain)
chain = RunnableSequence(first=prompt_parser, middle=[model_parser], last=output_parser)

# run the chain
response = chain.invoke({"topic": "automobiles", "car": "lamborghini"})

# output
print(response)