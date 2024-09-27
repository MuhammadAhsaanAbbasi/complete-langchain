from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# load environment variables from .env files
load_dotenv()

chat = ChatOpenAI(model="gpt-4o")

# Part 1: Create a ChatPromptTemplate using an Template String
prompt_template = ChatPromptTemplate.from_template("Tell me a joke about {topic}")

print('----Prompt From Template----')
prompt = prompt_template.invoke({"topic": "AI"})
result = chat.invoke(prompt)
print(result.content)


# Part 2: Prompt with Multiple Placeholders
prompt_template = ChatPromptTemplate.from_template("Tell me an {adjective} joke about {topic}")

print('----Prompt with Multiple Placeholders----')
prompt = prompt_template.invoke({"topic": "AI", "adjective": "new"})
result = chat.invoke(prompt)
print(result.content)


# Part 3: Prompt with System & Human Message using Tuple
messages = [
    ("system", "you are a comedian who tells jokes about {topic}"),
    ("human", "tell me {joke_count} joke about {topic}")
]

prompt_template = ChatPromptTemplate.from_messages(messages)

print('----Prompt with System & Human Message using Tuple----')
prompt = prompt_template.invoke({"topic": "AI", "joke_count": 2})
result = chat.invoke(prompt)
print(result.content)