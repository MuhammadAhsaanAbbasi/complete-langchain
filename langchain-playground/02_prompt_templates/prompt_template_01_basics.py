# Prompt Template Docs:
#   https://python.langchain.com/v0.2/docs/concepts/#prompt-templates
# https://python.langchain.com/v0.2/docs/concepts/#prompt-templates


from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# Part 1: Create a ChatPromptTemplate using an Template String
template = "tell me an details about {topic}"
prompt_template = ChatPromptTemplate.from_template(template)

print("----Prompt From Template----")
prompt = prompt_template.invoke({"topic": "AI"})
print(prompt)


# Part 2: Prompt with Multiple Placeholders
template = """You are an Helpful Assistant
    Human: Tell me an {adjective} details about {topic}
    Assistant: """
prompt_template = ChatPromptTemplate.from_template(template)

print("----Prompt with Multiple Placeholders----")
prompt = prompt_template.invoke({"topic": "AI", "adjective": "new"})
print(prompt)


# Part 3: Prompt with System & Human Message using Tuple
messages = [
    ("system", "you are a comedian who tells jokes about {topic}"),
    ("human", "tell me {joke_count} joke about {topic}")
]

prompt_template = ChatPromptTemplate.from_messages(messages)

print("----Prompt with System & Human Message using Tuple----")
prompt = prompt_template.invoke({"topic": "AI", "joke_count": 2})
print(prompt)


# # Extra Information on Part 3
# messages = [
# messages = [
#     ("system", "you are a comedian who tells jokes about {topic}"),
#     HumanMessage(content="tell me 3 jokes")
# ]

# prompt_template = ChatPromptTemplate.from_messages(messages)

# print("----Extra Information on Part 3----")
# prompt = prompt_template.invoke({"topic": "AI"})
# print(prompt)


# Part 4: Prompt with System & Human Message using Class
# This doesn't work
messages = [
    SystemMessage(content="you are a comedian who tells jokes about {topic}"),
    HumanMessage(content="tell me {joke_count} joke about {topic}")
] # type: ignore

prompt_template = ChatPromptTemplate.from_messages(messages)

print("----Prompt with System & Human Message using Class----")
prompt = prompt_template.invoke({"topic": "AI", "joke_count": 2})
print(prompt)