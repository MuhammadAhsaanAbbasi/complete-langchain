import os
from dotenv import load_dotenv
from langchain.tools import Tool, StructuredTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain import hub

# load environment variables from .env files
_ = load_dotenv()


# define a function that's greet to the user.
def greet(name: str) -> str:
    return f"Hello {name}!"


# define a function that's reverse a string.
def reverse_string(string: str) -> str:
    return string[::-1]


# define a function that's concatenate two strings.
def concatenate_strings(a: str, b: str) -> str:
    return a + b


class ConcatenateStringsArgs(BaseModel):
    a: str = Field(description="First string")
    b: str = Field(description="Second string")


greet_tool = Tool(
    name="greet",
    description="Greet the user",
    func=greet,
)

reverse_string_tool = Tool(
    name="reverse_string",
    description="Reverse a string",
    func=reverse_string,
)

concatenate_strings_tool = StructuredTool.from_function(
    func=concatenate_strings,  # Function to execute
    name="ConcatenateStrings",  # Name of the tool
    description="Concatenates two strings.",  # Description of the tool
    args_schema=ConcatenateStringsArgs,    # Schema defining the tool's input arguments
)

# tools
tools = [greet_tool, reverse_string_tool, concatenate_strings_tool]

# generate an llm model
llm = ChatOpenAI(model="gpt-4o-mini")

# Pull the prompt template from the hub
prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_tool_calling_agent(llm, tools, prompt)

# create an agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    response = agent_executor.invoke({"input": user_input})
    print("Agent:", response["output"])