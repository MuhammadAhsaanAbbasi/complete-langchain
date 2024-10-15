import os
from dotenv import load_dotenv
from langchain.tools import Tool, StructuredTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient # type: ignore
from langchain import hub 
from typing import Type

# load environment variables from .env files
_ = load_dotenv()

class SimpleSearchInput(BaseModel):
    query: str = Field(description="should be a simple search query")


class MultiplyNumbersInput(BaseModel):
    first_number: int = Field(description="should be a number")
    second_number: int = Field(description="should be a number")

class SimpleSearchTool(BaseTool):
    name: str = "simple_search"
    description: str = "Useful for when you need to answer questions about current events"
    args_schema: Type[BaseModel] = SimpleSearchInput

    def _run(self, query: str):
        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        result = client.search(query=query, search_depth="advanced", include_images=False)
        return f'Search results for {query}: \n\n {result} \n'


class MultiplyNumbersTool(BaseTool):
    name: str = "multiply_numbers"
    description: str = "Useful for when you need to multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyNumbersInput

    def _run(self, first_number: int, second_number: int):
        result = first_number * second_number
        return f'The result of multiplying {first_number} and {second_number} is {result}.'


# generate a list of tools
tools = [
    SimpleSearchTool(),
    MultiplyNumbersTool(),
]

# Instantiate the LLM
llm = ChatOpenAI(model="gpt-4o")

# Pull the prompt template from the hub
prompt = hub.pull("hwchase17/openai-tools-agent")

# Instantiate the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

while True:
    user_input = input("Enter your query (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    response = agent_executor.invoke({"input": user_input})
    print(response["output"])