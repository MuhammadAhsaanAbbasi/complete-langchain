from dotenv import load_dotenv
from langchain.agents import (
    AgentExecutor,
    create_react_agent
)
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain import hub
import datetime

# load environment variables from .env files
_ = load_dotenv()


# create a tool for current date & time.
def get_current_time(*args, **kwargs):
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p") # Format time in H:MM AM/PM format

current_time_tool = Tool(
    name="current_time",
    description="Useful for when you need to know the current time. \n\n",
    func=get_current_time
)

tools = [current_time_tool]

# Pull the prompt template from the hub
# ReAct = Reason and Action
# https://smith.langchain.com/hub/hwchase17/react
prompt = hub.pull("hwchase17/react")

# create a chat model
llm = ChatOpenAI(model="gpt-4o")

# create an agent
agent = create_react_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
    stop_sequence=True,
)

# create an agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

# Run the agent with query
result = agent_executor.invoke({"input": "What is the current time?"})

print(result)