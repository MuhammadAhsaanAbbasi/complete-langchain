from dotenv import load_dotenv
from langchain.agents import (
    AgentExecutor,
    create_structured_chat_agent
)
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain import hub
from wikipedia import search, summary # type: ignore
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

def search_wikipedia(query):
    try:
        results = search(query)
        if results:
            page = summary(results[0], sentences=3)
            return page
        else:
            return "No results found for the given query."
    except Exception as e:
        return f"An error occurred: {str(e)}"

wikipedia_tool = Tool(
    name="wikipedia",
    description="Useful for when you need to know about any topic. \n\n",
    func=search_wikipedia
)

tools = [current_time_tool, wikipedia_tool]

# Pull the prompt template from the hub
# ReAct = Reason and Action
# https://smith.langchain.com/hub/hwchase17/react
prompt = hub.pull("hwchase17/structured-chat-agent")

# create a memory object
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output",
)


# create a chat model
llm = ChatOpenAI(model="gpt-4o")

agent = create_structured_chat_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# create an agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

# Initial Messages
Initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: current_time and wikipedia."
memory.chat_memory.add_message(SystemMessage(content=Initial_message))

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    print(f"You: {user_input}")
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    response = agent_executor.invoke({"input": user_input})
    print("Agent:", response["output"])

    memory.chat_memory.add_message(AIMessage(content=response["output"]))

