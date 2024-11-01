from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.tools import tool
import wikipedia # type: ignore
from dotenv import load_dotenv
from sqlmodel import select, Session
from typing import List
from typing import Dict, Union
from fastapi import HTTPException, status
from .db import DB_SESSION, Chat
import json
import datetime
import requests as rq

_ = load_dotenv()

class OpenMeteoInput(BaseModel):
    latitude: float = Field(description="Latitude of the location")
    longitude: float = Field(description="Longitude of the location")

@tool(args_schema=OpenMeteoInput)
def get_weather(latitude: float, longitude: float):
    """Get weather data for a given latitude and longitude."""
    BASE_URL = f"https://api.open-meteo.com/v1/forecast"
    params: Dict[str, Union[str, int, float]] = {
        "latitude" : latitude,
        "longitude" : longitude,
        "hourly" : "temperature_2m",
        "forecast_days" : 1
    }

    response = rq.get(BASE_URL, params=params)

    if response.status_code == 200:
        data = response.json()
    else:
        raise Exception(f"API Request Failed: {response.status_code}, {response.text}")

    current_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(t.replace("Z", "+00:00")) for t in data["hourly"]["time"]]
    temperature_lists = data["hourly"]["temperature_2m"]

    closet_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_time))

    temperature = temperature_lists[closet_time_index]

    return f"The Current temperature at {latitude}, {longitude} is {temperature}°C"

@tool
def wikipedia_search(query:str):
    """Run Wikipedia Search & get pages summaries."""
    page_titles = wikipedia.search(query)
    summary_pages = []
    for title in page_titles[:1]:
        try:
            page = wikipedia.page(title=title, auto_suggest=False)
            summary_pages.append(f"Page: {title}\nSummary: {page.summary}")
        except Exception as e:
            raise e

    return "\n\n".join(summary_pages)


async def qa_chatbot(query: str, chat_id: int | None, session: DB_SESSION):
    tools = [get_weather, wikipedia_search]
    functions = [format_tool_to_openai_function(t) for t in tools]

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind(functions=functions)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent_chain = RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"]),
    ) | prompt | model | OpenAIFunctionsAgentOutputParser()

    if chat_id:
        db_chat = session.exec(select(Chat).where(Chat.id == chat_id)).first()
        if not db_chat:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Chat not found")
        
        chat = db_chat.chat_history
        # Converting JSON string to list of dictionaries
        try:
            chat_history_data = json.loads(chat)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON data provided for chat history")

        # Converting list of dictionaries to list of message objects
        chat_history = [
            HumanMessage(**msg) if msg["type"] == "human" else AIMessage(**msg)
            for msg in chat_history_data
        ]
        
        chat_memory = ChatMessageHistory(messages=chat_history)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, chat_memory=chat_memory)
    else:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        db_chat = None

    agent_executer = AgentExecutor(agent=agent_chain, tools=tools, memory=memory, verbose=True)

    response = await agent_executer.ainvoke({"input": query})

    # Serialize the updated chat history
    new_chat_history = json.dumps([msg.model_dump() for msg in memory.chat_memory.messages])

    if not db_chat:
        db_chat = Chat(chat_history=new_chat_history)
        session.add(db_chat) 
    else:
        db_chat.chat_history = new_chat_history

    session.commit()
    session.refresh(db_chat)

    return {"output": response["output"], "chat_history": db_chat.chat_history}

