from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from contextlib import asynccontextmanager
from .chatbot import qa_chatbot
from .db import create_db_and_tables, DB_SESSION

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Creating tables..")
    create_db_and_tables()
    yield

app = FastAPI(
    title="LangChain OpenAI function Calling Conversational QA",
    description="This Chatbot is Design for Conversational QA using LangChain withe help of OpenAI function Calling",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(SessionMiddleware, secret_key="secret")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Message": "LangChain OpenAI function Calling Conversational QA"}

@app.get("/chatbot")
async def openai_chatbot(query: str, session: DB_SESSION, chat_id: int | None = None):
    response = await qa_chatbot(query=query, chat_id=chat_id, session=session)
    return response