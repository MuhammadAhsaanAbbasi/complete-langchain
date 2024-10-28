from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from contextlib import asynccontextmanager
from .chatbot import qa_chatbot

@asynccontextmanager
async def lifespan(app: FastAPI):
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
async def openai_chatbot(query: str):
    response = await qa_chatbot(query=query)

    return response