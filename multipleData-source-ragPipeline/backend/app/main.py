from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import ORJSONResponse
from .chatbot import get_llm_response
from .settings import LANGCHAIN_API_KEY, USER_AGENT
import os

@asynccontextmanager
async def life_span(app:FastAPI):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = str(LANGCHAIN_API_KEY)
    os.environ["USER_AGENT"] = USER_AGENT
    yield

app = FastAPI(
    title="Multiple Data Sources RAG Pipeline ChatBot",
    description="Multiple Data Sources RAG Pipeline ChatBot for chat with Multiple Sources Like Wikipedia, arxiv ",
    version="1.0.0",
    terms_of_service="https://caxgpt.vercel.app/terms/",
    lifespan=life_span,
    contact={
        "name": "Muhammad Ahsaan Abbasi",
        "phone": "+92 349-204-7381",
        "email": "mahsaanabbasi@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
    },
    root_path="/multiple_data_source-chatbot",
    root_path_in_servers=True,
    docs_url="/docs"
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
    return {"Hello": "Multiple Data Sources RAG Pipeline ChatBot"}


@app.get("/chat")
async def chat(prompt: str):
    response = await get_llm_response(prompt)
    return ORJSONResponse({"response": response})