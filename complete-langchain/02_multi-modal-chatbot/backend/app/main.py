from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from contextlib import asynccontextmanager
from .chat import multi_modal_chat

@asynccontextmanager
async def life_span(app: FastAPI):
    yield

app = FastAPI(
    title="Multi-Modal Chatbot",
    description="Multi-Modal Chatbot for FAQ and Content Generation",
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
    root_path="/multi_modal-chatbot",
    root_path_in_servers=True,
    docs_url="/docs"
)


# SessionMiddleware must be installed to access request.session 
app.add_middleware(
    SessionMiddleware, secret_key="!secret") 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT"],
    allow_headers=["*"],
)



@app.get("/")
def read_root():
    return {"message": "Hello Multi-Modal Chatbot!"}


@app.get("/chat")
async def chat(prompt:str):
    response = await multi_modal_chat(prompt)
    return response