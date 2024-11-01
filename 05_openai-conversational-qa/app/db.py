from sqlmodel import SQLModel, create_engine, Field, Session
from starlette.datastructures import Secret
from typing import Annotated
from fastapi import Depends
from starlette.config import Config

try:
    config = Config(".env")
except FileNotFoundError:
    config = Config()

DATABASE_URL = config("DATABASE_URL",cast=Secret)

class Chat(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    chat_history: str


connection_string = str(DATABASE_URL).replace(
    "postgresql", "postgresql+psycopg2"
)

engine = create_engine(connection_string, connect_args={"sslmode": "require"}, pool_recycle=300, echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

DB_SESSION = Annotated[Session, Depends(get_session)]