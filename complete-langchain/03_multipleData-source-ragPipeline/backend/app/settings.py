from starlette.config import Config
from starlette.datastructures import Secret

try:
    config = Config(".env")
except FileNotFoundError:
    config = Config()

LANGCHAIN_API_KEY = config.get('LANGCHAIN_API_KEY', cast=Secret)
USER_AGENT = config.get('USER_AGENT', cast=str)