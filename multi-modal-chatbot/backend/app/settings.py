from starlette.config import Config
from starlette.datastructures import Secret

from starlette.config import Config
from starlette.datastructures import Secret

try:
    config = Config(".env")
except FileNotFoundError:
    config = Config()

HUGGINGFACE_TOKEN = config("HUGGINGFACE_TOKEN", cast=str)