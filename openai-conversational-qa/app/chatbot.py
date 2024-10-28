from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.agent import AgentFinish
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools.render import format_tool_to_openai_function
from langchain.tools import tool
from pydantic import BaseModel, Field
import requests

_ = load_dotenv()

class OpenMeteoInput(BaseModel):
    latitude: float = Field(description="Latitude of the location")
    longitude: float = Field(description="Longitude of the location")

@tool
def get_weather(latitude: float, longitude: float) -> dict:
    """Get weather data for a given latitude and longitude."""
    import requests
    BASE_URL = f"https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude" : latitude,
        "longitude" : longitude,
        "hourly" : "temperature_2m",
        "forecast_days" : 1
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        data = response.json()
    else:
        raise

async def qa_chatbot(query: str):
    return query