import os
from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools import ArxivQueryRun
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain import hub
from dotenv import load_dotenv
from sqlmodel import SQLModel

load_dotenv()

class Output(SQLModel):
    input: str
    output: str

# Create the Wikipedia API wrapper with the wiki client
api_wrapper = WikipediaAPIWrapper(
        wiki_client=None,
        top_k_results=1,
        doc_content_chars_max=4000
    )

# Create the Wikipedia retrieval tool
wiki_retrieval = WikipediaQueryRun(api_wrapper=api_wrapper)


# Web Base Panaverse Tool
loader = WebBaseLoader("https://panaverse-dao-peach.vercel.app/")
document = loader.load()

recursive_documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(document)

# Initialize Embedding 
openai_embedding = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma.from_documents(recursive_documents,openai_embedding)

retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult" : 0.5},
    )

retriever_tool = create_retriever_tool(
        retriever,
        name="panaverse_search",
        description="Search for Information about Panaverse Dao. For any question about Panaverse Dao, you must be used this tool!",
    )


    # Arxiv Wrapper
arxiv_api_wrapper = ArxivAPIWrapper(
        arxiv_search=None,  # Replace with the appropriate Arxiv search client
        arxiv_exceptions=None,  # Replace with appropriate Arxiv exceptions handling
        top_k_results=1, 
        doc_content_chars_max=4000
    )

arxiv = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

llm = ChatOpenAI(model="gpt-4o-mini")


async def get_llm_response(prompt:str):

    tools = [wiki_retrieval, retriever_tool, arxiv]

    openai_prompt = hub.pull("hwchase17/openai-functions-agent")

    agent = create_openai_tools_agent(llm, tools, openai_prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    result_dict = agent_executor.invoke({"input": prompt})

    result = Output(**result_dict)

    return result.output
