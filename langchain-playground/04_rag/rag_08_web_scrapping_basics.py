import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from typing import List, Union


# load environment variables from .env files
load_dotenv()

# Define the Directory Containing the Text Files & persistent Directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")


# Read the Context from Website
loader = WebBaseLoader("https://panaverse-dao-peach.vercel.app/")
document = loader.load()

# Split the content into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(document)

# Generate an Embedding Models
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

def create_vector_store(documents, store_name:str):
    persistant_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistant_directory):
        print("Persistent Directory Does Not Exist. Initializing the Vector Store")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persistant_directory
        )
        print(f"Persistent Directory Created: {persistant_directory}")
    else:
        print("Persistent Directory Exists. Loading the Vector Store")

create_vector_store(texts, "chroma_db_web_scrapping_basics")

# Query the Vector Store
def query_vector_store(query: str, store_name: str):
    persistant_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistant_directory):
        print("Persistent Directory Exists. Loading the Vector Store")
        vector_store = Chroma(persist_directory=persistant_directory, embedding_function=embedding)
        retriever  = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.1},
        )
        relevant_docs = retriever.invoke(query)
        print(f"Query through: {store_name}")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}: {doc.page_content}")
            if doc.metadata:
                print(f"Metadata: {doc.metadata.get('source', 'unknown')}")
            print("-" * 50)
    else:
        print("Persistent Directory Does Not Exist. Please check the store name & try again")


query_vector_store("What is the purpose of the Panaverse DAO?", "chroma_db_web_scrapping_basics")