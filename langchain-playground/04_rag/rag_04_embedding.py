# Note: Must Be Run this file on Google Colab because it's file have an package that consume so much 
# space so if you have low configuration system so run this file on google colab

import os
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env files
load_dotenv()

# Define the Directory Containing the TExt file & persistent Directory
current_directory = os.path.dirname(os.path.abspath(__file__))
book_path = os.path.join(current_directory, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_directory, "db")

if not os.path.exists(book_path):
    raise FileExistsError(f"Directory {book_path} does not exist.")

# Read the content from the file
loader = TextLoader(book_path, encoding="utf-8")
document = loader.load()

# Character Text Splitting
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(document)

# Create Chroma Vector Store
def create_vector_store(documents, embeddings, dir):
    persistant_directory = os.path.join(db_dir, dir)
    if not os.path.exists(persistant_directory):
        print("Persistent Directory Does Not Exist. Initializing the Vector Store")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persistant_directory,
        )
    else:
        print("Persistent Directory Exists. Loading the Vector Store")


# Generate a OpenAI Embeddings
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
)
print('\n---Creating OpenAI Embeddings---\n')
create_vector_store(docs, openai_embeddings, "chroma_openai_embeddings")

# Generate a HuggingFace Embeddings
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
)
print('\n---Creating HuggingFace Embeddings---\n')
create_vector_store(docs, huggingface_embeddings, "chroma_huggingface_embeddings")


def query_vector_store(embedding, query: str, store_name: str):
    persistant_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistant_directory):
        print('Persistent Directory Exists. Loading the Vector Store')
        vector_store = Chroma(persist_directory=persistant_directory, embedding_function=embedding)
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.2},
        )
        docs = retriever.invoke(query)
        print(f'Query through {store_name}:')
        for i, doc in enumerate(docs, 1):
            print(f"Document {i}: {doc.page_content}")
            if doc.metadata:
                print(f"Metadata: {doc.metadata.get('source', 'unknown')}")
            print("-" * 50)

# Query the Vector Stores
query = "How's die juliet?"

query_vector_store(openai_embeddings, query, "chroma_openai_embeddings")
query_vector_store(huggingface_embeddings, query, "chroma_huggingface_embeddings")