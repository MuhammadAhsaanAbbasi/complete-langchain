import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# load environment variables from .env files
load_dotenv()

# Define the Directory containing Text & persistent Directory
current_directory = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_directory, "db", "chroma_db_with_metadata")

# Generate an Embedding Model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# Create Chroma Vector Store
db = Chroma(
    embedding_function=embeddings,
    persist_directory=persistent_directory,
)

# Query the Existing Vector Store
query = "How did Juliet die?"

# Retrieve the relevant Documents based on User Querys
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.3},
)

relevant_docs = retriever.invoke(query)

for i, docs in enumerate(relevant_docs, 1):
    print(f"Document {i}: {docs.page_content}\n")
    if(docs.metadata):
        print(f"Metadata: {docs.metadata["source"]}\n")
    print("-" * 50)