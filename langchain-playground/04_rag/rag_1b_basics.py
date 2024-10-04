import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# load environment variables from .env files
load_dotenv()

# Define the Directory Containing the Text file & persistent Directory
current_directory = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_directory, "db", "chroma_db")

# Define the Embedding Model
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Load the Existing Vector Store with the Embedded Function
db = Chroma(persist_directory=persistent_directory, embedding_function=embedding)

# Query the Existing Vector Store
query = "Who is Odysseus' wife?"

# Retrieve the relevant Documents based on User Querys
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.4},
)

relevant_docs = retriever.invoke(query)

#  Display the Relevant Documents Results with MetaData
print('\n---Relevant Documents---\n')
for i, docs in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{docs.page_content}\n")
    if docs.metadata:
        print(f'Source: {docs.metadata.get('source', 'unknown')}\n')