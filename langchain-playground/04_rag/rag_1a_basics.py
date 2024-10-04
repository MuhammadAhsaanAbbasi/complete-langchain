import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# load environment variables from .env files
load_dotenv()

# Define the Directory Containing the Text file & persistent Directory
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, "books", "odyssey.txt")
persistent_directory = os.path.join(current_directory, "db", "chroma_db")

# Check if the Chroma Vector Store is Already Exist
if not os.path.exists(persistent_directory):
    print("Persistent Directory Does Not Exist. Initializing the Vector Store")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} doesn't Exist. Please check the file path & try again"
        )
    
    # Read the context from the file
    loader = TextLoader(file_path, encoding="utf-8")
    document = loader.load()

    # Split the documents into Chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(document)

    # Display Information about the Split Documents.
    print("\n---Documents Chunks Information---\n")
    print(f"Number of Documents Chunks {len(docs)}")
    print(f'Sample Chunk:\n {docs[1].page_content}\n')

    # Create Embeddings
    print("\n---Creating Embeddings---\n")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",)
    
    # Create Chroma Vector Store
    print("\n---Creating Vector Store---\n")
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("\n---Vector Store Created Successfully---\n")
else:
    print("\n---Persistent Directory Already Exist---\n")

