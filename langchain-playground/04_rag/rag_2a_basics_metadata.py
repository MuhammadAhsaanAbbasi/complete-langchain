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
book_path = os.path.join(current_directory, "books")
persistent_directory = os.path.join(current_directory, "db", "chroma_db_with_metadata")

# Check if the Chroma Vector Store is Already Exist
if not os.path.exists(persistent_directory):
    print("Persistent Directory Does Not Exist. Initializing the Vector Store")
    
    if not os.path.exists(book_path):
        raise ValueError(f"Directory {book_path} does not exist.")
    
    # List all the Files in the Directory
    book_files = [f for f in os.listdir(book_path) if f.endswith(".txt")]

    document = []
    for book_file in book_files:
        file_path = os.path.join(book_path, book_file)
        loader = TextLoader(file_path, encoding="utf-8")
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            document.append(doc)

    # Split the documents into Chunks
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
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
    print("\n---Done---\n")
else:
    print("Persistent Directory Exists. Loading the Vector Store")