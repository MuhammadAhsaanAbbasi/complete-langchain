import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# load environment variables from .env files
load_dotenv()

# Define the Directory Containing the Text Files & persistent Directory
current_dir = os.path.dirname(os.path.abspath(__file__))
book_dir  = os.path.join(current_dir, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "db")

if not os.path.exists(book_dir):
    raise FileExistsError(f"Directory {book_dir} does not exist.")

# Read the content from the file
loader = TextLoader(book_dir, encoding="utf-8")
document = loader.load()

# Split the content into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(document)

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Create Chroma Vector Store
def create_vector_store(docs, store_name:str):
    persistant_dir = os.path.join(db_dir, store_name)
    if not os.path.exists(persistant_dir):
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=persistant_dir
        )
        print(f"Persistent Directory Created: {persistant_dir}")
    else:
        print("Persistent Directory Exists. Loading the Vector Store")


print("Creating Vector Store")
create_vector_store(texts, "chroma_db_retriever")


print("Query Vector Store")
def query_vector_store(embedding, query, store_name, search_type, search_kwargs):
    persistant_dir = os.path.join(db_dir, store_name)
    if os.path.exists(persistant_dir):
        print("Persistent Directory Exists. Loading the Vector Store")
        vector_store = Chroma(
            persist_directory=persistant_dir,
            embedding_function=embedding
        )
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        print(f"Query through {search_type}:")
        docs = retriever.invoke(query)
        for i, doc in enumerate(docs, 1):
            print(f"Document {i}: {doc.page_content}")
            if doc.metadata:
                print(f"Metadata: {doc.metadata.get('source', 'unknown')}")
            print("-" * 50)


# Query Vector Store.
query = "How's die juliet?"

# 1. Similarity Search
# This Method retrieves the documents based on the vector similarity.
# It finds the most similar documents to the given query. 
# Use this when you find & retrieved the top k most similar documents.
print("\n---Similarity Search---\n")
query_vector_store(
    embedding=embedding,
    query=query,
    store_name="chroma_db_retriever",
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 2. Max Marginal Relevance (MMR)
# This method balances between selecting documents that are relevant to the query and diverse among themselves.
# 'fetch_k' specifies the number of documents to initially fetch based on similarity.
# 'lambda_mult' controls the diversity of the results: 1 for minimum diversity, 0 for maximum.
# Use this when you want to avoid redundancy and retrieve diverse yet relevant documents.
# Note: Relevance measures how closely documents match the query.
# Note: Diversity ensures that the retrieved documents are not too similar to each other,
#       providing a broader range of information.
print("\n---MMR Search---\n")
query_vector_store(
    embedding=embedding,
    query=query,
    store_name="chroma_db_retriever",
    search_type="mmr",
    search_kwargs={ "k": 3, "fetch_k": 20, "lambda_mult": 0.5}
)

# 3. Similarity Score Threshold
# This method retrieves documents based on a similarity score threshold.
# It selects documents whose similarity scores are above a specified threshold.
# Use this when you want to filter out documents below a certain similarity score.
print("\n---Similarity Score Threshold Search---\n")
query_vector_store(
    embedding=embedding,
    query=query,
    store_name="chroma_db_retriever",
    search_type="similarity_score_threshold",
    search_kwargs={ "k": 3, "score_threshold": 0.2}
)


print("\n---End of Script---\n")