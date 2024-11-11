import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from dotenv import load_dotenv

# load environment variables from .env files
load_dotenv()

# Define the Directory Containing the Text Files & persistent Directory
current_dir = os.path.dirname(os.path.abspath(__file__))
book_dir  = os.path.join(current_dir, "books", "langchain_demo.txt")
db_dir = os.path.join(current_dir, "db")

# if not os.path.exists(book_dir):
#     raise FileExistsError(f"Directory {book_dir} does not exist.")

# # Read the content from the file
# loader = TextLoader(book_dir, encoding="utf-8")
# document = loader.load()

# # Split the content into smaller chunks
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# texts = text_splitter.split_documents(document)

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# def create_vector_store(docs, store_name):
#     persistant_directory = os.path.join(db_dir, store_name)
#     if not os.path.exists(persistant_directory):
#         print("Persistent Directory Does not Exist. Creating a new one.")
#         vector_store = Chroma.from_documents(
#             documents=docs,
#             collection_name=store_name,
#             embedding=embedding,
#             persist_directory=persistant_directory,
#         )
#     else:
#         print("Persistent Directory Exists. Loading the Vector Store")


# print("Creating Vector Store")
# create_vector_store(texts, "chroma_db_langchain_demo")


def query_vector_store(query: str, store_name: str):
    persistant_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistant_directory):
        print("Persistent Directory Exists. Loading the Vector Store")
        vector_store = Chroma(persist_directory=persistant_directory, embedding_function=embedding)
        retriever  = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 1},
        )
        relevant_docs = retriever.invoke(query)
        print(f"Query through {store_name}:")
        print(relevant_docs)
        return relevant_docs

# Define the user's question
query = "How can I learn more about LangChain?"


print("Querying Vector Store")
relevant_docs = query_vector_store(query, "chroma_db_with_metadata")

# Combine the query and the relevant document contents
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)