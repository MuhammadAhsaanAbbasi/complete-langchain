# Note: Must Be Run this file on Google Colab because it's file have an package that consume so much 
# space so if you have low configuration system so run this file on google colab

import os
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TokenTextSplitter,
    TextSplitter,
)
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env files
load_dotenv()

current_directory = os.path.dirname(os.path.abspath(__file__))
book_path = os.path.join(current_directory, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_directory, "db")

if not os.path.exists(book_path):
    raise FileExistsError(f"Directory {book_path} does not exist.")

# Read the content from the file
loader = TextLoader(book_path, encoding="utf-8")
document = loader.load()

# Generate an Embedding Model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Create a Vector Store
def vector_store(docs, store_name: str):
    persistant_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistant_directory):
        print("Persistent Directory Does Not Exist. Initializing the Vector Store")
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persistant_directory,
        )
    else:
        print("Persistent Directory Exists. Loading the Vector Store")

# Character Text Splitter
# Split the text into chunks based on the specified number of characters.
# Useful for Consistent chunk size regardless of Content Structure.
print(f'\n---Character Text Splitter---\n')
char_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_text_docs = char_text_splitter.split_documents(document)
vector_store(char_text_docs, "chroma_db_character")

# Sentence Text Splitter
# Split the text into chunks based on the specified number of sentences.
# Ideal for Maintaining Semantic Coherence with Chunks.
print(f'\n---Sentence Text Splitter---\n')
sentence_text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sentence_text_docs = sentence_text_splitter.split_documents(document)
vector_store(sentence_text_docs, "chroma_db_sentence")

# Token Text Splitter
# Split the text into chunks based on the Tokens (Words or Subwords) using Tokenizer like GPT-2.
# Useful for Transformers Model with Strict Token Limits.
print(f'\n---Token Text Splitter---\n')
token_text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=100)
token_text_docs = token_text_splitter.split_documents(document)
vector_store(token_text_docs, "chroma_db_token")

# Recursive Character Text Splitter
# Recursively split the text into chunks at natural boundaries (sentences, paragraphs) within a character limit.
# Useful for Handling Long Documents with Overlapping Chunks.
print(f'\n---Recursive Character Text Splitter---\n')
recursive_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
recursive_text_docs = recursive_text_splitter.split_documents(document)
vector_store(recursive_text_docs, "chroma_db_recursive_character")

# Custom Text-Splitting
# Allow Creating Custom Splitting Logic based on Specified Requirements.
# Useful for Documents with Unique Structure that Standard splitters can't handle.
print(f'\n---Custom Text Splitter---\n')

class CustomSplitter(TextSplitter):
    def split_text(self, text):
        # Custom Logic for Text Splitting
        return text.split("\n\n")

custom_splitter = CustomSplitter()
custom_docs = custom_splitter.split_documents(document)
vector_store(custom_docs, "chroma_db_custom")

print(f'Query in Vector Store')

def query_vector_store(query: str, store_name: str):
    persistant_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistant_directory):
        print("Persistent Directory Exists. Loading the Vector Store")
        vector_store = Chroma(persist_directory=persistant_directory, embedding_function=embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.2},
        )
        docs = retriever.invoke(query)
        print(f'Query in {store_name}:')
        for i, doc in enumerate(docs, 1):
            print(f"Document {i}: {doc.page_content}")
            if doc.metadata:
                print(f"Metadata: {doc.metadata.get('source', 'unknown')}")
            print("-" * 50)
    else:
        print("Persistent Directory Does Not Exist. Please check the store name & try again")

# Query the Vector Store
query = "How did Juliet die?"

query_vector_store(query, "chroma_db_character")
query_vector_store(query, "chroma_db_sentence")
query_vector_store(query, "chroma_db_token")
query_vector_store(query, "chroma_db_recursive_character")
query_vector_store(query, "chroma_db_custom")