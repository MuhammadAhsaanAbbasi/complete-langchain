import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_chroma import Chroma
from dotenv import load_dotenv
from typing import List, Union
# load environment variables from .env files
load_dotenv()

# Define the Directory Containing the Text Files & persistent Directory
current_dir = os.path.dirname(os.path.abspath(__file__))
book_dir  = os.path.join(current_dir, "books", "langchain_demo.txt")
db_dir = os.path.join(current_dir, "db")

if not os.path.exists(book_dir):
    raise FileExistsError(f"Directory {book_dir} does not exist.")

# Generate an Embedding Model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# query vector store
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

if os.path.exists(persistent_directory):
    print("Persistent Directory Exists. Loading the Vector Store")

vector_store = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

# Generate a Chat Model
llm = ChatOpenAI(model="gpt-4o-mini")

# Contextualize Question Prompt
# The System Prompt helps AI to Understand that it should reformulate the question
# based on the chat history to make it StandAlone Question.
contextualize_q_system_prompt = (
    """ Given a chat history & latest user question
    which might reference context in the chat history,
    formulate a standalone question that can be understood
    without the chat history. Make sure to not repeat
    question, just reformulate it if needed & otherwise return it as is.
    """
)

# Create a prompt Template for Contextualize Question
contextualize_qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

# create history aware Retrieval
retriever_chain = create_history_aware_retriever(llm, retriever, contextualize_qa_prompt )

# Original Question Prompt.
qa_system_prompt = (
    """ You're assistant for Question Answering tasks. Use
    the following pieces of retrieved context to answer 
    the question. If you don't know the answer, just say
    "I don't know the answer.", Use three sentences maximum
    and keep the answer as concise as possible. \n
    \n
    {context} \n
    """
)

# Create a prompt Template for Question Answering
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

# Creating a stuff of documents chain
documents_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a Retrieval Chain
retrieval_chain = create_retrieval_chain(retriever_chain, documents_chain)

chat_history: List[Union[HumanMessage, AIMessage]] = []

def continue_question():
    """
    Continues the conversation by prompting the user for a new question.
    """
    question = input("Enter your question (or 'q' to quit): ")
    if question.lower() == 'q':
        return False
    response = retrieval_chain.invoke(
        {
            "chat_history": chat_history,
            "input": question,
        }
    )
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response["answer"]))
    print(response["answer"])
    return True

if __name__ == "__main__":
    while continue_question():
        pass

print(f"Thank you for using the Chatbot! {chat_history}")