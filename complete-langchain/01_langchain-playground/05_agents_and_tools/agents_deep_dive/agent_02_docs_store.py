import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import Tool
from typing import List, Union

# load environment variables from .env files
_ = load_dotenv()

# Define the Directory Containing the Text Files & persistent Directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "..", "..", "04_rag", "db")
persist_dir = os.path.join(db_dir, "chroma_db_with_metadata")

# generate an embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

if os.path.exists(persist_dir):
    print("Persistent Directory Exists. Loading the Vector Store")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
else:
    print("Persistent Directory Does Not Exist. Initializing the Vector Store then try it.")

# db retriever
retrieval = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# create a chat model
llm = ChatOpenAI(
    model="gpt-4o",
)

# contextualize system prompt
contextualize_system_prompt = """ Given a chat history and the latest user question
    which might reference context in the chat history, formulate a standalone question
    that can be understood without the chat history. Do NOT answer the question,
    just reformulate it if necessary & otherwise return it as is.
    """


contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# history of retriever
history_retriever = create_history_aware_retriever(
    llm, retrieval, contextualize_prompt
)

# Answer system prompt
qa_system_prompt = """ You are an Assistant for Question Answering tasks. 
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say "I don't know the answer."
    don't know. Use three sentences maximum and keep the answer concise.
    \n\n
    {context}
    """


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# create a stuff documents chain
documents_chain = create_stuff_documents_chain(llm, qa_prompt)

# create a retrieval chain
retrieval_chain = create_retrieval_chain(history_retriever, documents_chain)

retrieval_tool = Tool(
    name="Retrieval Tool",
    func=lambda input, **kwargs: retrieval_chain.invoke(
            {"input": input, "chat_history": kwargs.get("chat_history", [])}
        ),
    description="Use this tool when you need to answer questions about the context you have provided.",
)

tools = [retrieval_tool]

# Pull the prompt template from the hub
prompt = hub.pull("hwchase17/react")

# create react agent
agent = create_react_agent(llm, tools, prompt)

# Execute the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)


chat_history: List[Union[HumanMessage, AIMessage]] = []

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
    print(f"Agent: {response["output"]}")
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response["output"]))
    print(f"Chat History: {chat_history}")
    print("\n\n")