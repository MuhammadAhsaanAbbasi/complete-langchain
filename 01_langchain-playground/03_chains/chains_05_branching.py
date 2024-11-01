from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableBranch
from langchain.schema.output_parser import StrOutputParser

# load environment variables from .env files
load_dotenv()


# Create a models of Google and OpenAI
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

openai_model = ChatOpenAI(model="gpt-4o-mini")

# create a branches of prompt template for the chain
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful Customer Support Assistant."),
        ("human", "Generate a thank you message for this positive feedback: {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful Customer Support Assistant."),
        ("human", "Generate a response addressing for this negative feedback: {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful Customer Support Assistant."),
        ("human", "Generate a request for more details for this neutral feedback: {feedback}."),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful Customer Support Assistant."),
        ("human", "Generate an escalation this feedback for Human Agent: {feedback}."),
    ]
)


classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful Customer Support Assistant."),
        ("human", "Classify this feedback as positive, negative, neutral, or escalate: {feedback}."),
    ]
)


# Create a branches for these templates
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser(),
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser(),
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser(),
    ),
    escalate_feedback_template | model | StrOutputParser(),
)

# clarify the classification
classification_chain = classification_template | openai_model | StrOutputParser()

# Combine Classification and Branches
chain = RunnableLambda(lambda x: {"feedback": x}) | classification_chain | branches

# Feedbacks
# Positive Feedback: "This product exceeded my expectations! The quality and performance are top-notch, and I couldn't be happier with my purchase."

# Negative Feedback: "Unfortunately, the car didn't meet my expectations. It stopped working after a few uses, and I'm disappointed with the overall experience."

# Neutral Feedback: "The product works as described, but it's nothing exceptional. It gets the job done, but there's room for improvement."

# Escalate Feedback: "I'm very dissatisfied with this product. It arrived damaged, and customer support has not been helpful in resolving the issue. I need this escalated immediately."

# run the chain
review = "This Iphone16 exceeded my expectations! The quality and performance are top-notch, and I could be happier with my purchase."

response = chain.invoke({"feedback": review})
print(response)