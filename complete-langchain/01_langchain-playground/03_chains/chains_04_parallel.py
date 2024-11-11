from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser

# load environment variables from .env files
load_dotenv()


# Create a models of Google and OpenAI
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

openai_model = ChatOpenAI(model="gpt-4o-mini")


# Create a prompt for chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful automotive assistant to give me an expert review on automobiles?"),
        ("human", "List the main features of the car {car}?"),
    ]
)


# Define pros Analysis step
def pros_analysis(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert automobiles Reviewer?"),
            ("human", 
            "Given these features of the car: {features}, list the pros of these features?")
        ]
    )
    return pros_template.format_prompt(features=features)

# Define cons Analysis step
def cons_analysis(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert automobiles Reviewer?"),
            ("human", 
            "Given these features of the car: {features}, list the cons of these features?")
        ]
    )
    return cons_template.format_prompt(features=features)

# Combine pros & cons into a final Review
def combine_review(pros, cons):
    return f"Pros: {pros}\n\nCons: {cons}"


# Simplify Branches with LCEL
pros_branch_chain = (
    RunnableLambda(lambda x: pros_analysis(x)) | openai_model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: cons_analysis(x)) | openai_model | StrOutputParser()
)


# Create a combine Chain use Langchain Expression Language (LCEL)
chain = (
    prompt
    | model
    |StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_review(x["branches"]['pros'], x["branches"]['cons'],))
)


# run the chain
result = chain.invoke({"car": "Lamborghini Huracan Spider"})

# Output
print(result)