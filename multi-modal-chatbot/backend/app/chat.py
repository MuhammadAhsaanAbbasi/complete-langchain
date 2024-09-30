from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import (
    RunnableLambda,
    RunnableBranch,
)
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

# Create models
openai = ChatOpenAI(model="gpt-4o-mini")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Function for handling FAQ
def faq_handling(prompt: str):
    faq_handling_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful FAQ assistant that can answer questions about automobiles."),
            ("human", """You have to give out the details of the product for LinkedIn post about {prompt}.
        Please follow the following guidelines:
        - use an eye-catching & interactive hook as an headline of the post in the content of the post,
        - Focus on the latest trends, key players, and significant news related to {prompt}.
        - Develop a detailed content outline that includes an introduction, key points, and a call to action.
        - Include SEO-friendly keywords, relevant hashtags, and data or resources to enhance the post's visibility and impact.
    """),
        ]
    )
    return faq_handling_template.format_prompt(prompt=prompt)

# Function for content generation
def content_generation(prompt: str):
    content_generation_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a professional content writer. generate me an content for social media post"
            ),
            ("human", "{prompt} \n"),
        ]
    )
    return content_generation_template.format_prompt(prompt=prompt)

# Improved Classification template of the Input
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a knowledgeable assistant trained to classify user prompts."),
        ("human", "Based on the following prompt, determine whether it pertains to an FAQ about automobiles or content generation. Respond with 'FAQ' or 'Content Generation': \n\n{prompt}\n"),
    ]
)

# Create branches for these templates
classification = classification_template | openai | StrOutputParser()

faq_branch_chain = (
    RunnableLambda(lambda x: faq_handling(x)) | llm | StrOutputParser()
)

content_branch_chain = (
    RunnableLambda(lambda x: content_generation(x)) | llm | StrOutputParser()
)

# Create the RunnableBranch to route based on classification
branches: RunnableBranch[str, str] = RunnableBranch(
        (
            RunnableLambda(lambda x: "FAQ" in x),
            faq_branch_chain,
        ),
        RunnableLambda(lambda x: "Content" in x) | content_branch_chain,
)

# Combine Classification and Branches
chain = classification | branches

# Testing the chain with a prompt
prompt = "Content about Tesla Model S?"
response = chain.invoke({"prompt": prompt})  # Await the invocation

print(response)