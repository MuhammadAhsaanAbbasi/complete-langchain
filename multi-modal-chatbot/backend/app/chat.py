from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.runnable import (
    RunnableLambda,
    RunnableBranch,
)
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()

# Create models
openai = ChatOpenAI(model="gpt-4")

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    do_sample=True,
    max_new_tokens=1024,
    temperature=0.2,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN"),
)

# llm2 = HuggingFaceEndpoint(
#     model="meta-llama/Llama-3.1-8B-Instruct",
#     task="text-generation",
#     do_sample=False,
#     max_new_tokens=100,
#     temperature=0.2,
#     huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN"),
# )

headers = {
        "ngrok-skip-browser-warning": "true"  # This header bypasses the Ngrok browser warning
    }

llm2 = OllamaLLM(model="llama3.2", base_url="https://21f2-34-16-203-146.ngrok-free.app/", metadata=headers)

# Function for handling FAQ
def faq_handling(prompt: str):
    systemMessage = SystemMessagePromptTemplate.from_template(
        "You are a knowledgeable assistant trained to handle FAQs about automobiles."
    )
    humanMessage = HumanMessagePromptTemplate.from_template("{prompt}")
    faq_handling_template = ChatPromptTemplate.from_messages(
        [
            systemMessage,
            humanMessage,
        ]
    )
    return faq_handling_template.format_messages(prompt=prompt)

async def multi_modal_chat(prompt: str):
    # Function for content generation
    def content_generation(prompt: str):
        systemMessage = SystemMessagePromptTemplate.from_template("You are a professional content writer. Write me an interactive LinkedIn post on {prompt}.")
        humanMessage = HumanMessagePromptTemplate.from_template("""
        You have to give out the details of the product for LinkedIn post about {prompt}.
        Please follow the following guidelines:
            - use an eye-catching & interactive hook as an headline of the post in the content of the post,
            - Focus on the latest trends, key players, and significant news related to {prompt}.
            - Develop a detailed content outline that includes an introduction, key points, and a call to action.
            - Include SEO-friendly keywords, relevant hashtags, and data or resources to enhance the post's visibility and impact.
        """)
        content_generation_template = ChatPromptTemplate.from_messages(
            [
                systemMessage,
                humanMessage,
            ]
        )
        return content_generation_template.format_messages(prompt=prompt)

    # Improved Classification template of the Input
    classification_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a knowledgeable assistant trained to classify user prompts."),
            ("human", "Based on the following prompt, determine whether it pertains to an FAQ about automobiles or content generation. Respond with 'FAQ' or 'Content Generation': \n\n{prompt}\n"),
        ]
    )

    def classify_input(prompt: str):
        mesages = classification_template.format_messages(prompt=prompt)
        response = openai.invoke(mesages)
        print(f"Classification response: {response.content}")
        return response.content

    # Create branches for these templates
    classification = RunnableLambda(
        lambda x: {"classification": classify_input(x["prompt"]), "original_prompt": x}
    )

    # print(f"Classification: {classification}")

    faq_branch_chain = (
        RunnableLambda(lambda x: faq_handling(x["original_prompt"])) | llm | StrOutputParser()
    )

    content_branch_chain = (
        RunnableLambda(lambda x: content_generation(x["original_prompt"])) | llm2 | StrOutputParser()
    )

    # Create the RunnableBranch to route based on classification
    branches: RunnableBranch = RunnableBranch(
            (
                lambda x: "FAQ" in x["classification"],
                faq_branch_chain,
            ),
            content_branch_chain,
    )

    # Combine Classification and Branches
    chain = RunnableLambda(lambda x: {"prompt": x}) | classification | branches

    response = chain.invoke(prompt)  # Await the invocation if using async

    return response
