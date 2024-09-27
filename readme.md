# Langchain Course Repository

Welcome to the **Langchain Course** repository! This repository contains code samples, exercises, and resources designed to help you get started with Langchain, a powerful Python library for building applications that interact with language models like GPT-3, GPT-4, and others.

## Overview

Langchain is a framework that simplifies the development of language model-based applications. It enables developers to harness the capabilities of large language models (LLMs) for a variety of tasks, such as building chatbots, generating text, handling complex conversations, and creating advanced AI-driven applications.

This course will guide you through the core concepts of Langchain and help you build applications that can integrate language models effectively. Whether you're a beginner or an experienced developer, this course is designed to help you understand how to use Langchain in various real-world scenarios.

## Table of Contents

1. [ChatModel](#chatmodel)
2. [Prompt Template](#prompt-template)
3. [Chains](#chains)

## 1. ChatModel

A ChatModel in LangChain is a specialized version of a language model designed to handle conversation-based inputs and outputs. Rather than dealing with simple text input and output, a ChatModel works with messages—pieces of text that are part of a dialogue, such as user inputs and model responses.

### Key Features of ChatModel

- **Message Format**: ChatModels accept a structured format for inputs, typically with three types of messages:
  - **SystemMessage**: Sets the context or behavior for the conversation.
  - **HumanMessage**: Represents the user's input in a conversation.
  - **AIMessage**: Response from the model to the user's input.

- **Multi-turn Conversations**: Capable of handling context-aware conversations.

- **Support for Different Providers**: Can integrate with various providers such as OpenAI's GPT models and Google’s Generative AI models.

### Invoke

The `invoke` method in LangChain processes the input and returns the result. It ensures that the entire chain of models or prompts is executed in the correct order.

### How Invoke Works

- **Input**: You provide input to the invoke method, usually a dictionary of key-value pairs.
- **Sequential Processing**: Input is passed through each model or step in the chain.
- **Output**: Returns the final result after processing.

## 2. Prompt Template

A Prompt Template in LangChain defines reusable, structured prompts for language models. It allows you to define a general structure with placeholders for dynamic information.

### Key Concepts of Prompt Templates

- **Static Structure with Dynamic Placeholders**: Fixed structure with placeholders filled in at runtime.

  ```python
  template = "Write a blog post about {topic} that focuses on {key_points}."
  ```

- **Customizable Input Variables**: Can take multiple input variables for tailoring prompts.

    ```python
    from langchain.prompts import PromptTemplate

    template = "Translate the following text to {language}: {text}"
    prompt_template = PromptTemplate(
    input_variables=["language", "text"],
    template=template
    )
    prompt = prompt_template.format(language="French", text="Hello, how are you?")
    # Output: "Translate the following text to French: Hello, how are you?"
    ```

- **Multiple Use Cases**: Used for various applications such as content generation, translation, customer support, and more.

- **Maintaining Consistency**: Ensures consistently structured prompts.




<h3 align="center">Dear Brother and Sister Show some ❤ by <img src="https://imgur.com/o7ncZFp.jpg" height=25px width=25px> this repository!</h3>