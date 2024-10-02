# Langchain Course Repository

Welcome to the **Langchain Course** repository! This repository contains code samples, exercises, and resources designed to help you get started with Langchain, a powerful Python library for building applications that interact with language models like GPT-4, GPT-4o, llama-3.1 & others.

## Overview

Langchain is a framework that simplifies the development of language model-based applications. It enables developers to harness the capabilities of large language models (LLMs) for a variety of tasks, such as building chatbots, generating text, handling complex conversations, and creating advanced AI-driven applications.

This course will guide you through the core concepts of Langchain and help you build applications that can integrate language models effectively. Whether you're a beginner or an experienced developer, this course is designed to help you understand how to use Langchain in various real-world scenarios.

## 1. ChatModel

A ChatModel in LangChain is a specialized version of a language model designed to handle conversation-based inputs and outputs. Rather than dealing with simple text input and output, a ChatModel works with messages—pieces of text that are part of a dialogue, such as user inputs and model responses.

### Key Features of ChatModel

- **Message Format**: ChatModels accept a structured format for inputs, typically with three types of messages:
  - **SystemMessage**: A *SystemMessage* sets the context or behavior for the conversation. It’s like giving instructions to the AI on how to act, such as "You are a professional Python developer helping to write code." It defines the role or tone for the chat but doesn't directly interact with the conversation itself.
  - **HumanMessage**: Represents the user's input in a conversation (e.g., questions or prompts).
  - **AIMessage**: Response from the model to the user's input. It’s what the AI sends back based on the conversation or prompt provided.

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

## 3. Chains

In LangChain, **Chains** are like workflows that connect different steps or processes together. Each step in the chain takes some input, does something with it (like generating text or answering a question), and then passes the result to the next step.

Think of it like an assembly line: one machine does its job and then passes the product to the next machine to finish the process. In LangChain, these "machines" are language models (or other tools), and the "product" is the information they generate or process.

Chains make it easy to combine different tasks and create more complex systems by connecting multiple steps in a sequence!

### Chains Possibilities
- **Chain Extended**: This is when you connect multiple steps or language models in a sequence, with each step passing its output as input to the next step. It’s like a pipeline where each task builds on the result of the previous one, allowing you to perform a series of actions in a structured order.
   ![Chain Extended](https://myapplication-logos.s3.ap-south-1.amazonaws.com/extended.jpg)

- **Parallel Chain**: In a parallel chain, multiple steps run at the same time, each working independently on different tasks or inputs. After all the steps finish, their results are combined or used together. This is useful when you need to process several pieces of information simultaneously.
    ![Parallel Chain](https://myapplication-logos.s3.ap-south-1.amazonaws.com/parallel.jpg)

- **Chain Branching**: Chain branching involves splitting the workflow into different paths based on certain conditions or criteria. Depending on the input or result, the process can take different directions, allowing for more flexible and dynamic workflows.
    ![Chain Branching](https://myapplication-logos.s3.ap-south-1.amazonaws.com/branching.jpg)

<hr />

<h2 align="center">Dear Brother and Sister Show some ❤ by <img src="https://imgur.com/o7ncZFp.jpg" height=25px width=25px> this repository!</h2>