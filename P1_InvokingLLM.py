# Import required modules from langchain
from langchain_google_genai import ChatGoogleGenerativeAI  # For interacting with Google's Gemini model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # Message types for chat

import os
from dotenv import load_dotenv

load_dotenv()

# Function to create a Gemini LLM using type hints
def create_gemini_llm(model_name: str = "gemini-2.5-flash", temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """
    Creates a Gemini LLM.

    Args:
        model_name (str): The model to use. Default is "gemini-pro".
        temperature (float): The temperature to use.
    
    Returns:
        ChatGoogleGenerativeAI: The Gemini LLM.
    """
    # Initialize the Gemini language model with specified parameters
    llm = ChatGoogleGenerativeAI(
        model=model_name,  # Specify which Gemini model to use
        api_key="AIzaSyAHblrOEj33E8rSZWYixmc6V6J0y3lAbaE",  # API key for authentication
        temperature=temperature  # Controls randomness in responses (0.0 = deterministic)
    )
    
    return llm

# Create an instance of the Gemini LLM using default parameters
llm = create_gemini_llm()

# # Send a test message to the LLM and print its response
# # invoke() sends the message and returns an AIMessage object
# # .content gets the actual response text
# print(llm.invoke("Hello, how are you?").content)

# Initialize chat history with first human message
messages = [HumanMessage(content="Hello, how are you?")]

# Send messages to LLM and get response
# llm.invoke(messages) will return a AIMessage object and AIMessage object has a variable called content
response = llm.invoke(messages)
print(response.content)
# Add AI's response to chat history
messages.append(response)

# Add follow-up question to chat history
# This tests if the LLM can remember previous messages
messages.append(HumanMessage(content="What is the first thing I asked?"))

# Get response from LLM with full chat history context
response = llm.invoke(messages)

# Print just the content of the AI's response
print(response.content)

