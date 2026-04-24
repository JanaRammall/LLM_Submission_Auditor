from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent # create_agent is a function that takes a language model and tools and returns an agent

from langchain.tools import tool  # Decorator that converts a Python function into a tool that can be used by the LLM agent

import os
from dotenv import load_dotenv

load_dotenv()

# a decorator is a function that takes a function and returns a modified function
@tool # this decorator is used to tell the LLM that this is a tool
def add_two_numbers(a: int, b: int) -> str: # the colons and arrows are used to tell the type of the arguments and the return type
    """
    Adds two numbers together
    Args:
        a (int): The first number
        b (int): The second number
    Returns:
        str: The sum of the two numbers
    """ # this is the docstring of the tool, and particularly langchain will have the llm use the docstring to understand what the tool does
    return str(a + b) # return type must be a string because the LLM will try to format it as a string

class GeminiChat:
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        """
        Initialize GeminiChat with a language model.

        Args:
            model_name (str): The model to use. Default is "gemini-pro".
            temperature (float): The temperature to use. Default is 0.0.
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=os.environ.get("GOOGLE_API_KEY"), 
            temperature=temperature
        )
        # Create a react agent that can use the LLM and the tools
        self.agent = create_agent(self.llm, tools=[add_two_numbers])
        # Initialize empty list to store conversation history
        self.messages = []
        
    def send_message(self, message: str) -> str:
        """
        Send a message and get response from the model.
        
        Args:
            message (str): The message to send
            
        Returns:
            str: The model's response content
        """
        # Add user's message to conversation history
        self.messages.append(HumanMessage(content=message))
        # Get the length of the conversation history
        history_length = len(self.messages)
        # Before self.llm was returning an AI message, now it is returning the whole chat history, which might include tool messages
        self.messages = self.agent.invoke({"messages": self.messages})["messages"] # agent will return the whole chat history
        # self.messages[-1] is the last message in the chat history (after the tool call)
        new_messages = self.messages[history_length:] # new_messages is the messages after the tool call

        return new_messages 

# Example usage
chat = GeminiChat()

# Send a message asking for the sum of 1 and 2 and store the response messages
messages = chat.send_message("Whats the sum of 1 and 2?")

# Iterate through each message in the response
for message in messages:
    # If the message has content (regular message), print its type and content
    if message.content:
        print(type(message), message.content)
    # If the message has no content (tool call), print its type and tool call details
    else:
        print(type(message), message.tool_calls[0])
