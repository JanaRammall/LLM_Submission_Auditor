# Import required modules
import asyncio
from langchain_chroma import Chroma  # Vector database
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # Google's LLM and embeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # Message types for chat
from langchain.agents import create_agent  # For creating an agent that can use tools
from langchain.tools import tool  # Decorator to create tools
from pathlib import Path
import os
import streamlit as st


def _format_ai_content(content) -> str:
    """Normalize AI message content into plain text for Streamlit rendering."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)

    return str(content)

# API key for Google services
api_key = os.environ.get("GOOGLE_API_KEY")
BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = str(BASE_DIR / "vector_db")

# Tool definition for adding numbers
@tool
def add_two_numbers(a: int, b: int) -> str:
    """
    Adds two numbers together
    Args:
        a (int): The first number
        b (int): The second number
    Returns:
        str: The sum of the two numbers
    """
    # Convert result to string since LLM expects string output
    return str(a + b)

# Tool definition for searching vector database
@tool
def search_vector_db(query: str) -> str:
    """
    Search the vector database for documents similar to the query.
    Args:
        query (str): The search query string to find relevant documents
    Returns:
        str: A concatenated string of the top 5 most similar document contents found in the vector database
    """
    # Initialize embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)
    
    # Initialize/connect to vector database
    vector_store = Chroma(
        collection_name = "embeddings",
        embedding_function = embeddings,
        persist_directory = PERSIST_DIR,        
        collection_metadata = {"hnsw:space": "cosine"}  # Use cosine similarity
    )

    # Debug print
    print("Searching the vector database for: ", query)
    
    # Perform similarity search and get top 5 results
    result = vector_store.similarity_search(query = query, k = 5)
    # Combine all document contents into single string
    result_str = "\n".join([doc.page_content for doc in result])
    
    return result_str

# Main chat class that uses Gemini LLM
class GeminiChat:
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        """
        Initialize GeminiChat with a language model.

        Args:
            model_name (str): The model to use. Default is "gemini-pro".
            temperature (float): The temperature to use. Default is 0.0.
        """
        # Store API key
        self.api_key=api_key
        
        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=self.api_key, 
            temperature=temperature
        )
        
        # Create agent with both tools available
        self.agent = create_agent(self.llm, tools=[add_two_numbers, search_vector_db])
        
        # Initialize conversation history
        # self.messages = []
        self.messages = [SystemMessage(content="""You are a helpful AI assistant named COE548 that can search through documents and use the appropriate tool when needed to help answer questions. 
                                      When you use a tool you will strictly adhere to the tool output and not use additional information. You will be polite.""")]
        
    def send_message(self, message: str) -> str:
        """
        Send a message and get response from the model.
        
        Args:
            message (str): The message to send
            
        Returns:
            str: The model's response content
        """
        # Add user message to history
        self.messages.append(HumanMessage(content=message))
        
        # Store current history length to identify new messages later
        history_length = len(self.messages)
        
        # Get response from agent, including any tool usage
        self.messages = self.agent.invoke({"messages": self.messages})["messages"]
        
        # Extract only the new messages from this interaction
        new_messages = self.messages[history_length:]

        return new_messages 


# Streamlit interface
def main():
    # Set the title of the Streamlit app
    st.title("Gemini Chat")
    # Add descriptive text below the title to inform users what they can do
    st.write("Ask anything about the documents in the vector database")

    # Initialize LLM instance if not already in session state
    # This ensures the chat model persists across page refreshes
    # Also ensures that the LLM instance is created only once
    if "agent" not in st.session_state: # session state is a dictionary that stores the state of the application and does not get reset on page refresh
        st.session_state.agent = GeminiChat()
        
    # Initialize message history in session state if not already present
    # This stores the chat history between user and AI across page refreshes
    if "messages" not in st.session_state:
        st.session_state.messages = [] # Empty list to store message history

    # Display all previous messages from session state
    for message in st.session_state.messages:
        # Create chat message UI element with appropriate type (user/assistant) and display content
        
        # Handle AI message with content (regular response)
        if isinstance(message, AIMessage) and message.content:
            with st.chat_message("assistant"):
                st.markdown(_format_ai_content(message.content))
        # Handle user message
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
    
    # Get user input from chat interface using Streamlit's chat_input widget
    # Returns None if no input is provided
    prompt = st.chat_input("Your message")

    # Get user input from chat interface. 
    if prompt:
        # Add user's message to session state history
        st.session_state.messages.append(HumanMessage(content=prompt))
        # Display user's message in chat UI
        with st.chat_message("user"):
            st.markdown(prompt)

        # Send message to LLM and get response messages (may include tool usage)
        messages = st.session_state.agent.send_message(prompt)
        print(messages)
        # Add all new messages (including tool calls) to session state history
        st.session_state.messages.extend(messages)

        # Process response messages
        for message in messages:
            # Check if message is from AI (not a tool call) and has content
            # When it is a tool call, AIMessage object is created but it has no content
            # isinstance(message, AIMessage) will skip tool outputs
            if isinstance(message, AIMessage) and message.content:
                # Display AI response in chat UI
                with st.chat_message("assistant"):
                    st.markdown(_format_ai_content(message.content))

main()

