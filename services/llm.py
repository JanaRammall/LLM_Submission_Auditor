"""Shared Gemini client factory for LLM and embedding calls.

Centralizing model construction avoids repeated API-key handling across the
codebase. The retry wrapper is used for transient model/API failures.
"""

from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.config import get_settings


def get_llm(model_name: Optional[str] = None) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model_name or settings.chat_model,
        api_key=settings.google_api_key,
        temperature=0.0,
    )

# create the gemini embedding model 
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )

# calls the llm with retry -- decorator for invoke_with_retry function
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def invoke_with_retry(llm: ChatGoogleGenerativeAI, prompt):
    return llm.invoke(prompt)
