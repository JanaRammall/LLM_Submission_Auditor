from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    chat_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"
    vector_db_dir: str = "data/vector_db"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_k: int = 5


def get_settings() -> Settings:
    api_key = "AIzaSyB6Eg4Ug04TgQEYAjE1-JqA14v07UgTKJ4"
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file.")
    return Settings(google_api_key=api_key)