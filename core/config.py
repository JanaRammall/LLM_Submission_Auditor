"""Application settings loaded from the project-root .env file.

Most modules should call get_settings() instead of reading environment
variables directly. That keeps API-key aliases, model names, chunk sizes, and
retrieval defaults in one place.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    s2_api_key: str = ""
    chat_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"
    vector_db_dir: str = "data/vector_db"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_k: int = 5


def _get_env_value(name: str, aliases: tuple[str, ...] = ()) -> str:
    for key in (name, *aliases):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def get_settings() -> Settings:
    google_api_key = _get_env_value("GOOGLE_API_KEY", aliases=("GEMINI_API_KEY",))
    s2_api_key = _get_env_value("S2_API_KEY", aliases=("SEMANTIC_SCHOLAR_API_KEY",))

    if not google_api_key:
        raise ValueError(
            f"GOOGLE_API_KEY is missing. Create {ENV_PATH} with "
            "GOOGLE_API_KEY=your_google_api_key_here. "
            "S2_API_KEY is optional but recommended for Semantic Scholar."
        )

    return Settings(
        google_api_key=google_api_key,
        s2_api_key=s2_api_key,
    )
