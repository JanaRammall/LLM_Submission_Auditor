"""Chunk report text and manage the Chroma vector store used for retrieval.

The vector store supports two workflows:
- audit evidence retrieval for semantic rubric checks;
- chatbot retrieval so answers can cite submitted-paper chunks.
"""

import os
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import get_settings
from services.llm import get_embeddings


def chunk_artifact_text(text: str, artifact_type: str, source_name: str, chunk_prefix: str) -> List[Document]:
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)

    docs: List[Document] = []
    for idx, chunk in enumerate(chunks):
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "chunk_id": f"{chunk_prefix}{idx:03d}",
                    "artifact_type": artifact_type,
                    "source_name": source_name,
                }
            )
        )
    return docs


def build_or_load_vector_store_from_docs(docs: List[Document], store_key: str) -> Tuple[Chroma, str]:
    """Create a Chroma store once per report, or reuse it on later reruns.

    Streamlit reruns the script often. Reusing a persisted store avoids paying
    embedding cost every time the user changes tabs or refreshes the page.
    """
    settings = get_settings()
    persist_dir = os.path.join(settings.vector_db_dir, store_key)

    embeddings = get_embeddings()

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vector_store = Chroma(
            collection_name="submission_chunks",
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        return vector_store, store_key

    os.makedirs(persist_dir, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="submission_chunks",
        persist_directory=persist_dir
    )
    return vector_store, store_key


def load_vector_store(store_id: str) -> Chroma:
    settings = get_settings()
    persist_dir = os.path.join(settings.vector_db_dir, store_id)

    embeddings = get_embeddings()

    return Chroma(
        collection_name="submission_chunks",
        embedding_function=embeddings,
        persist_directory=persist_dir
    )


def retrieve_evidence(vector_store: Chroma, query: str, k: int | None = None):
    settings = get_settings()
    return vector_store.similarity_search(query=query, k=k or settings.retrieval_k)
