import hashlib
import os
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings


def _doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _chunk_documents(text: str) -> List[Document]:
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
                metadata={"chunk_id": f"C{idx:03d}"}
            )
        )
    return docs


def build_or_load_vector_store(text: str) -> Tuple[Chroma, str]:
    settings = get_settings()
    doc_id = _doc_hash(text)
    persist_dir = os.path.join(settings.vector_db_dir, doc_id)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key
    )

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vector_store = Chroma(
            collection_name="submission_chunks",
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        return vector_store, doc_id

    os.makedirs(persist_dir, exist_ok=True)
    docs = _chunk_documents(text)

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="submission_chunks",
        persist_directory=persist_dir
    )
    return vector_store, doc_id


def retrieve_evidence(vector_store: Chroma, query: str, k: int | None = None) -> List[Document]:
    settings = get_settings()
    return vector_store.similarity_search(query=query, k=k or settings.retrieval_k)