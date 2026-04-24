import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Config ---
BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "PDF files"  # Directory containing PDF files to index
PERSIST_DIR = str(BASE_DIR / "vector_db")
COLLECTION_NAME = "embeddings"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Embeddings ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

# --- Vector DB ---
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
    collection_metadata={"hnsw:space": "cosine"},
)

# Check if the vector DB already contains documents
if vector_store._collection.count() == 0:
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF directory not found: {PDF_DIR}")

    documents = []

    for pdf_file in PDF_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()   # one document per page

        for page in pages:
            page.metadata["filename"] = pdf_file.name

        documents.extend(pages)

    vector_store.add_documents(documents)
    print("Documents added to vector database.")
    print(f"Indexed {len(documents)} documents.")

else:
    print("Vector database already contains documents. Skipping insert.")

# Testing similarity search
query = input("Enter a search query: ")

results = vector_store.similarity_search(query=query, k=5)

print(f"\nTop {len(results)} semantic search results:\n")

for i, doc in enumerate(results, start=1):
    print(f"Result {i}")
    print(f"Source: {doc.metadata.get('filename', 'Unknown file')}")
    print(f"Page: {doc.metadata.get('page', 'Unknown page')}")
    print(doc.page_content[:700])
    print("=" * 100)


