from io import BytesIO
from typing import List
from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file) -> str:
    pdf_bytes = uploaded_file.read()
    reader = PdfReader(BytesIO(pdf_bytes))

    pages: List[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"\n--- PAGE {i} ---\n{text}")

    return "\n".join(pages)