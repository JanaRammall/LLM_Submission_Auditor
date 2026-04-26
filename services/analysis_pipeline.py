"""End-to-end Streamlit analysis pipeline for uploaded PDFs.

This module owns the expensive workflow triggered from the Submission tab:
1. Extract text from the rubric/instructions PDF.
2. Ask the LLM to compile the rubric, then map it to the compact course rubric.
3. Extract report text and deterministic report features.
4. Build or load a Chroma vector store for report chunks.
5. Run deterministic and semantic rubric checks.
6. Prepare related-paper and novelty-analysis results.

The cache decorators live here because Streamlit caches must wrap the functions
used by the UI session.
"""

import hashlib
import time

import streamlit as st

from audit.checker import run_audit
from audit.compact_rubric import build_compact_rubric, build_instructor_rubric
from audit.report_analyzer import build_report_features
from audit.rubric_compiler import compile_rubric
from rag.retrieval import chunk_artifact_text, build_or_load_vector_store_from_docs
from research.related_papers import get_related_papers_from_report


@st.cache_data(show_spinner=False)
def cached_compile_rubric(instructions_text: str, model_name: str):
    return compile_rubric(instructions_text, model_name=model_name)


@st.cache_data(show_spinner=False)
def cached_report_features(report_text: str):
    return build_report_features(report_text)


@st.cache_data(show_spinner=False)
def cached_pdf_text(file_bytes: bytes, kind: str) -> str:
    from io import BytesIO
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"\n--- PAGE {i} ---\n{text}")
    return "\n".join(pages)


@st.cache_data(show_spinner=False)
def cached_related_papers(
    report_text: str,
    model_name: str,
    limit: int = 5,
    pipeline_version: str = "semantic-v5",
):
    return get_related_papers_from_report(
        report_text=report_text,
        limit=limit,
        model_name=model_name,
    )


def run_evaluation_analysis(
    uploaded_instructions,
    uploaded_report,
    model_name: str,
    use_instructor_rubric: bool = True,
):
    """Run the rubric/evaluation part of the submission workflow.

    The UI calls this once after both PDFs are uploaded. The function writes
    evaluation outputs into session state immediately, before similarity starts.
    This keeps the user informed as soon as evaluation is actually done.
    """
    started_at = time.perf_counter()
    st.session_state.related_papers_result = None
    st.session_state.related_papers_status = "pending"
    st.session_state.analysis_completed = False
    st.session_state.analysis_duration_seconds = None
    st.session_state.similarity_completed = False
    st.session_state.submission_notice = "Analysis is running..."
    st.session_state.chat_messages = []

    if use_instructor_rubric:
        with st.spinner("Loading instructor rubric..."):
            st.session_state.compiled_rubric = build_instructor_rubric()
    else:
        with st.spinner("Extracting instructions text..."):
            instructions_text = cached_pdf_text(uploaded_instructions.getvalue(), "instructions")

        with st.spinner("Compiling custom rubric..."):
            full_rubric = cached_compile_rubric(instructions_text, model_name)
            compact_rubric = build_compact_rubric(full_rubric)
            st.session_state.compiled_rubric = compact_rubric

    with st.spinner("Extracting report text..."):
        report_text = cached_pdf_text(uploaded_report.getvalue(), "report")
        st.session_state.report_text = report_text

    with st.spinner("Running deterministic report analysis..."):
        report_features = cached_report_features(report_text)
        st.session_state.report_features = report_features

    with st.spinner("Building/loading vector store..."):
        docs = chunk_artifact_text(
            text=report_text,
            artifact_type="report",
            source_name=uploaded_report.name,
            chunk_prefix="R",
        )

        store_key_input = uploaded_report.name + report_text[:5000]
        store_key = hashlib.sha256(store_key_input.encode("utf-8")).hexdigest()[:16]
        vector_store, store_id = build_or_load_vector_store_from_docs(docs, store_key)
        st.session_state.store_id = store_id

    available_artifacts = {
        "report": True,
        "readme": False,
        "code": False,
        "slides": False,
    }

    with st.spinner("Running evaluation..."):
        audit_report = run_audit(
            compiled_rubric=st.session_state.compiled_rubric,
            vector_store=vector_store,
            report_features=report_features,
            available_artifacts=available_artifacts,
            model_name=model_name,
        )
        st.session_state.audit_report = audit_report

    st.session_state.analysis_completed = True
    st.session_state.analysis_duration_seconds = round(time.perf_counter() - started_at, 1)
    st.session_state.submission_notice = (
        f"Evaluation completed in {st.session_state.analysis_duration_seconds}s. "
        "Preparing similarity next."
    )


def prepare_similarity_analysis(report_text: str, model_name: str) -> None:
    """Prepare related-paper similarity after evaluation has already completed."""
    st.session_state.related_papers_status = "running"
    st.session_state.similarity_completed = False

    try:
        related_papers_result = cached_related_papers(
            report_text=report_text,
            model_name=model_name,
            limit=5,
        )
        st.session_state.related_papers_result = related_papers_result
        st.session_state.related_papers_status = "ready"
        st.session_state.similarity_completed = True
        st.session_state.submission_notice = (
            f"Evaluation completed in {st.session_state.analysis_duration_seconds}s. "
            "Similarity is ready."
        )
    except Exception as e:
        st.session_state.related_papers_result = None
        st.session_state.related_papers_status = "failed"
        st.session_state.submission_notice = (
            f"Evaluation completed in {st.session_state.analysis_duration_seconds}s. "
            "Similarity could not be prepared in this run."
        )
        st.warning(f"Evaluation finished, but related papers could not be prepared now: {e}")
