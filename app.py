import hashlib
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from checker import run_audit
from retrieval import chunk_artifact_text, build_or_load_vector_store_from_docs
from rubric_compiler import compile_rubric
from report_analyzer import build_report_features
from compact_rubric import build_compact_rubric
from related_papers import get_related_papers_from_report

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"

st.set_page_config(page_title="LLM Submission Auditor", layout="wide")


def load_css(file_name: str = "styles.css") -> None:
    css_path = Path(__file__).parent / file_name
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_compile_rubric(instructions_text: str):
    return compile_rubric(instructions_text, model_name=MODEL_NAME)


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
def cached_related_papers(report_text: str, limit: int = 5):
    return get_related_papers_from_report(
        report_text=report_text,
        limit=limit,
        model_name=MODEL_NAME,
    )


load_css()

defaults = {
    "compiled_rubric": None,
    "audit_report": None,
    "report_features": None,
    "store_id": None,
    "related_papers_result": None,
    "related_papers_status": None,
    "report_text": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def run_full_analysis(uploaded_instructions, uploaded_report):
    st.session_state.related_papers_result = None
    st.session_state.related_papers_status = None

    with st.spinner("Extracting instructions text..."):
        instructions_text = cached_pdf_text(uploaded_instructions.getvalue(), "instructions")

    with st.spinner("Compiling rubric..."):
        full_rubric = cached_compile_rubric(instructions_text)
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
            model_name=MODEL_NAME,
        )
        st.session_state.audit_report = audit_report

    with st.spinner("Preparing related papers..."):
        try:
            related_papers_result = cached_related_papers(
                report_text=report_text,
                limit=5,
            )
            st.session_state.related_papers_result = related_papers_result
            st.session_state.related_papers_status = "ready"
        except Exception as e:
            st.session_state.related_papers_result = None
            st.session_state.related_papers_status = "failed"
            st.warning(f"Evaluation finished, but related papers could not be prepared now: {e}")


compiled = st.session_state.compiled_rubric
audit_report = st.session_state.audit_report
report_features = st.session_state.report_features
related_papers_result = st.session_state.related_papers_result
related_papers_status = st.session_state.related_papers_status
report_text = st.session_state.report_text

# ---------------------------
# Header
# ---------------------------

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-kicker">Specialized LLM Agent</div>
        <div class="hero-title">LLM Submission Auditor</div>
        <div class="hero-subtitle">
            Analyze a student submission against project requirements and prepare related-paper similarity results.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Top navbar
# ---------------------------

st.markdown('<div class="top-nav-wrap">', unsafe_allow_html=True)
page = st.radio(
    "Navigation",
    ["Evaluation", "Similarity Check"],
    horizontal=True,
    label_visibility="collapsed",
    key="top_nav_page",
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Evaluation page
# ---------------------------

if page == "Evaluation":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Submission</div>', unsafe_allow_html=True)

    uploaded_instructions = st.file_uploader(
        "Upload the project instructions / rubric PDF",
        type=["pdf"],
        key="instructions_uploader",
    )

    uploaded_report = st.file_uploader(
        "Upload the student report PDF",
        type=["pdf"],
        key="report_uploader",
    )

    run_clicked = st.button("Run Analysis")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_instructions is not None and uploaded_report is not None and run_clicked:
        try:
            run_full_analysis(uploaded_instructions, uploaded_report)
            st.success("Analysis completed.")
        except Exception as e:
            st.error(f"Process failed: {e}")

    compiled = st.session_state.compiled_rubric
    audit_report = st.session_state.audit_report
    report_features = st.session_state.report_features
    related_papers_result = st.session_state.related_papers_result
    related_papers_status = st.session_state.related_papers_status

    if compiled:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Compact Rubric</div>', unsafe_allow_html=True)
        st.write(f"**Project Title:** {compiled.project_title}")
        st.caption(f"Criteria used: {len(compiled.criteria)}")

        with st.expander("View compact criteria", expanded=False):
            for criterion in compiled.criteria:
                st.markdown(f"**{criterion.criterion_id} — {criterion.title}**")
                st.write(criterion.description)
                st.divider()
        st.markdown("</div>", unsafe_allow_html=True)

    if report_features:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Deterministic Report Analysis</div>', unsafe_allow_html=True)
        st.json(report_features)
        st.markdown("</div>", unsafe_allow_html=True)

    if audit_report:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Evaluation Summary</div>', unsafe_allow_html=True)

        pass_count = sum(1 for r in audit_report.active_results if r.status == "Pass")
        partial_count = sum(1 for r in audit_report.active_results if r.status == "Partial")
        fail_count = sum(1 for r in audit_report.active_results if r.status == "Fail")
        unknown_count = sum(1 for r in audit_report.active_results if r.status == "Not enough evidence")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pass", pass_count)
        c2.metric("Partial", partial_count)
        c3.metric("Fail", fail_count)
        c4.metric("Not enough evidence", unknown_count)

        st.caption(f"Vector store ID: {st.session_state.get('store_id', 'N/A')}")
        st.divider()

        for item in audit_report.active_results:
            icon = {
                "Pass": "✅",
                "Partial": "🟡",
                "Fail": "❌",
                "Not enough evidence": "⚪",
            }[item.status]

            with st.expander(f"{icon} {item.criterion_id} — {item.status}", expanded=False):
                st.write(f"**Evidence found:** {item.evidence_found or 'None'}")
                st.write(f"**Missing / weak:** {item.missing_or_weak}")
                st.write(f"**Improvement:** {item.improvement}")
                st.write(
                    f"**Evidence chunk IDs:** {', '.join(item.evidence_chunk_ids) if item.evidence_chunk_ids else 'None'}"
                )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Related Papers Preview</div>', unsafe_allow_html=True)

        if related_papers_status == "ready" and related_papers_result:
            if not related_papers_result.papers:
                st.info("No strong related papers were found.")
            else:
                for paper in related_papers_result.papers[:2]:
                    title = paper.title or "Untitled Paper"
                    venue = paper.venue or "N/A"
                    year = paper.year if paper.year else "N/A"
                    reason = paper.similarity_reason or "No similarity note available."

                    st.markdown(
                        f"""
                        <div class="paper-card">
                            <div class="paper-title">{title}</div>
                            <div class="paper-meta">{year} • {venue}</div>
                            <div class="paper-reason">{reason}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.info("Open the Similarity Check page to see the full list.")
        elif related_papers_status == "failed":
            st.warning("Related papers could not be prepared in this run.")
        else:
            st.info("Related papers are not ready yet.")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Similarity page
# ---------------------------

elif page == "Similarity Check":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Similarity Check</div>', unsafe_allow_html=True)

    if not report_text:
        st.info("Run the analysis first so the system can save the report and prepare related papers.")
    else:
        if related_papers_status == "ready" and related_papers_result:
            st.success("Related papers are already prepared.")
            st.write(f"**Project Title:** {related_papers_result.query_title}")
            st.write(f"**Summary used for search:** {related_papers_result.query_summary}")

            if not related_papers_result.papers:
                st.info("No strong related papers were found.")
            else:
                for paper in related_papers_result.papers:
                    title = paper.title or "Untitled Paper"
                    year = paper.year if paper.year else "N/A"
                    venue = paper.venue or "N/A"
                    authors = ", ".join(paper.authors) if paper.authors else "N/A"
                    reason = paper.similarity_reason or "No similarity reason available."
                    abstract = paper.abstract or "No abstract available."
                    url = paper.url or ""

                    st.markdown(
                        f"""
                        <div class="paper-card">
                            <div class="paper-title">{title}</div>
                            <div class="paper-meta">{year} • {venue}</div>
                            <div class="paper-reason"><strong>Why it is similar:</strong> {reason}</div>
                            <div class="meta-pill-row">
                                <span class="meta-pill">{year}</span>
                                <span class="meta-pill">{venue}</span>
                            </div>
                            <div style="margin-top:12px;"><strong>Authors:</strong> {authors}</div>
                            <div class="paper-abstract"><strong>Abstract:</strong> {abstract}</div>
                            <div class="link-line">{f'<a href="{url}" target="_blank">Open paper</a>' if url else ''}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        elif related_papers_status == "failed":
            st.warning("Related papers were not prepared successfully in the previous run.")

            if st.button("Retry Similarity Check"):
                try:
                    with st.spinner("Retrying related paper retrieval..."):
                        related_papers_result = cached_related_papers(
                            report_text=report_text,
                            limit=5,
                        )
                        st.session_state.related_papers_result = related_papers_result
                        st.session_state.related_papers_status = "ready"
                    st.success("Related papers retrieved successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Retry failed: {e}")

        else:
            st.info("Related papers are not prepared yet.")

            if st.button("Run Similarity Check Now"):
                try:
                    with st.spinner("Finding related papers..."):
                        related_papers_result = cached_related_papers(
                            report_text=report_text,
                            limit=5,
                        )
                        st.session_state.related_papers_result = related_papers_result
                        st.session_state.related_papers_status = "ready"
                    st.success("Related papers retrieved successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Similarity check failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)