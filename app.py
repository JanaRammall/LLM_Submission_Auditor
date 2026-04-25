import hashlib
from pathlib import Path
import html

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from checker import get_llm, run_audit
from retrieval import chunk_artifact_text, build_or_load_vector_store_from_docs, load_vector_store, retrieve_evidence
from rubric_compiler import compile_rubric
from report_analyzer import build_report_features
from compact_rubric import build_compact_rubric
from related_papers import get_related_papers_from_report

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"

st.set_page_config(page_title="LLM Submission Auditor", layout="wide")


def shorten_text(text: str, max_len: int = 180) -> str:
    if not text:
        return "None"
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def format_ai_content(content) -> str:
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


def format_chat_html(text: str) -> str:
    return html.escape(text or "").replace("\n", "<br>")


def serialize_audit_context(compiled_rubric, audit_report, report_features, related_papers_result) -> str:
    blocks = []

    if compiled_rubric:
        blocks.append(f"Project title: {compiled_rubric.project_title}")
        blocks.append(
            "Rubric criteria:\n"
            + "\n".join(
                f"- {criterion.criterion_id}: {criterion.title} | {criterion.description}"
                for criterion in compiled_rubric.criteria
            )
        )

    if report_features:
        blocks.append(f"Deterministic report features: {report_features}")

    if audit_report:
        results = []
        for result in audit_report.active_results:
            results.append(
                "\n".join(
                    [
                        f"- {result.criterion_id}: {result.status}",
                        f"  Evidence: {result.evidence_found or 'None'}",
                        f"  Missing or weak: {result.missing_or_weak or 'None'}",
                        f"  Improvement: {result.improvement or 'None'}",
                        f"  Evidence chunks: {', '.join(result.evidence_chunk_ids) if result.evidence_chunk_ids else 'None'}",
                    ]
                )
            )
        blocks.append("Evaluation results:\n" + "\n".join(results))

    if related_papers_result:
        papers = []
        for paper in related_papers_result.papers[:5]:
            papers.append(
                f"- {paper.title or 'Untitled'} ({paper.year or 'N/A'}, {paper.venue or 'N/A'}): "
                f"{paper.similarity_reason or 'No similarity reason available.'}"
            )
        blocks.append("Related papers:\n" + ("\n".join(papers) if papers else "None found."))

    return "\n\n".join(blocks)


def serialize_retrieved_docs(docs) -> str:
    blocks = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "UNKNOWN")
        source = doc.metadata.get("source_name", "unknown")
        blocks.append(f"[{chunk_id}] [source={source}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def answer_chat_question(prompt: str, chat_history: list, vector_store, app_context: str) -> str:
    docs = retrieve_evidence(vector_store, query=prompt, k=5)
    paper_context = serialize_retrieved_docs(docs)

    history_lines = []
    for message in chat_history[-8:]:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        history_lines.append(f"{role}: {format_ai_content(message.content)}")

    messages = [
        SystemMessage(
            content=(
                "You are a careful assistant for a university submission audit app. "
                "Answer questions about the submitted paper and the evaluation results. "
                "Use only the provided paper chunks, rubric, evaluation, deterministic features, "
                "and related-paper context. If the answer is not supported by that context, say so. "
                "When useful, cite chunk IDs or criterion IDs."
            )
        ),
        HumanMessage(
            content=f"""
Conversation so far:
{chr(10).join(history_lines) if history_lines else "No previous messages."}

Current user question:
{prompt}

Relevant submitted-paper chunks:
{paper_context or "No matching chunks were found."}

Evaluation and app context:
{app_context or "No evaluation context is available."}
"""
        ),
    ]

    response = get_llm(model_name=MODEL_NAME).invoke(messages)
    return format_ai_content(response.content)


def load_css(file_name: str = "styles.css") -> None:
    css_path = Path(__file__).parent / file_name
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


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
    "chat_messages": [],
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def run_full_analysis(uploaded_instructions, uploaded_report):
    st.session_state.related_papers_result = None
    st.session_state.related_papers_status = None
    st.session_state.chat_messages = []

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

# Header
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

# Navbar
st.markdown('<div class="top-nav-wrap">', unsafe_allow_html=True)
page = st.radio(
    "Navigation",
    ["Evaluation", "Similarity Check", "Chatbot"],
    horizontal=True,
    label_visibility="collapsed",
    key="top_nav_page",
)
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Evaluation Page
# =========================
if page == "Evaluation":
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
        st.markdown('<div class="section-title">Compact Rubric</div>', unsafe_allow_html=True)
        st.write(f"**Project Title:** {compiled.project_title}")
        st.caption(f"Criteria used: {len(compiled.criteria)}")

        with st.expander("View compact criteria", expanded=False):
            for criterion in compiled.criteria:
                st.markdown(f"**{criterion.criterion_id} — {criterion.title}**")
                st.write(criterion.description)
                st.divider()

    if report_features:
        st.markdown('<div class="section-title">Deterministic Report Analysis</div>', unsafe_allow_html=True)
        st.json(report_features)

    if audit_report:
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

        criteria_map = {c.criterion_id: c for c in compiled.criteria} if compiled else {}

        for item in audit_report.active_results:
            criterion = criteria_map.get(item.criterion_id)
            title = criterion.title if criterion else item.criterion_id

            status_class = {
                "Pass": "status-pass",
                "Partial": "status-partial",
                "Fail": "status-fail",
                "Not enough evidence": "status-unknown",
            }[item.status]

            full_evidence = html.escape(item.evidence_found or "None")
            full_missing = html.escape(item.missing_or_weak or "None")
            full_improvement = html.escape(item.improvement or "None")
            chunk_ids = html.escape(", ".join(item.evidence_chunk_ids) if item.evidence_chunk_ids else "None")
            safe_title = html.escape(title)
            safe_status = html.escape(item.status)
            safe_criterion_id = html.escape(item.criterion_id)

            card_html = (
                '<details class="eval-card eval-details">'
                '<summary class="eval-summary">'
                '<div class="eval-card-header">'
                '<div class="eval-heading">'
                f'<div class="eval-card-id">{safe_criterion_id}</div>'
                f'<div class="eval-card-title">{safe_title}</div>'
                '</div>'
                f'<div class="status-pill {status_class}">{safe_status}</div>'
                '</div>'
                '<div class="eval-label">Evidence</div>'
                f'<div class="eval-preview">{full_evidence}</div>'
                '<div class="eval-label">Improvement</div>'
                f'<div class="eval-preview">{full_improvement}</div>'
                '</summary>'
                '<div class="eval-full">'
                f'<p><strong>Missing / weak:</strong> {full_missing}</p>'
                f'<p><strong>Evidence chunk IDs:</strong> {chunk_ids}</p>'
                '</div>'
                '</details>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Related Papers Preview</div>', unsafe_allow_html=True)

        if related_papers_status == "ready" and related_papers_result:
            if not related_papers_result.papers:
                st.info("No strong related papers were found.")
            else:
                for paper in related_papers_result.papers[:2]:
                    title = html.escape(paper.title or "Untitled Paper")
                    venue = html.escape(paper.venue or "N/A")
                    year = html.escape(str(paper.year) if paper.year else "N/A")
                    reason = html.escape(paper.similarity_reason or "No similarity note available.")

                    st.markdown(
                        f"""
<div class="paper-card paper-card-compact">
    <div class="paper-card-topline">
        <span class="paper-year">{year}</span>
        <span class="paper-venue">{venue}</span>
    </div>
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

# Similarity page
elif page == "Similarity Check":
    st.markdown('<div class="section-title">Similarity Check</div>', unsafe_allow_html=True)

    if not report_text:
        st.info("Run the analysis first so the system can save the report and prepare related papers.")
    else:
        if related_papers_status == "ready" and related_papers_result:
            query_title = html.escape(related_papers_result.query_title or "Submitted Project")
            query_summary = html.escape(related_papers_result.query_summary or "No summary available.")

            st.markdown(
                f"""
<div class="paper-query-panel">
    <div>
        <div class="paper-query-label">Search Basis</div>
        <div class="paper-query-title">{query_title}</div>
        <div class="paper-query-summary">{query_summary}</div>
    </div>
    <div class="paper-query-count">{len(related_papers_result.papers)} papers</div>
</div>
                """,
                unsafe_allow_html=True,
            )

            if not related_papers_result.papers:
                st.info("No strong related papers were found.")
            else:
                st.markdown('<div class="paper-list">', unsafe_allow_html=True)
                for paper in related_papers_result.papers:
                    title = html.escape(paper.title or "Untitled Paper")
                    year = html.escape(str(paper.year) if paper.year else "N/A")
                    venue = html.escape(paper.venue or "N/A")
                    authors = html.escape(", ".join(paper.authors) if paper.authors else "N/A")
                    reason = html.escape(paper.similarity_reason or "No similarity reason available.")
                    abstract = html.escape(paper.abstract or "No abstract available.")
                    url = paper.url or ""

                    link_html = f'<a class="paper-link" href="{html.escape(url)}" target="_blank">Open paper</a>' if url else ""

                    st.markdown(
                        f"""
<div class="paper-card paper-card-full">
    <div class="paper-card-topline">
        <span class="paper-year">{year}</span>
        <span class="paper-venue">{venue}</span>
    </div>
    <div class="paper-title">{title}</div>
    <div class="paper-meta">{year} • {venue}</div>
    <div class="paper-reason-label">Why it is similar</div>
    <div class="paper-reason">{reason}</div>
    <div class="meta-pill-row">
        <span class="meta-pill">Authors: {authors}</span>
    </div>
    <details class="paper-abstract-details">
        <summary>Abstract</summary>
        <div class="paper-abstract">{abstract}</div>
    </details>
    <div class="link-line">{link_html}</div>
</div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

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

# Chatbot page
elif page == "Chatbot":
    st.markdown(
        """
<div class="chat-page">
    <div class="chat-header">
        <div>
            <div class="chat-kicker">Paper Chatbot</div>
            <div class="chat-title">Ask about the submission and evaluation</div>
            <div class="chat-subtitle">Grounded in the uploaded paper chunks, rubric results, and related-paper analysis.</div>
        </div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    if not report_text or not audit_report or not st.session_state.get("store_id"):
        st.info("Run the evaluation first so the chatbot can use the submitted paper and audit results.")
    else:
        transcript_parts = ['<div class="chat-transcript">']

        if not st.session_state.chat_messages:
            transcript_parts.append(
                """
<div class="chat-empty">
    <div class="chat-empty-title">No messages yet</div>
    <div class="chat-empty-copy">Try asking for the paper title, whether it passed, what is missing, or which rubric item needs the most work.</div>
</div>
                """
            )

        for message in st.session_state.chat_messages:
            if isinstance(message, HumanMessage):
                transcript_parts.append(
                    f"""
<div class="chat-row chat-row-user">
    <div class="chat-bubble chat-bubble-user">
        <div class="chat-role">You</div>
        <div class="chat-text">{format_chat_html(message.content)}</div>
    </div>
</div>
                    """
                )
            elif isinstance(message, AIMessage):
                transcript_parts.append(
                    f"""
<div class="chat-row chat-row-assistant">
    <div class="chat-bubble chat-bubble-assistant">
        <div class="chat-role">Auditor Assistant</div>
        <div class="chat-text">{format_chat_html(format_ai_content(message.content))}</div>
    </div>
</div>
                    """
                )

        transcript_parts.append("</div>")
        st.markdown("".join(transcript_parts), unsafe_allow_html=True)

        with st.form("chatbot_question_form", clear_on_submit=True):
            prompt = st.text_input(
                "Ask about the submitted paper or evaluation",
                placeholder="Ask about the submitted paper or evaluation",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send")

        if submitted and prompt:
            user_message = HumanMessage(content=prompt)
            st.session_state.chat_messages.append(user_message)

            with st.spinner("Searching the paper and evaluation..."):
                try:
                    vector_store = load_vector_store(st.session_state.store_id)
                    app_context = serialize_audit_context(
                        compiled_rubric=compiled,
                        audit_report=audit_report,
                        report_features=report_features,
                        related_papers_result=related_papers_result,
                    )
                    answer = answer_chat_question(
                        prompt=prompt,
                        chat_history=st.session_state.chat_messages[:-1],
                        vector_store=vector_store,
                        app_context=app_context,
                    )
                except Exception as e:
                    answer = f"I could not answer that right now because the chatbot failed: {e}"

            st.session_state.chat_messages.append(AIMessage(content=answer))
            st.rerun()
