"""Streamlit page renderers for the submission auditor app.

Each public render_* function corresponds to one navigation tab. The heavy
analysis and model calls stay in services/, while this module focuses on:
- reading Streamlit session state,
- formatting cards and controls,
- calling service functions when the user clicks an action.
"""

import html

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from rag.retrieval import load_vector_store
from services.analysis_pipeline import (
    cached_related_papers,
    prepare_similarity_analysis,
    run_evaluation_analysis,
)
from services.chat_service import answer_chat_question, serialize_audit_context
from utils.formatting import format_ai_content, format_chat_html


def render_submission_page(compiled, related_papers_status, model_name: str) -> None:
    st.markdown('<div class="section-title">Upload Submission</div>', unsafe_allow_html=True)

    if st.session_state.get("submission_notice") and st.session_state.get("analysis_completed"):
        st.success(st.session_state.submission_notice)

    if compiled:
        criteria_count = len(compiled.criteria)
        store_id = html.escape(st.session_state.get("store_id", "N/A") or "N/A")
        project_title = html.escape(compiled.project_title)
        related_status = html.escape(related_papers_status or "not ready")
        duration = st.session_state.get("analysis_duration_seconds")
        completion_badges = []
        duration_html = ""
        if st.session_state.get("analysis_completed"):
            completion_badges.append('<span class="submission-complete">Evaluation complete</span>')
            if duration is not None:
                duration_html = f" &nbsp;|&nbsp; Finished in: {html.escape(str(duration))}s"
        if st.session_state.get("similarity_completed"):
            completion_badges.append('<span class="submission-complete">Similarity ready</span>')
        completion_html = "".join(completion_badges)

        st.markdown(
            f"""
<div class="submission-status">
    <div>
        <div class="submission-kicker">Current analysis</div>
        <div class="submission-title">{project_title}</div>
        <div class="submission-meta">Criteria: {criteria_count} &nbsp;|&nbsp; Vector store: {store_id} &nbsp;|&nbsp; Related papers: {related_status}{duration_html}</div>
        <div class="submission-badge-row">{completion_html}</div>
    </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Upload new files and click Run Analysis to replace the current analysis.")

    rubric_source = st.radio(
        "Rubric source",
        ["Use instructor rubric", "Upload custom rubric"],
        horizontal=True,
        key="rubric_source",
    )
    use_instructor_rubric = rubric_source == "Use instructor rubric"

    if use_instructor_rubric:
        uploaded_report = st.file_uploader(
            "Student report PDF",
            type=["pdf"],
            key="report_uploader",
        )
        uploaded_instructions = None
    else:
        upload_col1, upload_col2 = st.columns(2)
        with upload_col1:
            uploaded_instructions = st.file_uploader(
                "Project instructions / rubric PDF",
                type=["pdf"],
                key="instructions_uploader",
            )

        with upload_col2:
            uploaded_report = st.file_uploader(
                "Student report PDF",
                type=["pdf"],
                key="report_uploader",
            )

    run_clicked = st.button("Run Analysis")

    can_run = uploaded_report is not None and (use_instructor_rubric or uploaded_instructions is not None)

    if can_run and run_clicked:
        try:
            status = st.status("Running analysis...", expanded=True)
            with status:
                st.write("Preparing the rubric and report text.")
                run_evaluation_analysis(
                    uploaded_instructions,
                    uploaded_report,
                    model_name=model_name,
                    use_instructor_rubric=use_instructor_rubric,
                )
                st.write(st.session_state.submission_notice)
                status.update(label=st.session_state.submission_notice, state="complete", expanded=True)
                st.success(st.session_state.submission_notice)

                st.write("Preparing similarity results.")
                prepare_similarity_analysis(
                    report_text=st.session_state.report_text,
                    model_name=model_name,
                )
                if st.session_state.get("similarity_completed"):
                    st.write("Similarity results are ready.")
                else:
                    st.write("Similarity results were not prepared successfully.")
            status.update(label=st.session_state.get("submission_notice", "Analysis complete."), state="complete", expanded=True)
            st.rerun()
        except Exception as e:
            st.session_state.submission_notice = ""
            st.error(f"Process failed: {e}")
    elif run_clicked:
        st.warning("Upload the required PDF file(s) before running analysis.")


def render_evaluation_page(compiled, audit_report, report_features) -> None:
    if not compiled and not audit_report and not report_features:
        st.info("Run analysis from the Submission tab first.")

    if compiled:
        st.markdown('<div class="section-title">Compact Rubric</div>', unsafe_allow_html=True)
        st.caption(f"Criteria used: {len(compiled.criteria)}")

        with st.expander("View compact criteria", expanded=False):
            for criterion in compiled.criteria:
                st.markdown(f"**{criterion.criterion_id} - {criterion.title}**")
                st.write(criterion.description)
                st.divider()

    if report_features:
        render_report_features(report_features)

    if audit_report:
        render_audit_results(compiled, audit_report)


def render_report_features(report_features) -> None:
    st.markdown('<div class="section-title">Deterministic Report Analysis</div>', unsafe_allow_html=True)

    sections = report_features.get("sections", {})
    detected_sections = [name for name, present in sections.items() if present]
    missing_sections = [name for name, present in sections.items() if not present]
    reference_years = report_features.get("reference_years_found", [])
    recent_refs = report_features.get("recent_reference_count_2025_2026", 0)
    reference_count = report_features.get("reference_count", 0)
    page_count = report_features.get("page_count", 0)
    has_references = report_features.get("has_references_block", False)

    section_items_html = "".join(
        f'<span class="analysis-chip analysis-chip-pass">{html.escape(name.title())}</span>'
        for name in detected_sections
    ) or '<span class="analysis-chip analysis-chip-muted">No sections detected</span>'

    missing_items_html = "".join(
        f'<span class="analysis-chip analysis-chip-warn">{html.escape(name.title())}</span>'
        for name in missing_sections
    ) or '<span class="analysis-chip analysis-chip-pass">No missing sections</span>'

    unique_years = sorted(set(reference_years), reverse=True)[:12]
    years_html = "".join(
        f'<span class="analysis-chip">{year}</span>'
        for year in unique_years
    ) or '<span class="analysis-chip analysis-chip-muted">No years found</span>'

    references_status = "Detected" if has_references else "Not detected"

    st.markdown(
        f"""
<div class="analysis-panel">
    <div class="analysis-metric-grid">
        <div class="analysis-metric">
            <div class="analysis-metric-label">Estimated pages</div>
            <div class="analysis-metric-value">{page_count}</div>
        </div>
        <div class="analysis-metric">
            <div class="analysis-metric-label">References</div>
            <div class="analysis-metric-value">{reference_count}</div>
        </div>
        <div class="analysis-metric">
            <div class="analysis-metric-label">Recent refs 2025-2026</div>
            <div class="analysis-metric-value">{recent_refs}</div>
        </div>
        <div class="analysis-metric">
            <div class="analysis-metric-label">References block</div>
            <div class="analysis-metric-value analysis-metric-text">{references_status}</div>
        </div>
    </div>
    <div class="analysis-section">
        <div class="analysis-label">Detected sections</div>
        <div class="analysis-chip-row">{section_items_html}</div>
    </div>
    <div class="analysis-section">
        <div class="analysis-label">Missing / not detected</div>
        <div class="analysis-chip-row">{missing_items_html}</div>
    </div>
    <div class="analysis-section">
        <div class="analysis-label">Reference years found</div>
        <div class="analysis-chip-row">{years_html}</div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("View raw analysis data", expanded=False):
        st.json(report_features)


def render_audit_results(compiled, audit_report) -> None:
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


def render_similarity_page(report_text, related_papers_status, related_papers_result, model_name: str) -> None:
    st.markdown('<div class="section-title">Similarity Check</div>', unsafe_allow_html=True)

    if not report_text:
        st.info("Run the analysis first so the system can save the report and prepare related papers.")
        return

    if related_papers_status == "ready" and related_papers_result:
        render_related_papers(related_papers_result)
    elif related_papers_status == "failed":
        st.warning("Related papers were not prepared successfully in the previous run.")
        retry_related_papers_button("Retry Similarity Check", report_text, model_name)
    else:
        st.info("Related papers are not prepared yet.")
        retry_related_papers_button("Run Similarity Check Now", report_text, model_name)


def render_related_papers(related_papers_result) -> None:
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
    st.caption(
        "This score measures semantic relatedness between the submitted report and candidate academic papers. "
        "It is not a plagiarism score."
    )

    if not related_papers_result.papers:
        st.info("No strong related papers were found.")
        return

    st.markdown('<div class="paper-list">', unsafe_allow_html=True)
    for paper in related_papers_result.papers:
        render_related_paper_card(paper)
    st.markdown("</div>", unsafe_allow_html=True)


def render_related_paper_card(paper) -> None:
    title = html.escape(paper.title or "Untitled Paper")
    year = html.escape(str(paper.year) if paper.year else "N/A")
    venue = html.escape(paper.venue or "N/A")
    authors = html.escape(", ".join(paper.authors) if paper.authors else "N/A")
    reason = html.escape(paper.similarity_reason or "No similarity reason available.")
    abstract = html.escape(paper.abstract or "No abstract available.")
    url = paper.url or ""
    score = html.escape(f"{getattr(paper, 'similarity_score', 0.0):.1f}")
    confidence = html.escape(getattr(paper, "confidence", "Medium"))
    matched_chunks = getattr(paper, "matched_report_chunks", []) or []
    matched_chunks_html = "".join(
        f'<div class="matched-chunk">{html.escape(chunk)}</div>'
        for chunk in matched_chunks
    ) or '<div class="matched-chunk">No matched chunks available.</div>'

    link_html = f'<a class="paper-link" href="{html.escape(url)}" target="_blank">Open paper</a>' if url else ""

    st.markdown(
        f"""
<div class="paper-card paper-card-full">
    <div class="paper-card-topline">
        <span class="paper-year">{year}</span>
        <span class="paper-venue">{venue}</span>
        <span class="paper-score">Content Similarity: {score}%</span>
        <span class="paper-confidence">Confidence: {confidence}</span>
    </div>
    <div class="paper-title">{title}</div>
    <div class="paper-meta">{year} - {venue}</div>
    <div class="paper-reason-label">Why it is similar</div>
    <div class="paper-reason">{reason}</div>
    <div class="meta-pill-row">
        <span class="meta-pill">Authors: {authors}</span>
    </div>
    <details class="paper-abstract-details">
        <summary>Abstract</summary>
        <div class="paper-abstract">{abstract}</div>
    </details>
    <details class="paper-abstract-details">
        <summary>Matched report chunks</summary>
        <div class="matched-chunks">{matched_chunks_html}</div>
    </details>
    <div class="link-line">{link_html}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def retry_related_papers_button(label: str, report_text: str, model_name: str) -> None:
    if st.button(label):
        try:
            with st.spinner("Finding related papers..."):
                related_papers_result = cached_related_papers(
                    report_text=report_text,
                    model_name=model_name,
                    limit=5,
                )
                st.session_state.related_papers_result = related_papers_result
                st.session_state.related_papers_status = "ready"
            st.success("Related papers retrieved successfully.")
            st.rerun()
        except Exception as e:
            st.error(f"Related-paper analysis failed: {e}")


def render_novelty_page(report_text, related_papers_status, related_papers_result, model_name: str) -> None:
    st.markdown('<div class="section-title">Novelty Direction Recommender</div>', unsafe_allow_html=True)

    if not report_text:
        st.info("Run the evaluation first so the system can prepare related-paper evidence.")
    elif related_papers_status == "ready" and related_papers_result:
        novelty_analysis = getattr(related_papers_result, "novelty_analysis", None)

        if not novelty_analysis:
            st.warning("Novelty directions were not generated for this run.")
        else:
            render_novelty_analysis(related_papers_result, novelty_analysis)
    elif related_papers_status == "failed":
        st.warning("Related-paper analysis failed in the previous run.")
        retry_related_papers_button("Retry Novelty Analysis", report_text, model_name)
    else:
        st.info("Novelty directions are prepared after related-paper analysis.")
        retry_related_papers_button("Run Novelty Analysis Now", report_text, model_name)


def render_novelty_analysis(related_papers_result, novelty_analysis) -> None:
    query_title = html.escape(related_papers_result.query_title or "Submitted Project")
    query_summary = html.escape(related_papers_result.query_summary or "No summary available.")
    crowded_topics = getattr(novelty_analysis, "crowded_topics", []) or []
    crowded_topics_html = "".join(
        f'<span class="novelty-chip">{html.escape(topic)}</span>'
        for topic in crowded_topics
    ) or '<span class="novelty-chip">No crowded topics identified.</span>'

    st.markdown(
        f"""
<div class="novelty-panel">
    <div class="novelty-kicker">Research Opportunity Analysis</div>
    <div class="novelty-title">{query_title}</div>
    <div class="novelty-note">{query_summary}</div>
    <div class="novelty-note">Based on the retrieved related papers, these areas appear crowded or promising. This is advisory, not a claim about the entire research literature.</div>
    <div class="novelty-label">Crowded topics</div>
    <div class="novelty-chip-row">{crowded_topics_html}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    for direction in getattr(novelty_analysis, "novelty_directions", []) or []:
        render_novelty_card(direction)


def render_novelty_card(direction) -> None:
    direction_title = html.escape(direction.direction)
    why_promising = html.escape(direction.why_promising)
    how_to_extend = html.escape(direction.how_to_extend)
    expected_contribution = html.escape(direction.expected_contribution)

    st.markdown(
        f"""
<div class="novelty-card">
    <div class="novelty-card-title">{direction_title}</div>
    <div class="novelty-field">
        <div class="novelty-label">Why promising</div>
        <div class="novelty-text">{why_promising}</div>
    </div>
    <div class="novelty-field">
        <div class="novelty-label">How to extend</div>
        <div class="novelty-text">{how_to_extend}</div>
    </div>
    <div class="novelty-field">
        <div class="novelty-label">Expected contribution</div>
        <div class="novelty-text">{expected_contribution}</div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_chatbot_page(
    compiled,
    audit_report,
    report_features,
    related_papers_result,
    report_text,
    model_name: str,
) -> None:
    st.markdown(
        """
<div class="chat-page">
    <div class="chat-header">
        <div>
            <div class="chat-kicker">Paper Chatbot</div>
            <div class="chat-title">Ask about the submission</div>
            <div class="chat-subtitle">Answers are grounded in the uploaded report, rubric results, and related-paper analysis.</div>
        </div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    if not report_text or not audit_report or not st.session_state.get("store_id"):
        st.info("Run the evaluation first so the chatbot can use the submitted paper and audit results.")
        return

    render_chat_transcript()
    render_chat_input(compiled, audit_report, report_features, related_papers_result, model_name)


def render_chat_transcript() -> None:
    transcript_parts = ['<div class="chat-transcript">']

    if not st.session_state.chat_messages:
        transcript_parts.append(
            """
<div class="chat-empty">
    <div class="chat-row chat-row-assistant">
        <div class="chat-bubble chat-bubble-assistant chat-bubble-intro">
            <div class="chat-role">Auditor Assistant</div>
            <div class="chat-text">Ask me about the submitted paper, rubric evidence, missing requirements, similarity results, or improvement ideas.</div>
        </div>
    </div>
    <div class="chat-suggestion-panel">
        <div class="chat-empty-title">Try one of these</div>
        <div class="chat-suggestions">
            <div class="chat-suggestion">What is the paper about?</div>
            <div class="chat-suggestion">Which criteria need work?</div>
            <div class="chat-suggestion">What evidence supports IMP-01?</div>
            <div class="chat-suggestion">How can the submission improve?</div>
        </div>
    </div>
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


def render_chat_input(compiled, audit_report, report_features, related_papers_result, model_name: str) -> None:
    with st.form("chatbot_question_form", clear_on_submit=True):
        input_col, send_col = st.columns([7, 1])
        with input_col:
            prompt = st.text_input(
                "Ask about the submitted paper or evaluation",
                placeholder="Ask about the submitted paper, rubric results, or related papers",
                label_visibility="collapsed",
            )
        with send_col:
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
                    model_name=model_name,
                )
            except Exception as e:
                answer = f"I could not answer that right now because the chatbot failed: {e}"

        st.session_state.chat_messages.append(AIMessage(content=answer))
        st.rerun()
