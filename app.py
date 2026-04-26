"""Streamlit entry point and top-level page router.

Workflow:
1. Load environment variables and CSS.
2. Initialize Streamlit session-state slots used across tabs.
3. Render the shared header/navigation.
4. Delegate each tab to ui.pages so this file stay minimal.
"""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ui.pages import (
    render_chatbot_page,
    render_evaluation_page,
    render_novelty_page,
    render_similarity_page,
    render_submission_page,
)

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"
NAV_ITEMS = ["Submission", "Evaluation", "Similarity Check", "Novelty Directions", "Chatbot"]

st.set_page_config(page_title="LLM Submission Auditor", layout="wide")


def load_css(file_name: str = "styles.css") -> None:
    css_path = Path(__file__).parent / file_name
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def init_session_state() -> None:
    defaults = {
        "compiled_rubric": None,
        "audit_report": None,
        "report_features": None,
        "store_id": None,
        "related_papers_result": None,
        "related_papers_status": None,
        "report_text": None,
        "chat_messages": [],
        "analysis_completed": False,
        "analysis_duration_seconds": None,
        "similarity_completed": False,
        "submission_notice": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
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


def render_nav() -> str:
    st.markdown('<div class="top-nav-wrap">', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        NAV_ITEMS,
        horizontal=True,
        label_visibility="collapsed",
        key="top_nav_page",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return page


def main() -> None:
    load_css()
    init_session_state()

    compiled = st.session_state.compiled_rubric
    audit_report = st.session_state.audit_report
    report_features = st.session_state.report_features
    related_papers_result = st.session_state.related_papers_result
    related_papers_status = st.session_state.related_papers_status
    report_text = st.session_state.report_text

    render_header()
    page = render_nav()

    if page == "Submission":
        render_submission_page(
            compiled=compiled,
            related_papers_status=related_papers_status,
            model_name=MODEL_NAME,
        )
    elif page == "Evaluation":
        render_evaluation_page(
            compiled=compiled,
            audit_report=audit_report,
            report_features=report_features,
        )
    elif page == "Similarity Check":
        render_similarity_page(
            report_text=report_text,
            related_papers_status=related_papers_status,
            related_papers_result=related_papers_result,
            model_name=MODEL_NAME,
        )
    elif page == "Novelty Directions":
        render_novelty_page(
            report_text=report_text,
            related_papers_status=related_papers_status,
            related_papers_result=related_papers_result,
            model_name=MODEL_NAME,
        )
    elif page == "Chatbot":
        render_chatbot_page(
            compiled=compiled,
            audit_report=audit_report,
            report_features=report_features,
            related_papers_result=related_papers_result,
            report_text=report_text,
            model_name=MODEL_NAME,
        )


main()
