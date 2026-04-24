import streamlit as st
from dotenv import load_dotenv

from checker import run_audit
from parser_utils import extract_text_from_pdf
from retrieval import build_or_load_vector_store
from rubric_data import get_coe548_rubric

load_dotenv()

st.set_page_config(page_title="LLM Submission Auditor", layout="wide")
st.title("LLM Submission Auditor")
st.write("Upload a project report PDF and audit it against the course requirements.")

model_name = st.selectbox(
    "Choose Gemini model",
    ["gemini-2.5-flash", "gemini-2.5-pro"],
    index=0
)

uploaded_report = st.file_uploader("Upload the student report (PDF)", type=["pdf"])

if "audit_results" not in st.session_state:
    st.session_state.audit_results = None

if uploaded_report is not None and st.button("Run Audit"):
    try:
        with st.spinner("Extracting report text..."):
            report_text = extract_text_from_pdf(uploaded_report)

        with st.spinner("Building/loading vector database..."):
            vector_store, doc_id = build_or_load_vector_store(report_text)

        with st.spinner("Running criterion-by-criterion audit..."):
            rubric = get_coe548_rubric()
            audit = run_audit(rubric, vector_store, model_name=model_name)
            st.session_state.audit_results = audit
            st.session_state.doc_id = doc_id

        st.success("Audit complete.")

    except Exception as e:
        st.error(f"Audit failed: {e}")

audit = st.session_state.audit_results

if audit:
    pass_count = sum(1 for r in audit.results if r.status == "Pass")
    partial_count = sum(1 for r in audit.results if r.status == "Partial")
    fail_count = sum(1 for r in audit.results if r.status == "Fail")
    unknown_count = sum(1 for r in audit.results if r.status == "Not enough evidence")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pass", pass_count)
    c2.metric("Partial", partial_count)
    c3.metric("Fail", fail_count)
    c4.metric("Not enough evidence", unknown_count)

    st.caption(f"Vector store document ID: {st.session_state.get('doc_id', 'N/A')}")
    st.divider()

    for item in audit.results:
        icon = {
            "Pass": "✅",
            "Partial": "🟡",
            "Fail": "❌",
            "Not enough evidence": "⚪",
        }[item.status]

        st.subheader(f"{icon} {item.criterion_id} — {item.status}")
        st.write(f"**Evidence found:** {item.evidence_found}")
        st.write(f"**Missing / weak:** {item.missing_or_weak}")
        st.write(f"**Improvement:** {item.improvement}")
        st.write(f"**Evidence chunk IDs:** {', '.join(item.evidence_chunk_ids) if item.evidence_chunk_ids else 'None'}")
        st.divider()