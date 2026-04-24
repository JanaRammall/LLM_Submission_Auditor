import streamlit as st
from dotenv import load_dotenv

from parser_utils import extract_text_from_pdf
from rubric_data import get_coe548_rubric
from checker import audit_report

load_dotenv()

st.set_page_config(page_title="COE548 Auditor", layout="wide")
st.title("COE548/748 Submission Auditor")
st.write("Upload a project report PDF and audit it against the course requirements.")

model_name = st.selectbox(
    "Choose Gemini model",
    ["gemini-2.5-flash", "gemini-2.5-pro"],
    index=0
)

uploaded_report = st.file_uploader("Upload the student report (PDF)", type=["pdf"])

if uploaded_report is not None:
    if st.button("Run Audit"):
        with st.spinner("Extracting text from PDF..."):
            report_text = extract_text_from_pdf(uploaded_report)

        with st.spinner("Auditing report with Gemini..."):
            rubric = get_coe548_rubric()
            result = audit_report(report_text, rubric, model_name=model_name)

        st.success("Audit complete.")

        results = result.get("results", [])

        # Summary counts
        pass_count = sum(1 for r in results if r.get("status") == "Pass")
        partial_count = sum(1 for r in results if r.get("status") == "Partial")
        fail_count = sum(1 for r in results if r.get("status") == "Fail")

        col1, col2, col3 = st.columns(3)
        col1.metric("Pass", pass_count)
        col2.metric("Partial", partial_count)
        col3.metric("Fail", fail_count)

        st.divider()

        for r in results:
            status = r.get("status", "Unknown")
            criterion_id = r.get("criterion_id", "N/A")

            if status == "Pass":
                st.subheader(f"✅ {criterion_id} — Pass")
            elif status == "Partial":
                st.subheader(f"🟡 {criterion_id} — Partial")
            else:
                st.subheader(f"❌ {criterion_id} — Fail")

            st.write(f"**Evidence found:** {r.get('evidence_found', '')}")
            st.write(f"**Missing / weak:** {r.get('missing_or_weak', '')}")
            st.write(f"**Improvement:** {r.get('improvement', '')}")
            st.divider()