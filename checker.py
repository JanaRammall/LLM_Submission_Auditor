import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict

from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from config import get_settings
from models import RubricCriterion, CriterionResult, AuditReport, CompiledRubric
from retrieval import retrieve_evidence


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_llm(model_name: Optional[str] = None) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model_name or settings.chat_model,
        api_key=settings.google_api_key,
        temperature=0.0,
    )


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _invoke(llm: ChatGoogleGenerativeAI, prompt: str):
    return llm.invoke(prompt)


def _safe_json_load(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("Model did not return valid JSON.")


def _serialize_docs(docs: List[Document]) -> str:
    blocks = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "UNKNOWN")
        artifact_type = doc.metadata.get("artifact_type", "unknown")
        source_name = doc.metadata.get("source_name", "unknown")
        blocks.append(
            f"[{chunk_id}] [artifact_type={artifact_type}] [source={source_name}]\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def make_result(
    criterion_id: str,
    status: str,
    evidence_found: str,
    missing_or_weak: str,
    improvement: str,
    evidence_chunk_ids: Optional[List[str]] = None,
) -> CriterionResult:
    return CriterionResult(
        criterion_id=criterion_id,
        status=status,
        evidence_found=evidence_found,
        missing_or_weak=missing_or_weak,
        improvement=improvement,
        evidence_chunk_ids=evidence_chunk_ids or [],
    )


def evaluate_ieee_and_pages(report_features: Dict) -> CriterionResult:
    page_count = report_features["page_count"]
    status = "Pass" if page_count <= 6 else "Fail"
    return make_result(
        "REP-01",
        status,
        f"Estimated page count: {page_count}. The report structure appears IEEE-like from the parsed text and formatting cues.",
        "" if status == "Pass" else f"Estimated page count exceeds the 6-page limit: {page_count}.",
        "Keep the report in IEEE-style format and ensure it stays within 6 pages, including references.",
    )


def evaluate_literature_quality(report_features: Dict) -> CriterionResult:
    ref_count = report_features["reference_count"]
    recent_count = report_features["recent_reference_count_2025_2026"]
    has_related = report_features["sections"].get("related work", False)

    if has_related and ref_count >= 10 and recent_count >= 10:
        status = "Pass"
    elif has_related and ref_count >= 10:
        status = "Partial"
    else:
        status = "Fail"

    return make_result(
        "REP-02",
        status,
        f"Related Work detected: {has_related}. Reference count: {ref_count}. References from 2025–2026: {recent_count}.",
        "" if status == "Pass" else "The literature review is missing, too small, or lacks enough recent references from 2025–2026.",
        "Ensure a clear Related Work section, include at least 10 references, and strengthen the review with 2025–2026 papers.",
    )


def evaluate_core_sections(report_features: Dict) -> CriterionResult:
    required = [
        "abstract",
        "introduction",
        "related work",
        "methodology",
        "experimental setup",
        "results and discussion",
        "conclusion and future work",
        "references",
    ]
    sections = report_features["sections"]
    missing = [sec for sec in required if not sections.get(sec, False)]

    if not missing:
        status = "Pass"
    elif len(missing) <= 2:
        status = "Partial"
    else:
        status = "Fail"

    return make_result(
        "REP-03",
        status,
        f"Detected sections: {', '.join([sec for sec in required if sections.get(sec, False)])}.",
        "" if status == "Pass" else f"Missing sections: {', '.join(missing)}.",
        "Add clearly labeled core sections to complete the report structure.",
    )


def build_small_semantic_prompt(criteria: List[RubricCriterion], evidence_text: str, group_name: str) -> str:
    criteria_block = "\n".join(
        [f"- {c.criterion_id}: {c.title} | {c.description}" for c in criteria]
    )

    return f"""
You are a strict but fair university project auditor.

Evaluate the following criteria for the group: {group_name}

IMPORTANT:
- Use ONLY the evidence provided.
- Do not mix up criterion IDs.
- Return exactly one result for each criterion below.
- If evidence clearly supports the criterion, use Pass.
- If evidence clearly shows the requirement is not met, use Fail.
- If evidence is weak or incomplete, use Partial.
- If there is not enough explicit evidence, use Not enough evidence.

Return ONLY valid JSON in this format:
{{
  "results": [
    {{
      "criterion_id": "IMP-01",
      "status": "Pass",
      "evidence_found": "short grounded summary",
      "missing_or_weak": "what is missing or unclear",
      "improvement": "one concrete improvement",
      "evidence_chunk_ids": ["R000", "R001"]
    }}
  ]
}}

Criteria:
{criteria_block}

Evidence:
\"\"\"
{evidence_text}
\"\"\"
"""


def run_semantic_batch(
    llm: ChatGoogleGenerativeAI,
    vector_store,
    criteria: List[RubricCriterion],
    group_name: str,
) -> List[CriterionResult]:
    if not criteria:
        return []

    query = " ".join([f"{c.title}. {c.description}" for c in criteria])
    docs = retrieve_evidence(vector_store, query=query, k=5)
    evidence_text = _serialize_docs(docs)
    prompt = build_small_semantic_prompt(criteria, evidence_text, group_name)

    response = _invoke(llm, prompt)
    parsed = _safe_json_load(response.content)
    raw_results = parsed.get("results", [])

    results: List[CriterionResult] = []
    seen = set()

    for item in raw_results:
        try:
            result = CriterionResult(**item)
            if not result.evidence_chunk_ids:
                result.evidence_chunk_ids = [
                    d.metadata.get("chunk_id", "")
                    for d in docs
                    if d.metadata.get("chunk_id")
                ]
            results.append(result)
            seen.add(result.criterion_id)
        except ValidationError:
            continue

    for c in criteria:
        if c.criterion_id not in seen:
            results.append(
                make_result(
                    c.criterion_id,
                    "Not enough evidence",
                    "",
                    f"No valid semantic result returned for {c.criterion_id}.",
                    "Inspect prompt output or rerun the audit.",
                )
            )

    return results


def run_audit(
    compiled_rubric: CompiledRubric,
    vector_store,
    report_features: Dict,
    available_artifacts: Dict[str, bool],
    model_name: Optional[str] = None,
) -> AuditReport:
    active_results: List[CriterionResult] = []
    deferred_results: List[CriterionResult] = []

    # deterministic compact report checks
    active_results.append(evaluate_ieee_and_pages(report_features))
    active_results.append(evaluate_literature_quality(report_features))
    active_results.append(evaluate_core_sections(report_features))

    # semantic groups
    impl_criteria = [c for c in compiled_rubric.criteria if c.criterion_id in {
        "IMP-01", "IMP-02", "IMP-03", "IMP-04", "IMP-05", "IMP-06", "IMP-07"
    }]
    report_semantic = [c for c in compiled_rubric.criteria if c.criterion_id in {
        "REP-04", "REP-05"
    }]

    semantic_jobs = [
        (impl_criteria, "Implementation"),
        (report_semantic, "Report Semantics"),
    ]
    semantic_jobs = [(criteria, group_name) for criteria, group_name in semantic_jobs if criteria]

    if semantic_jobs:
        max_workers = min(2, len(semantic_jobs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_semantic_batch,
                    get_llm(model_name=model_name),
                    vector_store,
                    criteria,
                    group_name,
                ): group_name
                for criteria, group_name in semantic_jobs
            }

            for future in as_completed(futures):
                active_results.extend(future.result())

    # preserve compact rubric order
    order_map = {c.criterion_id: i for i, c in enumerate(compiled_rubric.criteria)}
    active_results.sort(key=lambda r: order_map.get(r.criterion_id, 10**9))

    return AuditReport(
        active_results=active_results,
        deferred_results=deferred_results,
    )
