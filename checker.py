import json
import logging
from typing import Dict, List

from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from config import get_settings
from prompts import criterion_prompt, criterion_query
from retrieval import retrieve_evidence
from rubric_data import Criterion


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CriterionResult(BaseModel):
    criterion_id: str
    status: str = Field(pattern="^(Pass|Partial|Fail|Not enough evidence)$")
    evidence_found: str
    missing_or_weak: str
    improvement: str
    evidence_chunk_ids: List[str]


class AuditReport(BaseModel):
    results: List[CriterionResult]


def get_llm(model_name: str | None = None) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model_name or settings.chat_model,
        api_key="AIzaSyB6Eg4Ug04TgQEYAjE1-JqA14v07UgTKJ4",
        temperature=0.0,
    )


def _serialize_docs(docs: List[Document]) -> str:
    blocks = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "UNKNOWN")
        blocks.append(f"[{chunk_id}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def _safe_json_load(text: str) -> Dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _invoke(llm: ChatGoogleGenerativeAI, prompt: str):
    return llm.invoke(prompt)


def evaluate_single_criterion(llm: ChatGoogleGenerativeAI, vector_store, criterion: Criterion) -> CriterionResult:
    query = criterion_query(criterion)
    docs = retrieve_evidence(vector_store, query=query)
    evidence_text = _serialize_docs(docs)
    prompt = criterion_prompt(criterion, evidence_text)

    response = _invoke(llm, prompt)
    parsed = _safe_json_load(response.content)

    try:
        return CriterionResult(**parsed)
    except ValidationError as e:
        logger.exception("Validation failed for %s", criterion.id)
        return CriterionResult(
            criterion_id=criterion.id,
            status="Not enough evidence",
            evidence_found="Model output could not be validated.",
            missing_or_weak=f"Validation/parsing issue: {str(e)}",
            improvement="Refine the prompt or inspect model output formatting.",
            evidence_chunk_ids=[],
        )


def run_audit(criteria: List[Criterion], vector_store, model_name: str | None = None) -> AuditReport:
    llm = get_llm(model_name=model_name)
    results = []

    for criterion in criteria:
        logger.info("Evaluating %s", criterion.id)
        result = evaluate_single_criterion(llm, vector_store, criterion)
        results.append(result)

    return AuditReport(results=results)