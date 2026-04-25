import json
import logging
import os
import time
from typing import List, Optional

import requests
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import get_settings
from langchain_google_genai import ChatGoogleGenerativeAI


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class RelatedPaper(BaseModel):
    title: str
    year: Optional[int] = None
    authors: List[str] = Field(default_factory=list)
    venue: Optional[str] = None
    abstract: Optional[str] = None
    url: Optional[str] = None
    similarity_reason: str = ""


class RelatedPaperSearchResult(BaseModel):
    query_title: str
    query_summary: str
    papers: List[RelatedPaper] = Field(default_factory=list)


def get_llm(model_name: Optional[str] = None) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model_name or settings.chat_model,
        api_key=settings.google_api_key,
        temperature=0.0,
    )


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=20),
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


def build_query_extraction_prompt(report_text: str) -> str:
    return f"""
You are helping extract a search query for related academic papers.

From the report text below, extract:
1. project_title
2. short_summary
3. search_query

IMPORTANT:
- search_query should be compact and useful for academic paper search
- include task + method + domain when possible
- do not make it too long
- return ONLY valid JSON in this format:

{{
  "project_title": "...",
  "short_summary": "...",
  "search_query": "..."
}}

Report text:
\"\"\"
{report_text[:12000]}
\"\"\"
"""


def extract_paper_query(report_text: str, model_name: Optional[str] = None) -> dict:
    llm = get_llm(model_name=model_name)
    prompt = build_query_extraction_prompt(report_text)
    response = _invoke(llm, prompt)
    return _safe_json_load(response.content)


def search_semantic_scholar(query: str, limit: int = 5) -> List[dict]:
    """
    Search Semantic Scholar Graph API for related papers.
    Works with or without S2_API_KEY.
    Retries on 429 with backoff and respects low rate limits.
    """
    api_key = "s2k-Iwq4sX1o0K1F2BFKPpqugMcO7FfSJEnOuXuqiI8e"
    logger.info("Semantic Scholar key loaded: %s", bool(api_key))
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": query,
        "limit": limit,
        "fields": "title,year,authors,venue,abstract,url",
    }

    max_attempts = 4
    wait_seconds = 2

    for attempt in range(max_attempts):
        # Respect low rate limits
        time.sleep(1.1)

        response = requests.get(
            SEMANTIC_SCHOLAR_SEARCH_URL,
            params=params,
            headers=headers,
            timeout=30,
        )

        if response.status_code == 200:
            payload = response.json()
            return payload.get("data", [])

        if response.status_code == 429:
            logger.warning(
                "Semantic Scholar rate limited request. Attempt %s/%s",
                attempt + 1,
                max_attempts,
            )
            if attempt < max_attempts - 1:
                time.sleep(wait_seconds)
                wait_seconds *= 2
                continue
            raise requests.HTTPError(
                f"Semantic Scholar rate limit hit after retries: {response.status_code} {response.text}",
                response=response,
            )

        response.raise_for_status()

    return []


def build_similarity_prompt(project_title: str, short_summary: str, candidate_papers: List[dict]) -> str:
    candidates_text = []
    for i, paper in enumerate(candidate_papers, start=1):
        authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])[:4]])
        candidates_text.append(
            f"""
Candidate {i}
Title: {paper.get("title", "")}
Year: {paper.get("year", "")}
Venue: {paper.get("venue", "")}
Authors: {authors}
Abstract: {paper.get("abstract", "")}
URL: {paper.get("url", "")}
""".strip()
        )

    joined = "\n\n".join(candidates_text)

    return f"""
You are ranking related academic papers for a student project.

Project title:
{project_title}

Project summary:
{short_summary}

For each candidate paper, decide whether it is relevant and provide a short similarity reason.

IMPORTANT:
- Keep only papers that are genuinely relevant.
- Prefer task/method/domain overlap.
- Return at most 5 papers.
- Return ONLY valid JSON in this format:

{{
  "papers": [
    {{
      "title": "...",
      "year": 2025,
      "authors": ["A", "B"],
      "venue": "...",
      "abstract": "...",
      "url": "...",
      "similarity_reason": "..."
    }}
  ]
}}

Candidate papers:
\"\"\"
{joined}
\"\"\"
"""


def rank_related_papers(
    project_title: str,
    short_summary: str,
    candidate_papers: List[dict],
    model_name: Optional[str] = None,
) -> List[RelatedPaper]:
    if not candidate_papers:
        return []

    llm = get_llm(model_name=model_name)
    prompt = build_similarity_prompt(project_title, short_summary, candidate_papers)
    response = _invoke(llm, prompt)
    parsed = _safe_json_load(response.content)

    papers = []
    for item in parsed.get("papers", []):
        try:
            papers.append(RelatedPaper(**item))
        except Exception:
            continue

    return papers


def get_related_papers_from_report(
    report_text: str,
    limit: int = 5,
    model_name: Optional[str] = None,
) -> RelatedPaperSearchResult:
    extracted = extract_paper_query(report_text, model_name=model_name)

    project_title = extracted.get("project_title", "Unknown Project")
    short_summary = extracted.get("short_summary", "")
    search_query = extracted.get("search_query", project_title)

    logger.info("Searching related papers with query: %s", search_query)

    # Keep candidate fetch small to reduce throttling
    candidates = search_semantic_scholar(search_query, limit=5)

    ranked = rank_related_papers(
        project_title=project_title,
        short_summary=short_summary,
        candidate_papers=candidates,
        model_name=model_name,
    )

    return RelatedPaperSearchResult(
        query_title=project_title,
        query_summary=short_summary,
        papers=ranked[:limit],
    )