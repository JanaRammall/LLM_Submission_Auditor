"""Find related academic papers and suggest novelty directions for a report.

Pipeline:
1. Extract an academic search basis from the submitted report.
2. Query Semantic Scholar for candidate papers.
3. Remove self-matches and duplicate papers.
4. Compare candidate title/abstract embeddings against report chunks.
5. Explain semantic relatedness and generate novelty directions.

The score is semantic relatedness, not plagiarism detection.
"""

import logging
import math
import re
import time
from typing import List, Optional

import requests
from pydantic import BaseModel, Field

from core.config import get_settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.llm import get_embeddings, get_llm, invoke_with_retry
from utils.json_utils import safe_json_load


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
    similarity_score: float = 0.0
    confidence: str = "Medium"
    matched_report_chunks: List[str] = Field(default_factory=list)
    similarity_reason: str = ""


class NoveltyDirection(BaseModel):
    direction: str
    why_promising: str
    how_to_extend: str
    expected_contribution: str


class NoveltyAnalysis(BaseModel):
    crowded_topics: List[str] = Field(default_factory=list)
    novelty_directions: List[NoveltyDirection] = Field(default_factory=list)


class RelatedPaperSearchResult(BaseModel):
    query_title: str
    query_summary: str
    papers: List[RelatedPaper] = Field(default_factory=list)
    novelty_analysis: Optional[NoveltyAnalysis] = None


def build_query_extraction_prompt(report_text: str) -> str:
    return f"""
You are helping extract a search query for related academic papers.

From the report text below, extract:
1. project_title
2. short_summary
3. keywords
4. search_queries

IMPORTANT:
- search_queries should be compact and useful for academic paper search
- include task + method + domain when possible
- return 2 or 3 search queries, not more
- keywords should include 5 to 8 useful academic terms
- do not use only the exact project title as a search query
- return ONLY valid JSON in this format:

{{
  "project_title": "...",
  "short_summary": "...",
  "keywords": ["...", "..."],
  "search_queries": ["...", "..."]
}}

Report text:
\"\"\"
{report_text[:12000]}
\"\"\"
"""


def extract_paper_query(report_text: str, model_name: Optional[str] = None) -> dict:
    llm = get_llm(model_name=model_name)
    prompt = build_query_extraction_prompt(report_text)
    response = invoke_with_retry(llm, prompt)
    return safe_json_load(response.content)


def search_semantic_scholar(query: str, limit: int = 8) -> List[dict]:
    """
    Search Semantic Scholar Graph API for related papers.
    Works with or without S2_API_KEY.
    Retries on 429 with backoff and respects low rate limits.
    """
    settings = get_settings()
    logger.info("Semantic Scholar key loaded: %s", bool(settings.s2_api_key))
    headers = {}
    if settings.s2_api_key:
        headers["x-api-key"] = settings.s2_api_key

    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,title,year,authors,venue,abstract,url",
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


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def calibrate_similarity_score(raw_score: float) -> float:
    """
    Embedding cosine values are usually high.
    This rescales them into a more realistic 0-100 relatedness score.
    """
    low = 0.68
    high = 0.94

    normalized = (raw_score - low) / (high - low)
    normalized = max(0.0, min(1.0, normalized))

    return round(normalized * 100, 1)


def chunk_report_for_similarity(report_text: str) -> List[str]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=min(settings.chunk_size, 1000),
        chunk_overlap=min(settings.chunk_overlap, 160),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [" ".join(chunk.split()) for chunk in splitter.split_text(report_text)]
    return [chunk for chunk in chunks if len(chunk) >= 80]


def remove_references_section(text: str) -> str:
    lower = text.lower()
    idx = lower.find("references")
    if idx == -1:
        return text
    return text[:idx]


def normalize_title(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def token_jaccard(a: str, b: str) -> float:
    tokens_a = set(normalize_title(a).split())
    tokens_b = set(normalize_title(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def is_self_match(paper: dict, project_title: str, report_text: str) -> bool:
    title = paper.get("title") or ""
    normalized_title = normalize_title(title)
    normalized_project_title = normalize_title(project_title)
    normalized_report_start = normalize_title(report_text[:3000])

    if not normalized_title or not normalized_project_title:
        return False

    if normalized_title == normalized_project_title:
        return True

    if token_jaccard(normalized_title, normalized_project_title) >= 0.92:
        return True

    return normalized_title in normalized_report_start


def dedupe_candidates(
    candidate_groups: List[List[dict]],
    project_title: str,
    report_text: str,
    max_candidates: int = 15,
) -> List[dict]:
    seen = set()
    deduped = []

    for group in candidate_groups:
        for paper in group:
            title = (paper.get("title") or "").strip()
            abstract = (paper.get("abstract") or "").strip()
            if not title or not abstract:
                continue

            if is_self_match(paper, project_title=project_title, report_text=report_text):
                logger.info("Skipping self-match candidate: %s", title)
                continue

            key = paper.get("paperId") or title.lower()
            if key in seen:
                continue

            seen.add(key)
            deduped.append(paper)
            if len(deduped) >= max_candidates:
                return deduped

    return deduped


def get_similarity_confidence(
    paper: dict,
    best_score: float,
    avg_top_score: float,
    coverage_score: float,
    final_score: float,
) -> str:
    has_abstract = bool((paper.get("abstract") or "").strip())

    if has_abstract and final_score >= 80 and best_score >= 0.88 and coverage_score >= 0.08:
        return "High"

    if has_abstract and final_score >= 60 and best_score >= 0.82:
        return "Medium"

    return "Low"


def score_candidate_papers(
    report_text: str,
    candidate_papers: List[dict],
    limit: int,
    top_chunks_per_paper: int = 3,
) -> List[RelatedPaper]:
    if not candidate_papers:
        return []

    clean_report_text = remove_references_section(report_text)
    report_chunks = chunk_report_for_similarity(clean_report_text)
    if not report_chunks:
        return []

    # Keep embedding work bounded. The first chunks usually contain title, abstract, and method.
    report_chunks = report_chunks[:80]
    candidate_texts = [
        f"{paper.get('title', '')}\n\n{paper.get('abstract', '')}"
        for paper in candidate_papers
    ]

    embeddings = get_embeddings()
    report_vectors = embeddings.embed_documents(report_chunks)
    candidate_vectors = embeddings.embed_documents(candidate_texts)

    scored = []
    for paper, candidate_vector in zip(candidate_papers, candidate_vectors):
        chunk_scores = [
            (idx, _cosine_similarity(candidate_vector, report_vector))
            for idx, report_vector in enumerate(report_vectors)
        ]
        chunk_scores.sort(key=lambda item: item[1], reverse=True)
        best_matches = chunk_scores[:top_chunks_per_paper]
        best_score = best_matches[0][1] if best_matches else 0.0
        avg_top_score = sum(score for _, score in best_matches) / max(len(best_matches), 1)
        high_similarity_threshold = 0.84

        coverage_score = sum(
            1 for _, score in chunk_scores if score >= high_similarity_threshold
        ) / max(len(chunk_scores), 1)

        raw_similarity = (
            (best_score * 0.50)
            + (avg_top_score * 0.35)
            + (coverage_score * 0.15)
        )

        final_score = calibrate_similarity_score(raw_similarity)

        authors = [a.get("name", "") for a in paper.get("authors", []) if a.get("name")]
        matched_chunks = [
            f"R{idx:03d}: {report_chunks[idx][:650]}"
            for idx, _ in best_matches
        ]
        scored.append(
            RelatedPaper(
                title=paper.get("title") or "Untitled Paper",
                year=paper.get("year"),
                authors=authors,
                venue=paper.get("venue"),
                abstract=paper.get("abstract"),
                url=paper.get("url"),
                similarity_score=final_score,
                confidence=get_similarity_confidence(
                    paper=paper,
                    best_score=best_score,
                    avg_top_score=avg_top_score,
                    coverage_score=coverage_score,
                    final_score=final_score,
                ),
                matched_report_chunks=matched_chunks,
            )
        )

    scored.sort(key=lambda paper: paper.similarity_score, reverse=True)
    return scored[:limit]


def build_similarity_explanation_prompt(project_title: str, short_summary: str, scored_papers: List[RelatedPaper]) -> str:
    candidates_text = []
    for i, paper in enumerate(scored_papers, start=1):
        candidates_text.append(
            f"""
Candidate {i}
Title: {paper.title}
Score: {paper.similarity_score}
Confidence: {paper.confidence}
Year: {paper.year or ""}
Venue: {paper.venue or ""}
Abstract: {paper.abstract or ""}
Matched report chunks:
{chr(10).join(paper.matched_report_chunks)}
""".strip()
        )

    return f"""
You are explaining semantic relatedness between a student report and academic papers.

Project title:
{project_title}

Project summary:
{short_summary}

For each candidate, write a short grounded explanation using the score, abstract, and matched report chunks.

IMPORTANT:
- Do not claim plagiarism.
- Call the number a semantic relatedness score, not a plagiarism score.
- Mention concrete overlaps in task, method, data, or domain.
- Keep each explanation to 1 or 2 sentences.
- Return ONLY valid JSON in this format:

{{
  "papers": [
    {{
      "title": "...",
      "similarity_reason": "..."
    }}
  ]
}}

Candidate papers:
\"\"\"
{chr(10).join(candidates_text)}
\"\"\"
"""


def explain_related_papers(
    project_title: str,
    short_summary: str,
    scored_papers: List[RelatedPaper],
    model_name: Optional[str] = None,
) -> List[RelatedPaper]:
    if not scored_papers:
        return []

    llm = get_llm(model_name=model_name)
    prompt = build_similarity_explanation_prompt(project_title, short_summary, scored_papers)
    response = invoke_with_retry(llm, prompt)
    parsed = safe_json_load(response.content)

    reasons_by_title = {
        (item.get("title") or "").strip().lower(): item.get("similarity_reason", "")
        for item in parsed.get("papers", [])
    }

    papers = []
    for paper in scored_papers:
        reason = reasons_by_title.get(paper.title.strip().lower())
        if reason:
            paper.similarity_reason = reason
        elif not paper.similarity_reason:
            paper.similarity_reason = (
                "This paper is semantically related based on overlap between its title/abstract "
                "and the matched report chunks."
            )
        papers.append(paper)

    return papers


def build_novelty_prompt(project_title: str, short_summary: str, related_papers: List[RelatedPaper]) -> str:
    paper_blocks = []
    for i, paper in enumerate(related_papers, start=1):
        paper_blocks.append(
            f"""
Paper {i}
Title: {paper.title}
Score: {paper.similarity_score}
Confidence: {paper.confidence}
Abstract: {paper.abstract or ""}
Similarity reason: {paper.similarity_reason}
Matched report chunks:
{chr(10).join(paper.matched_report_chunks[:2])}
""".strip()
        )

    return f"""
You are a research advisor analyzing a submitted project against retrieved related papers.

Submitted project title:
{project_title}

Submitted project summary:
{short_summary}

Retrieved related papers and evidence:
\"\"\"
{chr(10).join(paper_blocks)}
\"\"\"

Identify crowded areas and suggest novelty directions that appear less emphasized in the retrieved papers.

IMPORTANT:
- Do not claim that a topic has no research.
- Say "based on the retrieved related papers" when discussing gaps.
- Focus on concrete, implementable project extensions.
- Return ONLY valid JSON in this format:

{{
  "crowded_topics": [
    "..."
  ],
  "novelty_directions": [
    {{
      "direction": "...",
      "why_promising": "...",
      "how_to_extend": "...",
      "expected_contribution": "..."
    }}
  ]
}}
"""


def generate_novelty_directions(
    project_title: str,
    short_summary: str,
    related_papers: List[RelatedPaper],
    model_name: Optional[str] = None,
) -> Optional[NoveltyAnalysis]:
    if not related_papers:
        return None

    llm = get_llm(model_name=model_name)
    prompt = build_novelty_prompt(project_title, short_summary, related_papers)
    response = invoke_with_retry(llm, prompt)
    parsed = safe_json_load(response.content)

    try:
        return NoveltyAnalysis(**parsed)
    except Exception:
        return None


def build_search_queries(extracted: dict, project_title: str, short_summary: str) -> List[str]:
    raw_queries = extracted.get("search_queries") or []
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]

    keywords = extracted.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]

    fallback = " ".join(keywords[:6]).strip()
    queries = [q.strip() for q in raw_queries if q and q.strip()]
    if fallback:
        queries.append(fallback)
    if short_summary:
        queries.append(short_summary[:140])

    deduped = []
    seen = set()
    for query in queries:
        compact = " ".join(query.split())[:180]
        if token_jaccard(compact, project_title) >= 0.92:
            continue
        key = compact.lower()
        if compact and key not in seen:
            deduped.append(compact)
            seen.add(key)
        if len(deduped) >= 3:
            break

    return deduped or [fallback or project_title]


def get_related_papers_from_report(
    report_text: str,
    limit: int = 5,
    model_name: Optional[str] = None,
) -> RelatedPaperSearchResult:
    """Build the complete related-paper and novelty analysis for one report.

    This is intentionally an advisory workflow. The result highlights papers
    that are semantically related to the submitted report and suggests possible
    novelty directions based only on the retrieved candidate papers.
    """
    extracted = extract_paper_query(report_text, model_name=model_name)

    project_title = extracted.get("project_title", "Unknown Project")
    short_summary = extracted.get("short_summary", "")
    search_queries = build_search_queries(extracted, project_title, short_summary)

    logger.info("Searching related papers with queries: %s", search_queries)

    candidate_groups = []
    for query in search_queries:
        try:
            candidate_groups.append(search_semantic_scholar(query, limit=8))
        except Exception as e:
            logger.warning("Semantic Scholar query failed for %r: %s", query, e)

    candidates = dedupe_candidates(
        candidate_groups,
        project_title=project_title,
        report_text=report_text,
        max_candidates=15,
    )

    scored = score_candidate_papers(
        report_text=report_text,
        candidate_papers=candidates,
        limit=limit,
    )

    explained = explain_related_papers(
        project_title=project_title,
        short_summary=short_summary,
        scored_papers=scored,
        model_name=model_name,
    )
    novelty_analysis = generate_novelty_directions(
        project_title=project_title,
        short_summary=short_summary,
        related_papers=explained,
        model_name=model_name,
    )

    return RelatedPaperSearchResult(
        query_title=project_title,
        query_summary=short_summary,
        papers=explained[:limit],
        novelty_analysis=novelty_analysis,
    )
