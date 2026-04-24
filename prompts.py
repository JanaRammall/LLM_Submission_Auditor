from rubric_data import Criterion


def criterion_query(criterion: Criterion) -> str:
    return f"{criterion.title}. {criterion.description}"


def criterion_prompt(criterion: Criterion, evidence_text: str) -> str:
    return f"""
You are a strict but fair university project auditor.

Evaluate ONLY this single criterion:

Criterion ID: {criterion.id}
Category: {criterion.category}
Title: {criterion.title}
Description: {criterion.description}

Use ONLY the evidence below.
If the evidence is insufficient, unclear, or only partially relevant, do not mark Pass.

Return ONLY valid JSON in this format:
{{
  "criterion_id": "{criterion.id}",
  "status": "Pass" | "Partial" | "Fail" | "Not enough evidence",
  "evidence_found": "short grounded summary",
  "missing_or_weak": "what is missing or unclear",
  "improvement": "one concrete improvement",
  "evidence_chunk_ids": ["C000", "C001"]
}}

Evidence:
\"\"\"
{evidence_text}
\"\"\"
"""