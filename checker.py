import json
import os
from typing import List, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from rubric_data import Criterion


def build_prompt(report_text: str, criteria: List[Criterion]) -> str:
    criteria_text = "\n".join(
        [
            f"- {c.id} | {c.category} | {c.title}: {c.description}"
            for c in criteria
        ]
    )

    prompt = f"""
You are an auditing assistant for a university project submission.

Your task:
1. Read the submission/report text.
2. Evaluate it against the rubric criteria.
3. For EACH criterion, output:
   - criterion_id
   - status: one of ["Pass", "Partial", "Fail"]
   - evidence_found: short quote or summary from the report
   - missing_or_weak: what is missing or unclear
   - improvement: one short actionable suggestion

Important rules:
- Be strict but fair.
- Only use the provided report text.
- If something is unclear or weakly implied, use "Partial" not "Pass".
- Return ONLY valid JSON.
- The JSON format must be:
{{
  "results": [
    {{
      "criterion_id": "IMP-01",
      "status": "Pass",
      "evidence_found": "...",
      "missing_or_weak": "...",
      "improvement": "..."
    }}
  ]
}}

Rubric criteria:
{criteria_text}

Submission/report text:
\"\"\"
{report_text[:30000]}
\"\"\"
"""
    return prompt


def get_llm(model_name: str = "gemini-2.5-flash"):
    api_key = "AIzaSyAHblrOEj33E8rSZWYixmc6V6J0y3lAbaE"
    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0.0,
    )


def safe_json_load(text: str) -> Dict:
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError("Model did not return valid JSON.")


def audit_report(report_text: str, criteria: List[Criterion], model_name: str = "gemini-2.5-flash") -> Dict:
    llm = get_llm(model_name=model_name)
    prompt = build_prompt(report_text, criteria)
    response = llm.invoke(prompt)
    return safe_json_load(response.content)