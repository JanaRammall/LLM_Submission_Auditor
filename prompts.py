from models import RubricCriterion


def rubric_compiler_prompt(instructions_text: str) -> str:
    return f"""
You are a precise academic project requirements analyst.

Your job is to read the project instructions and convert them into a structured rubric for an automated submission auditor.

IMPORTANT RULES:
1. Extract only requirements explicitly supported by the instructions.
2. Do not invent requirements that are not grounded in the instructions.
3. Group criteria into sensible categories such as:
   - Implementation
   - Report
   - Code
   - Presentation
   - Safety/Submission
4. Each criterion must include:
   - criterion_id
   - category
   - title
   - description
   - expected_evidence_source
   - priority
   - auditability
   - notes
5. Use expected_evidence_source from this allowed set only:
   - report
   - readme
   - slides
   - code
   - external
   - multiple
   - unknown
6. Use priority from this allowed set only:
   - high
   - medium
   - low
7. Use auditability from this allowed set only:
   - direct
   - partial
   - external_only
8. Return ONLY valid JSON in this format:
{{
  "project_title": "...",
  "summary": "...",
  "criteria": [
    {{
      "criterion_id": "IMP-01",
      "category": "Implementation",
      "title": "Custom LLM Agent",
      "description": "The project must implement a custom LLM agent for a specialized task.",
      "expected_evidence_source": "multiple",
      "priority": "high",
      "auditability": "direct",
      "notes": "..."
    }}
  ]
}}

GUIDANCE:
- If the instructions contain evaluation weights, use them to help assign priority.
- If the criterion depends on uploaded artifacts such as report, code, README, or slides, use:
  - direct = can normally be checked from uploaded artifacts
  - partial = can be partially judged from uploaded artifacts but may need more context
  - external_only = usually cannot be judged from uploaded artifacts alone
- Group size, administrative constraints, or other non-artifact requirements should usually be external_only.

Project instructions:
\"\"\"
{instructions_text[:30000]}
\"\"\"
"""


def semantic_criterion_query(criterion: RubricCriterion) -> str:
    return f"{criterion.title}. {criterion.description}"


def semantic_criterion_prompt(criterion: RubricCriterion, evidence_text: str) -> str:
    return f"""
You are a strict but fair university project auditor.

Evaluate ONLY this semantic criterion.

Criterion ID: {criterion.criterion_id}
Category: {criterion.category}
Title: {criterion.title}
Description: {criterion.description}
Expected evidence source: {criterion.expected_evidence_source}
Priority: {criterion.priority}
Auditability: {criterion.auditability}
Notes: {criterion.notes or "N/A"}

IMPORTANT RULES:
1. Use ONLY the provided evidence.
2. Do not confuse this criterion with neighboring criteria.
3. If the evidence clearly supports the criterion, return Pass.
4. If the evidence clearly shows the requirement is not met, return Fail.
5. If the evidence is partial, vague, or incomplete, return Partial.
6. If there is not enough evidence from the provided artifacts, return Not enough evidence.
7. Return ONLY valid JSON in this format:
{{
  "criterion_id": "{criterion.criterion_id}",
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