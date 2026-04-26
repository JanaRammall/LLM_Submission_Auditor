"""Course-specific compact rubric used by the auditor UI and checker.

The LLM first extracts a broad rubric from the uploaded instructions. This file
then maps the result to the stable rubric used by the app so the UI and audit
logic always work with predictable criterion IDs.
"""

from typing import List

from core.models import CompiledRubric, RubricCriterion


COMPACT_CRITERIA = [
    {
        "criterion_id": "IMP-01",
        "category": "Implementation",
        "title": "Custom LLM Agent",
        "description": "Does the project clearly implement a specialized LLM agent for a meaningful task?",
        "expected_evidence_source": "report",
        "priority": "high",
        "auditability": "direct",
        "notes": "Core implementation requirement.",
    },
    {
        "criterion_id": "IMP-02",
        "category": "Implementation",
        "title": "RAG with Vector Embeddings",
        "description": "Does the project implement Retrieval-Augmented Generation using vector embeddings or vector retrieval?",
        "expected_evidence_source": "report",
        "priority": "high",
        "auditability": "direct",
        "notes": "Core implementation requirement.",
    },
    {
        "criterion_id": "IMP-03",
        "category": "Implementation",
        "title": "Multiple Tool Integration",
        "description": "Are at least three tools, APIs, or external components clearly integrated into the system?",
        "expected_evidence_source": "report",
        "priority": "high",
        "auditability": "direct",
        "notes": "Core implementation requirement.",
    },
    {
        "criterion_id": "IMP-04",
        "category": "Implementation",
        "title": "Custom Tool",
        "description": "Is at least one integrated tool or component clearly custom-designed by the team?",
        "expected_evidence_source": "report",
        "priority": "high",
        "auditability": "direct",
        "notes": "One of the integrated tools should be custom.",
    },
    {
        "criterion_id": "IMP-05",
        "category": "Implementation",
        "title": "User Interface",
        "description": "Is there a clear user-facing interface for interacting with the system?",
        "expected_evidence_source": "report",
        "priority": "high",
        "auditability": "direct",
        "notes": "Interface may be Streamlit, Gradio, web app, or equivalent.",
    },
    {
        "criterion_id": "IMP-06",
        "category": "Implementation",
        "title": "Conversation History",
        "description": "Does the system maintain conversation history or memory across interactions?",
        "expected_evidence_source": "report",
        "priority": "medium",
        "auditability": "direct",
        "notes": "Only judge from explicit evidence.",
    },
    {
        "criterion_id": "IMP-07",
        "category": "Implementation",
        "title": "Error Handling",
        "description": "Is there evidence of proper error handling or robustness in the system design?",
        "expected_evidence_source": "report",
        "priority": "medium",
        "auditability": "direct",
        "notes": "Only judge from explicit evidence.",
    },
    {
        "criterion_id": "REP-01",
        "category": "Report",
        "title": "IEEE Format and Page Limit",
        "description": "Is the report in IEEE-style format and within the 6-page limit?",
        "expected_evidence_source": "report",
        "priority": "high",
        "auditability": "direct",
        "notes": "Combine formatting and page-limit checks.",
    },
    {
        "criterion_id": "REP-02",
        "category": "Report",
        "title": "Literature Review Quality",
        "description": "Does the report include a literature review with enough references, including recent papers from 2025-2026?",
        "expected_evidence_source": "report",
        "priority": "high",
        "auditability": "direct",
        "notes": "Combine related-work presence, quantity, and recency.",
    },
    {
        "criterion_id": "REP-03",
        "category": "Report",
        "title": "Required Core Sections",
        "description": "Does the report include the essential sections: Abstract, Introduction, Related Work, Methodology, Experimental Setup, Results and Discussion, Conclusion and Future Work, and References?",
        "expected_evidence_source": "report",
        "priority": "high",
        "auditability": "direct",
        "notes": "Grouped section check.",
    },
    {
        "criterion_id": "REP-04",
        "category": "Report",
        "title": "System Design and Tool Documentation",
        "description": "Does the report clearly explain the system architecture, methodology, tools, and experimental setup?",
        "expected_evidence_source": "report",
        "priority": "high",
        "auditability": "direct",
        "notes": "Grouped design/documentation quality check.",
    },
    {
        "criterion_id": "REP-05",
        "category": "Report",
        "title": "Novelty and Differentiation",
        "description": "Does the report clearly explain how the project differs from existing work and what its main contributions are?",
        "expected_evidence_source": "report",
        "priority": "high",
        "auditability": "direct",
        "notes": "Grouped novelty/differentiation check.",
    },
]


def build_compact_rubric(original: CompiledRubric) -> CompiledRubric:
    criteria: List[RubricCriterion] = [RubricCriterion(**item) for item in COMPACT_CRITERIA]
    return CompiledRubric(
        project_title=original.project_title,
        summary=original.summary,
        criteria=criteria,
    )


def build_instructor_rubric(
    project_title: str = "COE548/748 Final Project: Building a Specialized LLM Agent",
) -> CompiledRubric:
    """Return the built-in instructor rubric without making an LLM call."""
    criteria: List[RubricCriterion] = [RubricCriterion(**item) for item in COMPACT_CRITERIA]
    return CompiledRubric(
        project_title=project_title,
        summary="Instructor-defined compact rubric for the specialized LLM agent final project.",
        criteria=criteria,
    )
