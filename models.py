from typing import List, Literal, Optional
from pydantic import BaseModel, Field


PriorityLevel = Literal["high", "medium", "low"]
EvidenceSource = Literal["report", "readme", "slides", "code", "external", "multiple", "unknown"]
Auditability = Literal["direct", "partial", "external_only"]


class RubricCriterion(BaseModel):
    criterion_id: str
    category: str
    title: str
    description: str
    expected_evidence_source: EvidenceSource
    priority: PriorityLevel
    auditability: Auditability
    notes: Optional[str] = ""


class CompiledRubric(BaseModel):
    project_title: str
    summary: str
    criteria: List[RubricCriterion]


class CriterionResult(BaseModel):
    criterion_id: str
    status: Literal["Pass", "Partial", "Fail", "Not enough evidence"]
    evidence_found: str
    missing_or_weak: str
    improvement: str
    evidence_chunk_ids: List[str] = Field(default_factory=list)


class AuditReport(BaseModel):
    active_results: List[CriterionResult]
    deferred_results: List[CriterionResult]