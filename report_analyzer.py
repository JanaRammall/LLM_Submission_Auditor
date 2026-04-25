import re
from typing import Dict, List


SECTION_PATTERNS = {
    "abstract": [r"\babstract\b"],
    "introduction": [r"\bintroduction\b", r"\bi\.\s*introduction\b"],
    "related work": [r"\brelated work\b", r"\bliterature review\b"],
    "methodology": [r"\bmethodology\b", r"\bmethods?\b"],
    "experimental setup": [r"\bexperimental setup\b", r"\bexperiments?\b"],
    "results and discussion": [r"\bresults and discussion\b", r"\bresults\b", r"\bdiscussion\b"],
    "conclusion and future work": [r"\bconclusion\b", r"\bfuture work\b"],
    "author contributions": [r"\bauthor contributions\b"],
    "references": [r"\breferences\b"],
}


def estimate_page_count(report_text: str) -> int:
    return len(re.findall(r"--- PAGE \d+ ---", report_text))


def detect_sections(report_text: str) -> Dict[str, bool]:
    text = report_text.lower()
    return {
        name: any(re.search(pattern, text) for pattern in patterns)
        for name, patterns in SECTION_PATTERNS.items()
    }


def extract_references_block(report_text: str) -> str:
    lower = report_text.lower()
    idx = lower.find("references")
    if idx == -1:
        return ""
    return report_text[idx:]


def count_reference_entries(references_block: str) -> int:
    if not references_block.strip():
        return 0
    bracket_refs = re.findall(r"\[\d+\]", references_block)
    unique_refs = sorted(set(bracket_refs), key=lambda x: int(x.strip("[]")))
    return len(unique_refs)


def extract_reference_years(references_block: str) -> List[int]:
    years = re.findall(r"\b(20\d{2}|19\d{2})\b", references_block)
    return [int(y) for y in years]


def count_recent_references(years: List[int], start_year: int = 2025, end_year: int = 2026) -> int:
    return sum(1 for y in years if start_year <= y <= end_year)


def build_report_features(report_text: str) -> Dict:
    sections = detect_sections(report_text)
    references_block = extract_references_block(report_text)
    years = extract_reference_years(references_block)

    return {
        "page_count": estimate_page_count(report_text),
        "sections": sections,
        "reference_count": count_reference_entries(references_block),
        "recent_reference_count_2025_2026": count_recent_references(years, 2025, 2026),
        "reference_years_found": years,
        "has_references_block": bool(references_block.strip()),
    }