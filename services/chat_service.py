"""Build grounded chatbot answers from report chunks and audit context.

The chatbot does not answer from general model knowledge. For each user
question, it retrieves report chunks from the vector store, serializes the
rubric/audit/related-paper context, and sends those bounded inputs to Gemini.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from rag.retrieval import retrieve_evidence
from services.llm import get_llm
from utils.formatting import format_ai_content


def serialize_audit_context(compiled_rubric, audit_report, report_features, related_papers_result) -> str:
    blocks = []

    if compiled_rubric:
        blocks.append(f"Project title: {compiled_rubric.project_title}")
        blocks.append(
            "Rubric criteria:\n"
            + "\n".join(
                f"- {criterion.criterion_id}: {criterion.title} | {criterion.description}"
                for criterion in compiled_rubric.criteria
            )
        )

    if report_features:
        blocks.append(f"Deterministic report features: {report_features}")

    if audit_report:
        results = []
        for result in audit_report.active_results:
            results.append(
                "\n".join(
                    [
                        f"- {result.criterion_id}: {result.status}",
                        f"  Evidence: {result.evidence_found or 'None'}",
                        f"  Missing or weak: {result.missing_or_weak or 'None'}",
                        f"  Improvement: {result.improvement or 'None'}",
                        f"  Evidence chunks: {', '.join(result.evidence_chunk_ids) if result.evidence_chunk_ids else 'None'}",
                    ]
                )
            )
        blocks.append("Evaluation results:\n" + "\n".join(results))

    if related_papers_result:
        papers = []
        for paper in related_papers_result.papers[:5]:
            papers.append(
                f"- {paper.title or 'Untitled'} ({paper.year or 'N/A'}, {paper.venue or 'N/A'}): "
                f"{paper.similarity_reason or 'No similarity reason available.'}"
            )
        blocks.append("Related papers:\n" + ("\n".join(papers) if papers else "None found."))

    return "\n\n".join(blocks)


def serialize_retrieved_docs(docs) -> str:
    blocks = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "UNKNOWN")
        source = doc.metadata.get("source_name", "unknown")
        blocks.append(f"[{chunk_id}] [source={source}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def answer_chat_question(
    prompt: str,
    chat_history: list,
    vector_store,
    app_context: str,
    model_name: str,
) -> str:
    docs = retrieve_evidence(vector_store, query=prompt, k=5)
    paper_context = serialize_retrieved_docs(docs)

    history_lines = []
    for message in chat_history[-8:]:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        history_lines.append(f"{role}: {format_ai_content(message.content)}")

    messages = [
        SystemMessage(
            content=(
                "You are a careful assistant for a university submission audit app. "
                "Answer questions about the submitted paper and the evaluation results. "
                "Use only the provided paper chunks, rubric, evaluation, deterministic features, "
                "and related-paper context. If the answer is not supported by that context, say so. "
                "When useful, cite chunk IDs or criterion IDs."
            )
        ),
        HumanMessage(
            content=f"""
Conversation so far:
{chr(10).join(history_lines) if history_lines else "No previous messages."}

Current user question:
{prompt}

Relevant submitted-paper chunks:
{paper_context or "No matching chunks were found."}

Evaluation and app context:
{app_context or "No evaluation context is available."}
"""
        ),
    ]

    response = get_llm(model_name=model_name).invoke(messages)
    return format_ai_content(response.content)
