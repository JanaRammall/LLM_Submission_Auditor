from dataclasses import dataclass
from typing import List


@dataclass
class Criterion:
    id: str
    category: str
    title: str
    description: str


def get_coe548_rubric() -> List[Criterion]:
    return [
        Criterion(
            id="IMP-01",
            category="Implementation",
            title="Custom LLM Agent",
            description="The project must implement a custom LLM agent for a specialized task."
        ),
        Criterion(
            id="IMP-02",
            category="Implementation",
            title="RAG with Vector Embeddings",
            description="The project must implement Retrieval-Augmented Generation using vector embeddings."
        ),
        Criterion(
            id="IMP-03",
            category="Implementation",
            title="At Least Three Tools",
            description="The system must integrate at least three additional tools."
        ),
        Criterion(
            id="IMP-04",
            category="Implementation",
            title="At Least One Custom Tool",
            description="At least one integrated tool must be customized/designed by the team."
        ),
        Criterion(
            id="IMP-05",
            category="Implementation",
            title="User Interface",
            description="The project must include an interface such as Streamlit, Gradio, or a web app."
        ),
        Criterion(
            id="IMP-06",
            category="Implementation",
            title="Conversation History",
            description="The system should maintain conversation history."
        ),
        Criterion(
            id="IMP-07",
            category="Implementation",
            title="Error Handling",
            description="The system should implement proper error handling."
        ),
        Criterion(
            id="REP-01",
            category="Report",
            title="IEEE Format",
            description="The final report must be written in IEEE conference paper format."
        ),
        Criterion(
            id="REP-02",
            category="Report",
            title="Literature Review with 10 Recent Papers",
            description="The report must include at least 10 research papers from 2025–2026 in the literature review."
        ),
        Criterion(
            id="REP-03",
            category="Report",
            title="Novelty Statement",
            description="The report must explain clearly how the system differs from existing work."
        ),
        Criterion(
            id="REP-04",
            category="Report",
            title="Required Report Sections",
            description=(
                "The report should include Abstract, Introduction, Related Work, Methodology, "
                "System design diagram, Tool documentation, Experimental Setup, Models, Datasets, "
                "Prompts and Design Decisions, Evaluation Metrics, Results and Discussion, "
                "Conclusion and Future Work, Author Contributions, and References."
            )
        ),
        Criterion(
            id="CODE-01",
            category="Code",
            title="Complete Code Submission",
            description="The submitted code should include source code, configuration files, and dependencies."
        ),
        Criterion(
            id="CODE-02",
            category="Code",
            title="Requirements File",
            description="The code submission should include a requirements.txt or equivalent dependency file."
        ),
        Criterion(
            id="CODE-03",
            category="Code",
            title="README Run Instructions",
            description="The code submission should include a README with clear instructions for running the project."
        ),
        Criterion(
            id="SAFE-01",
            category="Safety/Submission",
            title="No API Keys in Submission",
            description="API keys should not be included in submitted files and should be stored securely using environment variables."
        ),
    ]