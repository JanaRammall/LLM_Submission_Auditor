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
        Criterion("IMP-01", "Implementation", "Custom LLM Agent",
                  "The project must implement a custom LLM agent for a specialized task."),
        Criterion("IMP-02", "Implementation", "RAG with Vector Embeddings",
                  "The project must implement Retrieval-Augmented Generation using vector embeddings."),
        Criterion("IMP-03", "Implementation", "At Least Three Tools",
                  "The system must integrate at least three additional tools."),
        Criterion("IMP-04", "Implementation", "At Least One Custom Tool",
                  "At least one integrated tool must be customized/designed by the team."),
        Criterion("IMP-05", "Implementation", "User Interface",
                  "The project must include an interface such as Streamlit, Gradio, or a web app."),
        Criterion("IMP-06", "Implementation", "Conversation History",
                  "The system should maintain conversation history."),
        Criterion("IMP-07", "Implementation", "Error Handling",
                  "The system should implement proper error handling."),
        Criterion("REP-01", "Report", "IEEE Format",
                  "The final report must be written in IEEE conference paper format."),
        Criterion("REP-02", "Report", "Literature Review with 10 Recent Papers",
                  "The report must include at least 10 research papers from 2025–2026 in the literature review."),
        Criterion("REP-03", "Report", "Novelty Statement",
                  "The report must clearly explain how the system differs from existing work."),
        Criterion("REP-04", "Report", "Required Report Sections",
                  "The report should include Abstract, Introduction, Related Work, Methodology, System design diagram, Tool documentation, Experimental Setup, Models, Datasets, Prompts and Design Decisions, Evaluation Metrics, Results and Discussion, Conclusion and Future Work, Author Contributions, and References."),
        Criterion("CODE-01", "Code", "Complete Code Submission",
                  "The submitted code should include source code, configuration files, and dependencies."),
        Criterion("CODE-02", "Code", "Requirements File",
                  "The code submission should include a requirements.txt or equivalent dependency file."),
        Criterion("CODE-03", "Code", "README Run Instructions",
                  "The code submission should include a README with clear instructions for running the project."),
        Criterion("SAFE-01", "Safety/Submission", "No API Keys in Submission",
                  "API keys should not be included in submitted files and should be stored securely using environment variables."),
    ]