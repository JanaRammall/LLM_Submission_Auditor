# LLM Submission Auditor

A Streamlit app for auditing a student paper against project requirements. The app compiles a rubric from uploaded project instructions, evaluates the submitted report, finds related academic papers, suggests novelty directions, and provides a grounded chatbot for questions about the submission.

## Features

- Upload a project instructions/rubric PDF and a student report PDF.
- Compile the uploaded instructions into a structured rubric.
- Evaluate the report using deterministic checks and LLM-based semantic checks.
- Build a Chroma vector store from report chunks for evidence retrieval.
- Search Semantic Scholar for related academic papers.
- Compare report chunks with candidate paper titles/abstracts using embeddings.
- Generate novelty-direction suggestions based on retrieved related papers.
- Ask questions through a chatbot grounded in the report, rubric, audit results, and related-paper analysis.

## Project Structure

```text
app.py                  Streamlit entry point and tab router
styles.css              App styling

core/
  config.py             Loads .env settings and model/vector-store defaults
  models.py             Shared Pydantic models

audit/
  checker.py            Deterministic and semantic rubric evaluation
  compact_rubric.py     Stable course-specific rubric used by the app
  prompts.py            Prompt templates
  report_analyzer.py    Fast deterministic report checks
  rubric_compiler.py    LLM-based instructions-to-rubric compiler

rag/
  retrieval.py          Report chunking, Chroma storage, evidence retrieval

research/
  related_papers.py     Semantic Scholar search, similarity scoring, novelty analysis

services/
  analysis_pipeline.py  End-to-end analysis workflow
  chat_service.py       Grounded chatbot context and responses
  llm.py                Shared Gemini LLM/embedding clients and retry wrapper

ui/
  pages.py              Streamlit tab renderers

utils/
  formatting.py         Chat/content formatting helpers
  json_utils.py         JSON parsing helper for model responses


```

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=your_google_api_key_here
S2_API_KEY=your_semantic_scholar_api_key_here
```

`GOOGLE_API_KEY` is required. `S2_API_KEY` is optional but recommended because Semantic Scholar has stricter limits without an API key.

Aliases are also supported:

```env
GEMINI_API_KEY=your_google_api_key_here
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here
```

## Run

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Workflow

1. Open the `Submission` tab.
2. Upload the project instructions/rubric PDF.
3. Upload the student report PDF.
4. Click `Run Analysis`.
5. Review rubric results in `Evaluation`.
6. Review related papers in `Similarity Check`.
7. Review possible research improvements in `Novelty Directions`.
8. Ask grounded questions in `Chatbot`.

## How the Evaluation Works

The app combines deterministic and LLM-based checks:

- Deterministic checks inspect page count, detected sections, references, and reference recency.
- Semantic checks retrieve relevant report chunks from Chroma and ask Gemini to judge implementation and report-quality criteria.
- Semantic checks are grouped and run in parallel where possible to reduce audit time.
- Results are ordered by the compact rubric IDs so the UI stays predictable.

## Similarity and Novelty Notes

The related-paper score is a semantic relatedness score, not a plagiarism score.

The app:

1. Extracts a search basis from the report.
2. Searches Semantic Scholar for candidate papers.
3. Removes duplicate and self-match candidates.
4. Embeds report chunks and candidate title/abstract text.
5. Scores semantic overlap and shows matched report chunks.
6. Uses retrieved related papers to suggest novelty directions.

Novelty suggestions are advisory and based only on retrieved candidates, not the entire research literature.

## Generated Data

Generated vector stores are written under:

```text
data/vector_db/
```

Uploaded/generated runtime data should not be committed. The repository keeps `.gitkeep` files so the directories exist.
