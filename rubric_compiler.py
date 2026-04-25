import json
import logging
from typing import Optional

from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_settings
from models import CompiledRubric
from prompts import rubric_compiler_prompt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_llm(model_name: Optional[str] = None) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model_name or settings.chat_model,
        api_key=settings.google_api_key,
        temperature=0.0,
    )


def _safe_json_load(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("Model did not return valid JSON.")


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _invoke(llm: ChatGoogleGenerativeAI, prompt: str):
    return llm.invoke(prompt)


def compile_rubric(instructions_text: str, model_name: Optional[str] = None) -> CompiledRubric:
    llm = get_llm(model_name=model_name)
    prompt = rubric_compiler_prompt(instructions_text)

    logger.info("Compiling rubric from instructions...")
    response = _invoke(llm, prompt)
    parsed = _safe_json_load(response.content)

    try:
        return CompiledRubric(**parsed)
    except ValidationError as e:
        logger.exception("Rubric validation failed.")
        raise ValueError(f"Rubric validation failed: {e}") from e