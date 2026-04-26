"""Compile project instructions into a structured rubric model.

This is the first model call in the analysis workflow. It converts the uploaded
project instructions into the CompiledRubric Pydantic model before the compact
course rubric is applied.
"""

import logging
from typing import Optional

from pydantic import ValidationError

from audit.prompts import rubric_compiler_prompt
from core.models import CompiledRubric
from services.llm import get_llm, invoke_with_retry
from utils.json_utils import safe_json_load


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compile_rubric(instructions_text: str, model_name: Optional[str] = None) -> CompiledRubric:
    llm = get_llm(model_name=model_name)
    prompt = rubric_compiler_prompt(instructions_text)

    logger.info("Compiling rubric from instructions...")
    response = invoke_with_retry(llm, prompt)
    parsed = safe_json_load(response.content)

    try:
        return CompiledRubric(**parsed)
    except ValidationError as e:
        logger.exception("Rubric validation failed.")
        raise ValueError(f"Rubric validation failed: {e}") from e
