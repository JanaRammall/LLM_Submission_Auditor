"""Helpers for model responses that should contain JSON."""

import json


def safe_json_load(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("Model did not return valid JSON.")
