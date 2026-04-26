import html


def format_ai_content(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)

    return str(content)


def format_chat_html(text: str) -> str:
    return html.escape(text or "").replace("\n", "<br>")
