import json
from datetime import datetime
from typing import Any


def parse_first_json_value(input_text: str) -> Any:
    """Parse and return the first valid JSON value in a string.

    This mirrors the TypeScript strategy:
    1) Attempt full-string parse first.
    2) If that fails, scan for the first JSON object/array prefix that parses.
    """
    text = input_text.lstrip()
    if not text:
        raise SyntaxError("Unexpected end of JSON input")

    try:
        return json.loads(text)
    except Exception:
        pass

    for start in range(len(text)):
        first_char = text[start]
        if first_char not in ["{", "["]:
            continue

        for end in range(start + 1, len(text) + 1):
            try:
                return json.loads(text[start:end])
            except Exception:
                pass

    raise SyntaxError("Unexpected token in JSON input")


def current_datetime_system_message() -> dict[str, str]:
    """Build the standard datetime system message prepended to submissions."""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "role": "system",
        "content": f"!DATETIME: The current date and time is {timestamp}",
    }
