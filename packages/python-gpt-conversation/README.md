# gpt-conversation

Utilities for managing multi-shot LLM conversations and structured JSON responses with OpenAI's Responses API.

## Installation

```bash
pip install gpt-conversation
```

## Quick Start

### `gpt_submit`

```python
from openai import OpenAI
from gpt_conversation import gpt_submit

client = OpenAI()

reply = gpt_submit(
    messages=[{"role": "user", "content": "Say hello."}],
    openai_client=client,
)
print(reply)
```

### `GptConversation`

```python
from openai import OpenAI
from gpt_conversation import GptConversation

client = OpenAI()
conversation = GptConversation(openai_client=client)

reply = conversation.submit_user_message("Give me three project name ideas.")
print(reply)
```

## JSON Response Mode

```python
from openai import OpenAI
from gpt_conversation import gpt_submit

client = OpenAI()

result = gpt_submit(
    messages=[{"role": "user", "content": "Return JSON with keys a and b."}],
    openai_client=client,
    json_response=True,
)

print(type(result))  # dict or list
print(result)
```

## `JSONSchemaFormat`

```python
from openai import OpenAI
from gpt_conversation import gpt_submit, json_schema_format, JSON_INTEGER

client = OpenAI()

response_format = json_schema_format(
    "answer_payload",
    {
        "answer": "The final answer",
        "confidence": ["Confidence score", [0, 100], []],
        "rank": JSON_INTEGER,
    },
    "Structured answer payload",
)

result = gpt_submit(
    messages=[{"role": "user", "content": "Return answer as structured JSON."}],
    openai_client=client,
    json_response=response_format,
)
print(result)
```

## Local dev (Windows)

From `packages/python-gpt-conversation`, activate the package venv and run tests:

```powershell
.\venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m unittest discover -v -s tests
```

Live integration tests (real API) require `OPENAI_API_KEY` in your environment.

## Notes

- Package name for `pip install` is `gpt-conversation`.
- Python import package is `gpt_conversation`.
- `gpt_submit` supports optional warning reporting via `warning_callback`.

