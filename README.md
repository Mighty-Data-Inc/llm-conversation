# gpt-conversation

Utilities for managing multi-shot LLM conversations and structured JSON responses with OpenAI's Responses API.

This repo contains shared, production-focused helpers from **Mighty Data Inc.** for building reliable LLM applications without rewriting the same plumbing in every project.

## Design goals

* Minimal abstractions
* Predictable behavior
* Cross-language parity (Python + TypeScript)
* Easy to drop into real projects

This is not a framework — just a clean, reusable toolkit for the parts of LLM integration that tend to get copy-pasted everywhere.

## Packages

- TypeScript: `@mdi/gpt-conversation` (npm) in `packages/typescript-gpt-conversation`
- Python: `gpt-conversation` (PyPI, import as `gpt_conversation`) in `packages/python-gpt-conversation`

Package-specific docs:

- TypeScript: [packages/typescript-gpt-conversation/README.md](packages/typescript-gpt-conversation/README.md)
- Python: [packages/python-gpt-conversation/README.md](packages/python-gpt-conversation/README.md)

## Feature overview

Core capabilities (Python + TypeScript):

- Conversation and multi-message submission helpers (`GptConversation` / `gpt_submit`)
- Structured JSON response support
- JSON schema helpers for structured output (`JSONSchemaFormat`)

## Quick start

### Python

```python
from gpt_conversation import GptConversation

conversation = GptConversation(openai_client=client)
reply = conversation.submit_user_message('Say hello.')
print(reply)
```

### TypeScript

```ts
import OpenAI from 'openai';
import { GptConversation } from '@mdi/gpt-conversation';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const conversation = new GptConversation([], { openaiClient: client });
const reply = await conversation.submitUserMessage('Say hello.');
console.log(reply);
```

## Local dev (Windows)

### Python

From `packages/python-gpt-conversation`, activate the package venv and run tests:

```powershell
.\venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m unittest discover -v -s tests
```

Live integration tests (real API) require `OPENAI_API_KEY` in your environment.

### TypeScript

From `packages/typescript-gpt-conversation`, install dependencies and run tests/build:

```powershell
npm ci
npm test
npm run build
```

## Unit testing with live API calls

Some tests intentionally call the real OpenAI API instead of mocking model responses.

This is by design: the core contract includes prompt wording, output parsing, and model behavior working together. Mock-only tests cannot verify whether production prompts still elicit the required structured output.

These tests do have tradeoffs:

- They require `OPENAI_API_KEY` in the test environment.
- They incur a small API cost when run.
- They can be slower than pure unit tests.

Deterministic assertions are still intentional here: tests are written with tightly scoped instructions and clearly defined JSON outcomes, so stable structured output is treated as a baseline requirement. If those tests fail, we treat it as a bug in prompt design, output handling, or integration behavior.

## Release process

This repo ships two public packages with aligned versions:

- npm: `@mdi/gpt-conversation` from `packages/typescript-gpt-conversation`
- PyPI: `gpt-conversation` from `packages/python-gpt-conversation`

GitHub release automation publishes each package automatically on push to `main`
when its package version changes:

- TypeScript checks `packages/typescript-gpt-conversation/package.json`
- Python checks `packages/python-gpt-conversation/pyproject.toml`

Before publishing, ensure both versions are updated (`package.json` and `pyproject.toml`), then authenticate once locally:

- npm: `npm login`
- PyPI: configure `~/.pypirc` or use `python -m twine upload --repository pypi dist/*`

After publish, tag and push a release tag (example):

```powershell
git tag v1.1.3
git push origin v1.1.3
```
