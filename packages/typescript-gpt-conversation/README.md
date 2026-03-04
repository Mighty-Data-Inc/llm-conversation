# @mightydatainc/gpt-conversation

Utilities for managing multi-shot LLM conversations and structured JSON responses with OpenAI's Responses API.

## Installation

```bash
npm install @mightydatainc/gpt-conversation
```

## Quick Start

### `gptSubmit`

```ts
import OpenAI from 'openai';
import { gptSubmit } from '@mightydatainc/gpt-conversation';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const reply = await gptSubmit(
  [{ role: 'user', content: 'Say hello.' }],
  client
);

console.log(reply);
```

### `GptConversation`

```ts
import OpenAI from 'openai';
import { GptConversation } from '@mightydatainc/gpt-conversation';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const conversation = new GptConversation([], { openaiClient: client });

const reply = await conversation.submitUserMessage(
  'Give me three project name ideas.'
);
console.log(reply);
```

### `JSONSchemaFormat`

```ts
import { JSONSchemaFormat, JSON_INTEGER, gptSubmit } from '@mightydatainc/gpt-conversation';

const responseFormat = JSONSchemaFormat(
  {
    answer: 'The final answer',
    confidence: ['Confidence score', [0, 100], []],
    rank: JSON_INTEGER,
  },
  'answer_payload',
  'Structured answer payload'
);

const result = await gptSubmit(
  [{ role: 'user', content: 'Return answer as structured JSON.' }],
  client,
  { jsonResponse: responseFormat }
);
```

## JSON Response Mode

```ts
import OpenAI from 'openai';
import { gptSubmit } from '@mightydatainc/gpt-conversation';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const result = await gptSubmit(
  [{ role: 'user', content: 'Return JSON with keys a and b.' }],
  client,
  { jsonResponse: true }
);

console.log(result);
```

## CI and Release

- Unified CI + release workflow: `.github/workflows/typescript-release.yml`
  - Runs CI on pull requests and on pushes to `main` when TypeScript package files change.
  - Executes `npm ci`, `npm test`, and `npm run build` in `packages/typescript-gpt-conversation`.
  - On push to `main`, publishes to npm only if `package.json` version changed and that version is not already published.
  - Uses repository secret `NPM_TOKEN` for npm authentication.
