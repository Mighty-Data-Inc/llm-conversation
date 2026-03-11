# @mightydatainc/gpt-conversation

Utilities for managing multi-shot LLM conversations and structured JSON responses with OpenAI's Responses API.

## Purpose and Rationale

This package exists to reduce the size, complexity, and repetitiveness of code used for interacting with LLM services -- specifically, with OpenAI's GPT.

OpenAI's Responses API is flexible, but application code often repeats the same plumbing:

- message shaping and role management
- retry and transient failure handling
- forcing machine-readable JSON output
- keeping conversation state coherent over multiple turns or "shots"

This package gives you small, composable building blocks for those recurring concerns. The design goal is to keep your app code focused on product logic while these utilities handle the repetitive conversation and formatting mechanics.

## Components And Why They Exist

| Component          | Why it exists                                                                                                               | When to use it                                                                                           |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `gptSubmit`        | Centralizes a robust "submit messages and return reply" workflow, including retries and optional structured output parsing. | One-off prompts or service-layer functions where you already manage message history yourself.            |
| `GptConversation`  | Wraps a message list with role-aware helpers and submit methods, so stateful chat flows stay readable and less error-prone. | Multi-turn workflows where you want to append/submit messages incrementally.                             |
| `JSONSchemaFormat` | Provides a compact TypeScript DSL to describe structured output schemas without hand-writing large JSON Schema objects.     | You need stricter contracts than "just return JSON" and want fields/types/ranges/enums defined up front. |

## Quick Start

### Stateless Prompt Submission With `gptSubmit`

Use this pattern as a wrapper around OpenAI's Responses API. gptSubmit provides error handling, "smart" retries, structured input/output type management, and shotgunning.

```ts
import OpenAI from 'openai';
import { gptSubmit } from '@mightydatainc/gpt-conversation';

const client = new OpenAI();

async function main(): Promise<void> {
  const story = await gptSubmit(
    [
      {
        role: 'user',
        content:
          'Write a short story about a raccoon who steals the Mona Lisa.',
      },
    ],
    client
  );

  console.log(story);
}

void main();
```

### Stateful Chat Flows With `GptConversation`

Use this pattern when your application relies on multi-stage (multi-"shot") conversation flows, especially if any of the shots are conditional.

```ts
import OpenAI from 'openai';
import { GptConversation } from '@mightydatainc/gpt-conversation';

const client = new OpenAI();
const conversation = new GptConversation(client);

const shouldIncludeSidekick = true;
const shouldEmphasizeCharacterDevelopment = true;

// The `submit*` methods make a call to the LLM, and take a few seconds to run.
// They'll return the string reply produced by the LLM.
let story = await conversation.submitUserMessage(
  'Write a short story about a raccoon who steals the Mona Lisa.'
);

if (shouldIncludeSidekick || shouldEmphasizeCharacterDevelopment) {
  // The `add*` methods append messages to the conversation without actually
  // sending a request to the LLM, allowing us to queue multiple instructions
  // or conditionally adjust conversation topology.
  conversation.addUserMessage(
    "That's a good first draft. But I'd now like you to enhance your story as follows."
  );
  if (shouldIncludeSidekick) {
    // Note that, in this example, we don't know if the LLM did or didn't include
    // a sidekick in its original draft. We're performing a blind multi-shot.
    // We're essentially saying, "Look, I don't know what you just wrote, but
    // we're willing to bet that, whatever it was, it needs a sidekick."
    conversation.addUserMessage(
      "- If the story doesn't already have a sidekick, add one."
    );
  }
  if (shouldEmphasizeCharacterDevelopment) {
    // Again, this blind multi-shot conversation is essentially structured to
    // implicitly flow with the understanding that we don't know what story
    // the LLM originally wrote -- but whatever it was, we bet it needs more
    // character development.
    conversation.addUserMessage(
      "- Focus more on the protagonist's character development."
    );
  }

  // Use a chain-of-thought stage to let the LLM talk through its intended changes.
  // This is what "thinking" models actually do under-the-hood. Here, you can get
  // specialized "thinking" performance for your own specific needs, by telling the
  // LLM exactly what it needs to deliberate with itself about.
  // This is a `submit*` method, which will actually send a call to the LLM.
  // Its reply will be implicitly added to the conversation history.
  await conversation.submitUserMessage(
    "Discuss how you'd go about revising your story to integrate these suggestions. " +
      "Don't actually write a new draft yet. Just talk about it for now."
  );

  story = await conversation.submitUserMessage(
    'Now emit your final draft of the story, starting with the title.'
  );
}

console.log(story);
```

## JSON Response Mode and `JSONSchemaFormat`

GPT has the ability to emit responses in JSON format -- and in fact, it even has the ability to enforce specific JSON schemas with concretely defined structures.

This functionality is extremely valuable in integrating AI capabilities into traditional procedural workflows, but it's overlooked by many developers due in part to the syntactic complexity of its invocation. Instead, many developers tend to take much more fragile approaches, such as using string parsing (e.g. regexes and substring matching) to extract answers from LLM responses -- a technique that often ends up falling back on prompt engineering to beg, plead, cajole, bribe, or threaten the AI to please just produce a properly formatted reply. This package's JSON response mode capabilities are designed to make AI-integrated data processing feel more like software engineering and less like an Inquisitorial confession session.

The function `gptSubmit` and the `GptConversation` class's `submit` method both take an optional named argument: `jsonResponse`. This can be set simply to `true` for a "lazy" JSON response using a structure that's described in plain English, in the bodies of prior messages. Or, it can be set to a structured format description object (with the help of `JSONSchemaFormat`) to ensure that your output will be guaranteed to conform to a schema of your specification.

### Unenforced ("advisory") JSON response: `jsonResponse=true`

The "lazy" approach to producing JSON output is to set the optional `jsonResponse` argument to `true`. This allows you to specify a desired JSON format by simply describing it in plain English in the body of your submitted message strings.

```ts
import OpenAI from 'openai';
import { GptConversation } from '@mightydatainc/gpt-conversation';

const client = new OpenAI();
const conversation = new GptConversation(client);

// The `submit*` methods actually send a real call to the LLM, and will
// take a few seconds to run. When it's finished, the story will implicitly
// get stored as part of the conversation history.
const story = await conversation.submitUserMessage(
  'Write a short story about a raccoon who steals the Mona Lisa.'
);

// We'll use an `add*` method, because the `submit*` methods are primarily
// convenience methods. The real workhorse is the `submit(...)` method itself, which
// exposes a finer array of options and variants. We'll call that next.
conversation.addUserMessage(`
Answer a few questions about the story you just wrote.
- What is the name of the protagonist?
- Where does the story take place?
- How many attempts at theft does the protagonist make during the course of the story?
- Does the protagonist ultimately succeed?

Provide your response as a JSON object using the following structure:
{
    "protagonist_name": (string)
    "city": (string, one of: "Paris", "New York", "Tokyo", "Other")
    "number_of_theft_attempts": (int, between 0 and 10)
    "do_they_ultimately_succeed": (boolean)
}
`);
await conversation.submit(undefined, undefined, { jsonResponse: true });

// In this example, we get the entire response object as a dict.
// Alternatively, we could use the helper method `getLastReplyDictField(...)`.
const questionnaire = conversation.getLastReplyDict();

console.log(story);
console.log('Protagonist: ', questionnaire['protagonist_name']);
console.log('City where the story takes place: ', questionnaire['city']);
console.log(
  'Number of theft attempts: ',
  questionnaire['number_of_theft_attempts']
);
console.log(
  'Ultimately successful? ',
  questionnaire['do_they_ultimately_succeed']
);
```

In this example, we made the AI come up with its own document and then answer structured questions about that document. However, in practice, you could of course submit your _own_ source document (e.g. in a call to `addUserMessage(...)`), and have the AI answer questions about it.

Some caveats to keep in mind when using this approach:

- You _must_ have a message in your conversation history that describes a JSON structure.
- The JSON structure is not enforced by any kind of calling framework. The LLM takes your JSON structure description "under advisement", but is not obligated to adhere to it in any way.
- Particularly large or complex structures can cause the LLM to hang.

### Structured JSON response with OpenAI json_schema: `jsonResponse={"format": {...}}`

To ensure that the JSON output is always consistent with a desired schema, OpenAI's API allows you to specify structured outputs. The full specification can be found here:

https://developers.openai.com/api/docs/guides/structured-outputs/

```ts
import OpenAI from 'openai';
import { GptConversation } from '@mightydatainc/gpt-conversation';

const client = new OpenAI();
const conversation = new GptConversation(client);

const story = await conversation.submitUserMessage(
  'Write a short story about a raccoon who steals the Mona Lisa.'
);

// We *could* add a separate user message telling the AI to answer a few questions,
// but honestly just submitting this JSON query is enough to make the AI "understand"
// what we want it to do.
await conversation.submit(undefined, undefined, {
  jsonResponse: {
    format: {
      type: 'json_schema',
      name: 'raccoon_story_questionnaire',
      description: 'Answer a few questions about this raccoon story.',
      strict: true,
      schema: {
        type: 'object',
        properties: {
          protagonist_name: {
            type: 'string',
            description: 'What is the name of the protagonist?',
          },
          city: {
            type: 'string',
            description: 'Where does the story take place?',
            enum: ['Paris', 'New York', 'Tokyo', 'Other'],
          },
          number_of_theft_attempts: {
            type: 'integer',
            description:
              'How many attempts at theft does the protagonist make during the course of the story?',
            minimum: 0,
            maximum: 10,
          },
          do_they_ultimately_succeed: {
            type: 'boolean',
            description: 'Does the protagonist ultimately succeed?',
          },
        },
        required: [
          'protagonist_name',
          'city',
          'number_of_theft_attempts',
          'do_they_ultimately_succeed',
        ],
        additionalProperties: false,
      },
    },
  },
});

console.log(story);

// Use the helper method `getLastReplyDictField(...)` to get the parsed JSON responses.
console.log(
  'Protagonist: ',
  conversation.getLastReplyDictField('protagonist_name')
);
console.log(
  'City where the story takes place: ',
  conversation.getLastReplyDictField('city')
);
console.log(
  'Number of theft attempts: ',
  conversation.getLastReplyDictField('number_of_theft_attempts')
);
console.log(
  'Ultimately successful? ',
  conversation.getLastReplyDictField('do_they_ultimately_succeed')
);
```

### Structured JSON response with JSONSchemaFormat: `jsonResponse=JSONSchemaFormat({...})`

While OpenAI's calling conventions are powerful and flexible, they are not particularly laconic or human-readable. As such, this package provides a specialized helper function called `JSONSchemaFormat`, which provides a translation layer that allows you to provide a `jsonResponse` argument in a convenient form of shorthand.

JSONSchemaFormat takes a data structure that "looks like" the structure you want returned. It's extremely flexible (to the point of being somewhat sloppy), and not quite as expressive as the real `json_schema` structure. However, it's much more compact and readable.

How to use `JSONSchemaFormat` shorthand:

- A field's value is simply the type that you want that field to be returned as. E.g. `"protagonist_name": String`
- If a field's name isn't self-explanatory, it can be specified as an array containing type and string, where the string is a description. E.g. `"city": [String, "What city does the story take place in?"]`
- If a field is a string, then that array can include a list specifying valid values. E.g. `"species": [String, "What is the species of the accomplice?", ["raccoon", "squirrel", "human", "pigeon", "other"]]`
- If a field is a number, then that array can include a 2-element tuple specifying min and max values (`null` means open-ended on one side). E.g. `"number_of_accomplices": [JSON_INTEGER, "How many accomplices did the protagonist have?", [0, 5]]`
- To make a field's value be returned as a list, present that field's value as a list.
- Fields can be objects, with this entire pattern nested multiple layers deep.

```ts
import OpenAI from 'openai';
import {
  GptConversation,
  JSON_INTEGER,
  JSONSchemaFormat,
} from '@mightydatainc/gpt-conversation';

const client = new OpenAI();
const conversation = new GptConversation(client);

const story = await conversation.submitUserMessage(
  'Write a short story about a raccoon who steals the Mona Lisa. ' +
    'Give the raccoon between 0 and 4 accomplices who help them in their heist, ' +
    'each of a different species.'
);

// We *could* add a separate user message telling the AI to answer a few questions,
// but honestly just submitting this JSON query is enough to make the AI "understand"
// what we want it to do.
await conversation.submit(undefined, undefined, {
  jsonResponse: JSONSchemaFormat({
    protagonist_name: String,
    city: [String, 'Where does this story take place?'],
    number_of_theft_attempts: [
      JSON_INTEGER,
      'How many attempts do they make during the course of the story?',
      [0, 10],
    ],
    do_they_ultimately_succeed: Boolean,
    accomplices: [
      {
        name: String,
        species: [
          String,
          'What species is this accomplice?',
          [
            'raccoon',
            'cat',
            'dog',
            'ferret',
            'squirrel',
            'pigeon',
            'human',
            'other',
          ],
        ],
        species_unusual: [
          String,
          "If the species is 'other', please specify it here. Otherwise, leave this blank.",
        ],
      },
    ],
  }),
});

console.log(story);

// Use the helper method `getLastReplyDictField(...)` to get the parsed JSON responses.
console.log(
  'Protagonist: ',
  conversation.getLastReplyDictField('protagonist_name')
);
console.log(
  'City where the story takes place: ',
  conversation.getLastReplyDictField('city')
);
console.log(
  'Number of theft attempts: ',
  conversation.getLastReplyDictField('number_of_theft_attempts')
);
console.log(
  'Ultimately successful? ',
  conversation.getLastReplyDictField('do_they_ultimately_succeed')
);

for (const accomplice of conversation.getLastReplyDictField(
  'accomplices'
) as Array<Record<string, unknown>>) {
  let species = String(accomplice['species']);
  if (species === 'other') {
    species = String(accomplice['species_unusual']) + ' (unusual)';
  }
  console.log(`Accomplice ${accomplice['name']} is a ${species}.`);
}
```

Remember that, ultimately, `JSONSchemaFormat` is just a translation function. It produces a data structure that conforms to the JSON schema expected by the OpenAI API.

## Shotgunning

The `gptSubmit` function and the `GptConversation` class's `submit` method take an optional `shotgun` argument. `shotgun` takes a numerical argument that specifies the number of "barrels" (parallel workers) to launch. Use this when you want to spend extra cost/latency for potentially better output quality.

```ts
import OpenAI from 'openai';
import { GptConversation } from '@mightydatainc/gpt-conversation';

const client = new OpenAI();
const conversation = new GptConversation(client);

await conversation.submitUserMessage(
  'Write a short story about a raccoon who steals the Mona Lisa.'
);
conversation.addUserMessage(
  'Think of one improvement that can be made to the story, and write a revised version of the story that incorporates that improvement.'
);
await conversation.submit(undefined, undefined, { shotgun: 3 });
const story = conversation.getLastReplyStr();

console.log(story);
```

Shotgunning is particularly useful when you have a multi-component AI-related task where any given component tends to pass, but at least one component somewhere in the task tends to fail -- i.e. a different component fails every time. Under such circumstances, running 3 or 4 parallel workers could dramatically improve the reliability of the final output. For details, study the shotgunning unit test.

## API

### GptConversation

Stateful conversation container built on top of `gptSubmit`.

What it does:

- Stores message history as a mutable list of `{role, content}` message objects.
- Provides role-specific helpers for appending and submitting messages.
- Tracks the latest assistant result in `lastReply` for easy typed access.

Initialization:

- `openaiClient`: OpenAI client (or compatible object) used for submissions.
- `messages`: Optional initial message list.
- `model`: Optional default model used by `submit(...)` when no per-call model is provided.

Core submission behavior:

- `submit(...)` optionally appends a message, then calls `gptSubmit(...)` with current history.
- If `message` is a `Record` containing `format` and `jsonResponse` is not explicitly provided, that object is used as `jsonResponse`.
- Supports per-call `model`, `jsonResponse`, and `shotgun` options.
- Appends the final assistant reply back into conversation history and updates `lastReply`.

Message helpers:

- `addMessage(role, content)` plus role helpers: `addUserMessage`, `addAssistantMessage`, `addSystemMessage`, `addDeveloperMessage`.
- `addImage(role, text, imageDataUrl)` for multimodal text+image message payloads.
- Non-string `Record` content is serialized to JSON text; list content is preserved for multimodal payloads.

Submit convenience methods:

- `submitMessage(role, content)`
- `submitUserMessage(content)`
- `submitAssistantMessage(content)`
- `submitSystemMessage(content)`
- `submitDeveloperMessage(content)`
- `submitImage(role, text, imageDataUrl)`

Inspection and utility methods:

- `getLastMessage()`
- `getMessagesByRole(role)`
- `getLastReplyStr()`
- `getLastReplyDict()`
- `getLastReplyDictField(fieldName, defaultValue=null)`
- `toDictList()`
- `clone()`
- `assignMessages(messages)`

Failure behavior:

- Raises `Error` if `submit(...)` is called without an `openaiClient`.
- Exceptions from `gptSubmit(...)` are propagated.

### gptSubmit

Primary stateless submit helper for OpenAI Responses API calls.

What it does:

- Submits a list of OpenAI-style messages and returns the model reply.
- Injects a fresh `!DATETIME` system message on each call.
- Optionally prepends a `systemAnnouncementMessage` before the datetime message.
- Supports retry/backoff behavior for transient failures.
- Supports JSON response modes and optional shotgunning.

Arguments:

- `messages`: Conversation payload in OpenAI message format.
- `openaiClient`: OpenAI client (or compatible object) exposing `responses.create(...)`.
- `model`: Optional model override. Defaults to the package smart model.
- `jsonResponse`: Output mode control.
- `systemAnnouncementMessage`: Optional additional top-level system instruction.
- `retryLimit`: Maximum retry attempts for retryable failures.
- `retryBackoffTimeSeconds`: Backoff delay between retry attempts.
- `shotgun`: Number of parallel worker barrels to launch (`>1` enables reconciliation mode).
- `warningCallback`: Optional callback for non-fatal warnings/retry notices.

`jsonResponse` modes:

- `null` or `false`: Return plain text.
- `true`: Request JSON object mode (`{"format": {"type": "json_object"}}`).
- `Record`: Use provided OpenAI text-format config.
- `string`: Parse as JSON and use as OpenAI text-format config.

Return value:

- Text mode: `string` (trimmed).
- JSON mode: parsed JSON value from the model output (commonly `Record` or `array`, but can also be scalar JSON values).

Failure behavior:

- Retries transient OpenAI API failures up to `retryLimit` with backoff.
- Retries JSON parsing failures in JSON mode.
- Raises immediately (no retry) for non-retryable protocol/request failures.

## Installation and usage

```bash
npm install @mightydatainc/gpt-conversation openai
```

```typescript
import {
  gptSubmit,
  GptConversation,
  JSONSchemaFormat,
} from '@mightydatainc/gpt-conversation';
```

Requires Node.js runtime version >=24 (ESM + async/await).
