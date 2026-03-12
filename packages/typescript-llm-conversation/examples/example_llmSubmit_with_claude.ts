import Anthropic from '@anthropic-ai/sdk';
import { llmSubmit } from '@mightydatainc/llm-conversation';

const client = new Anthropic();

const reply = await llmSubmit(
  [
    {
      role: 'user',
      content:
        'Summarize the causes of the French Revolution in one concise paragraph.',
    },
  ],
  client
);

console.log(reply);
