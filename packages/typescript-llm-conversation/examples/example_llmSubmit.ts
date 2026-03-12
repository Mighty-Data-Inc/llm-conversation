import OpenAI from 'openai';
import { llmSubmit } from '@mightydatainc/llm-conversation';

const client = new OpenAI();

async function main(): Promise<void> {
  const story = await llmSubmit(
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
