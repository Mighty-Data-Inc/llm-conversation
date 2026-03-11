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
