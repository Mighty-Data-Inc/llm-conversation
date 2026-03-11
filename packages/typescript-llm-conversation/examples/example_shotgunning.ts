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
