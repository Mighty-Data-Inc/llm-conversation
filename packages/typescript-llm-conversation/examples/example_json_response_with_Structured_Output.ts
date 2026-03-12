import OpenAI from 'openai';
import { LLMConversation } from '@mightydatainc/llm-conversation';

const client = new OpenAI();
const conversation = new LLMConversation(client);

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
