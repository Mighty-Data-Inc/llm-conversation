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
