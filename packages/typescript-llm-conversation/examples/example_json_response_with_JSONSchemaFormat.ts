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
