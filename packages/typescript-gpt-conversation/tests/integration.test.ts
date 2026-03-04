import { readFileSync } from 'fs';
import { join } from 'path';
import { OpenAI } from 'openai';
import { describe, expect, it } from 'vitest';
import {
  GptConversation,
  GptConversationOptions,
} from '../src/gptConversation.js';
import { JSONSchemaFormat } from '../src/jsonSchemaFormat.js';
import { GPT_MODEL_VISION } from '../src/functions.js';

const OPENAI_API_KEY = process.env.OPENAI_API_KEY?.trim();
if (!OPENAI_API_KEY) {
  throw new Error(
    'OPENAI_API_KEY is required for live API tests. Configure your test environment to provide it.'
  );
}

const createClient = (): OpenAI =>
  new OpenAI({
    apiKey: OPENAI_API_KEY,
  });

describe('integration tests (live API)', () => {
  it('should repeat Hello World', async () => {
    const openaiClient = createClient();
    const convo = new GptConversation([], { openaiClient });

    convo.addUserMessage(`
This is a test to see if I'm correctly calling the OpenAI API to invoke GPT.
If you can see this, please respond with "Hello World" -- just like that,
with no additional text or explanation. Do not include punctuation or quotation
marks. Emit only the words "Hello World", capitalized as shown.
`);
    await convo.submit();

    const reply = convo.getLastReplyStr();
    expect(reply).toBe('Hello World');
  }, 180000);

  it('should invoke an LLM with some nominal intelligence', async () => {
    const openaiClient = createClient();
    const convo = new GptConversation([], { openaiClient });

    // We'll use this opportunity to test the submitUserMessage convenience method.
    await convo.submitUserMessage(`
I'm conducting a test of my REST API response parsing systems.
If you can see this, please reply with the capital of France.
Reply *only* with the name of the city, with no additional text, punctuation,
or explanation. I'll be comparing your output string to a standard known
value, so it's important to the integrity of my system that the *only*
response you produce be *just* the name of the city. Standard capitalization
please -- first letter capitalized, all other letters lower-case.
`);

    const reply = convo.getLastReplyStr();
    expect(reply).toBe('Paris');
  });

  it('should reply with a general-form JSON object', async () => {
    const openaiClient = createClient();
    const convo = new GptConversation([], { openaiClient });

    convo.addUserMessage(`
This is a test to see if I'm correctly calling the OpenAI API to invoke GPT.

Please reply with the following JSON object, exactly as shown:

{
  "text": "Hello World",
  "success": true,
  "sample_array_data": [1, 2, {"nested_key": "nested_value"}]
}
`);
    await convo.submit(undefined, undefined, {
      jsonResponse: true,
    });

    const replyObj = convo.getLastReplyDict();

    // We'll test the structure of the JSON object piecemeal.
    expect(replyObj).toHaveProperty('text', 'Hello World');
    expect(replyObj).toHaveProperty('success', true);
    expect(replyObj).toHaveProperty('sample_array_data');
    expect(Array.isArray(replyObj.sample_array_data)).toBe(true);
    expect(replyObj.sample_array_data).toHaveLength(3);
    const arr = replyObj.sample_array_data as unknown[];
    expect(arr[0]).toBe(1);
    expect(arr[1]).toBe(2);
    expect(arr[2]).toHaveProperty('nested_key', 'nested_value');

    // While we're here, we'll also test to make sure that the shortcut
    // accessors also work.
    expect(convo.getLastReplyDictField('text')).toBe('Hello World');
    expect(convo.getLastReplyDictField('success')).toBe(true);
    expect(convo.getLastReplyDictField('sample_array_data')).toHaveLength(3);
  }, 180000);

  it('should reply with structured JSON using JSON schema spec', async () => {
    const openaiClient = createClient();
    const convo = new GptConversation([], { openaiClient });

    const schema = {
      format: {
        type: 'json_schema',
        strict: true,
        name: 'TestSchema',
        description: 'A test schema for structured JSON response',
        schema: {
          type: 'object',
          properties: {
            text: { type: 'string' },
            success: { type: 'boolean' },
            sample_array_data: {
              type: 'array',
              items: { type: 'number' },
            },
            nested_dict: {
              type: 'object',
              properties: {
                nested_key: { type: 'string' },
              },
              required: ['nested_key'],
              additionalProperties: false,
            },
          },
          required: ['text', 'success', 'sample_array_data', 'nested_dict'],
          additionalProperties: false,
        },
      },
    };

    convo.addUserMessage(`
Please reply with a JSON object that contains the following data:

Success flag: true
Text: "Hello World"
Sample array data (2 elements long):
    Element 0: 5
    Element 1: 33
Nested dict (1 item long):
    Value under "nested_key": "foobar"
`);
    await convo.submit(undefined, undefined, {
      jsonResponse: schema,
    });

    expect(convo.getLastReplyDictField('success')).toBe(true);
    expect(convo.getLastReplyDictField('text')).toBe('Hello World');

    const nestedDict = convo.getLastReplyDictField('nested_dict') as Record<
      string,
      unknown
    >;
    expect(nestedDict).toHaveProperty('nested_key', 'foobar');
    expect(Object.keys(nestedDict)).toHaveLength(1);

    const sampleArrayData = convo.getLastReplyDictField(
      'sample_array_data'
    ) as unknown[];
    expect(sampleArrayData).toHaveLength(2);
    expect(sampleArrayData[0]).toBe(5);
    expect(sampleArrayData[1]).toBe(33);
  }, 180000);

  it('should reply with structured JSON using JSON formatter shorthand', async () => {
    const openaiClient = createClient();
    const convo = new GptConversation([], { openaiClient });

    const schema = JSONSchemaFormat(
      'TestSchema',
      {
        text: String,
        success: Boolean,
        sample_array_data: [Number],
        nested_dict: {
          nested_key: String,
        },
      },
      'A test schema for structured JSON response'
    );

    convo.addUserMessage(`
Please reply with a JSON object that contains the following data:

Success flag: true
Text: "Hello World"
Sample array data (2 elements long):
    Element 0: 5
    Element 1: 33
Nested dict (1 item long):
    Value under "nested_key": "foobar"
`);
    await convo.submit(undefined, undefined, {
      jsonResponse: schema,
    });

    expect(convo.getLastReplyDictField('success')).toBe(true);
    expect(convo.getLastReplyDictField('text')).toBe('Hello World');

    const nestedDict = convo.getLastReplyDictField('nested_dict') as Record<
      string,
      unknown
    >;
    expect(nestedDict).toHaveProperty('nested_key', 'foobar');
    expect(Object.keys(nestedDict)).toHaveLength(1);

    const sampleArrayData = convo.getLastReplyDictField(
      'sample_array_data'
    ) as unknown[];
    expect(sampleArrayData).toHaveLength(2);
    expect(sampleArrayData[0]).toBe(5);
    expect(sampleArrayData[1]).toBe(33);
  }, 180000);

  it('should be capable of image recognition', async () => {
    const openaiClient = createClient();
    const convo = new GptConversation([], {
      openaiClient,
      model: GPT_MODEL_VISION,
    });

    // Load the image ./fixtures/phoenix.png
    const pngBuffer = readFileSync(join(__dirname, 'fixtures', 'phoenix.png'));
    const imgDataUrl = `data:image/png;base64,${pngBuffer.toString('base64')}`;

    const gptMsgWithImage = {
      role: 'user',
      content: [
        {
          type: 'input_text',
          text: 'An image submitted by a user, needing identification',
        },
        {
          type: 'input_image',
          image_url: imgDataUrl,
          detail: 'high',
        },
      ],
    };
    // We don't use our convenience methods here.
    // We just build the message directly for now.
    // We'll test the image message add methods later.
    convo.push(gptMsgWithImage);

    convo.addUserMessage('What is this a picture of?');

    await convo.submit(undefined, undefined, {
      jsonResponse: JSONSchemaFormat(
        'ImageIdentification',
        {
          image_subject_enum: [
            'house',
            'chair',
            'boat',
            'car',
            'cat',
            'dog',
            'telephone',
            'duck',
            'city_skyline',
            'still_life',
            'bed',
            'headphones',
            'skull',
            'photo_camera',
            'unknown',
          ],
        },
        'A test schema for image identification response'
      ),
    });

    expect(convo.getLastReplyDictField('image_subject_enum')).toBe('cat');
  }, 180000);
});
