import { describe, expect, it } from 'vitest';

import { GptConversation } from '../src/gptConversation.js';

class FakeResponse {
  output_text: any;
  error: any;
  incomplete_details: any;

  constructor(
    outputText = '',
    error: any = null,
    incompleteDetails: any = null
  ) {
    this.output_text = outputText;
    this.error = error;
    this.incomplete_details = incompleteDetails;
  }
}

class FakeResponsesAPI {
  sideEffects: any[];
  createCalls: Array<Record<string, unknown>>;

  constructor(sideEffects: any[] = []) {
    this.sideEffects = [...sideEffects];
    this.createCalls = [];
  }

  create(kwargs: Record<string, unknown>) {
    this.createCalls.push(kwargs);

    if (!this.sideEffects.length) {
      return new FakeResponse();
    }

    const next = this.sideEffects.shift();
    if (next instanceof Error) {
      throw next;
    }
    return next;
  }
}

class FakeOpenAIClient {
  responses: FakeResponsesAPI;

  constructor(sideEffects: any[] = []) {
    this.responses = new FakeResponsesAPI(sideEffects);
  }
}

describe('GptConversation', () => {
  it('defaults to empty state with no client or model', () => {
    const conversation = new GptConversation();

    expect(conversation).toEqual([]);
    expect(conversation.openaiClient).toBeUndefined();
    expect(conversation.model).toBeUndefined();
    expect(conversation.lastReply).toBeNull();
  });

  it('assignMessages replaces contents and returns self', () => {
    const conversation = new GptConversation(undefined, [
      { role: 'user', content: 'old' },
    ]);
    const next = [
      { role: 'system', content: 'new' },
      { role: 'user', content: 'hello' },
    ];

    const returned = conversation.assignMessages(next);

    expect(returned).toBe(conversation);
    expect(conversation).toEqual(next);
  });

  it('clone deep copies messages instead of preserving message references', () => {
    const client = new FakeOpenAIClient();
    const conversation = new GptConversation(client, [
      {
        role: 'user',
        content: 'foo',
      },
    ]);

    const cloned = conversation.clone();

    expect(cloned).not.toBe(conversation);
    expect(cloned).toEqual(conversation);

    conversation[0].content = 'bar';
    expect(cloned[0].content).toBe('foo');
  });

  it('clone copies client and model', () => {
    const client = new FakeOpenAIClient();
    const conversation = new GptConversation(client, undefined, 'gpt-custom');

    const cloned = conversation.clone();

    expect(cloned.openaiClient).toBe(client);
    expect(cloned.model).toBe('gpt-custom');
  });

  it('clone deep-copies lastReply', () => {
    const client = new FakeOpenAIClient();
    const conversation = new GptConversation(client, [
      { role: 'user', content: 'hello' },
    ]);
    conversation.lastReply = { result: 'ok' };

    const cloned = conversation.clone();

    expect(cloned.lastReply).toEqual({ result: 'ok' });
    expect(cloned.lastReply).not.toBe(conversation.lastReply);
    (conversation.lastReply as any).result = 'changed';
    expect(cloned.lastReply).toEqual({ result: 'ok' });
  });

  it('addMessage serializes object content as pretty JSON', () => {
    const conversation = new GptConversation();

    conversation.addMessage('user', { a: 1 });

    expect(conversation[0]).toEqual({
      role: 'user',
      content: '{\n  "a": 1\n}',
    });
  });

  it('role-specific helpers add expected role labels', () => {
    const conversation = new GptConversation();

    conversation.addUserMessage('u');
    conversation.addAssistantMessage('a');
    conversation.addSystemMessage('s');
    conversation.addDeveloperMessage('d');

    expect(conversation.map((m) => m.role)).toEqual([
      'user',
      'assistant',
      'system',
      'developer',
    ]);
  });

  it('addImage appends a multi-modal message with text and image parts', () => {
    const conversation = new GptConversation();
    const imgDataUrl = 'data:image/png;base64,abc123';

    const returned = conversation.addImage(
      'user',
      'Describe this.',
      imgDataUrl
    );

    expect(returned).toBe(conversation);
    expect(conversation[0]).toEqual({
      role: 'user',
      content: [
        { type: 'input_text', text: 'Describe this.' },
        { type: 'input_image', image_url: imgDataUrl, detail: 'high' },
      ],
    });
  });

  it('submit throws if no openai client is configured', async () => {
    const conversation = new GptConversation(undefined, [
      { role: 'user', content: 'hello' },
    ]);

    await expect(conversation.submit()).rejects.toThrow(
      'OpenAI client is not set. Please provide an OpenAI client.'
    );
  });

  it('submit appends user message then assistant reply', async () => {
    const client = new FakeOpenAIClient([new FakeResponse('assistant reply')]);
    const conversation = new GptConversation(client);

    const result = await conversation.submit('hello');

    expect(result).toBe('assistant reply');
    expect(conversation).toEqual([
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: 'assistant reply' },
    ]);
    expect(conversation.lastReply).toBe('assistant reply');
  });

  it('submit infers jsonResponse and role from dict-like message', async () => {
    const client = new FakeOpenAIClient([new FakeResponse('{"ok": true}')]);
    const conversation = new GptConversation(client);

    const message = {
      format: { type: 'json_object' },
      role: 'developer',
      content: 'Return JSON only.',
    };

    const result = await conversation.submit(message);

    expect(result).toEqual({ ok: true });
    expect(conversation[0]).toEqual({
      role: 'developer',
      content: 'Return JSON only.',
    });
    expect(conversation[1]).toEqual({
      role: 'assistant',
      content: '{\n  "ok": true\n}',
    });
  });

  it('submit passes through array content for multi-modal messages', async () => {
    const client = new FakeOpenAIClient([new FakeResponse('Nice image!')]);
    const conversation = new GptConversation(client);

    const imgDataUrl = 'data:image/png;base64,abc123';
    const message = {
      role: 'user',
      content: [
        { type: 'input_text', text: "Hello World. Here's an image." },
        { type: 'input_image', image_url: imgDataUrl, detail: 'high' },
      ],
    };

    const result = await conversation.submit(message);

    expect(result).toBe('Nice image!');
    expect(conversation[0]).toEqual({
      role: 'user',
      content: [
        { type: 'input_text', text: "Hello World. Here's an image." },
        { type: 'input_image', image_url: imgDataUrl, detail: 'high' },
      ],
    });
    expect(conversation[1]).toEqual({
      role: 'assistant',
      content: 'Nice image!',
    });
  });

  it('submit wrapper methods add role and return reply', async () => {
    const client = new FakeOpenAIClient([
      new FakeResponse('r1'),
      new FakeResponse('r2'),
      new FakeResponse('r3'),
      new FakeResponse('r4'),
      new FakeResponse('r5'),
    ]);
    const conversation = new GptConversation(client);

    await expect(conversation.submitMessage('system', 'm1')).resolves.toBe(
      'r1'
    );
    await expect(conversation.submitUserMessage('m2')).resolves.toBe('r2');
    await expect(conversation.submitAssistantMessage('m3')).resolves.toBe('r3');
    await expect(conversation.submitSystemMessage('m4')).resolves.toBe('r4');
    await expect(conversation.submitDeveloperMessage('m5')).resolves.toBe('r5');

    expect(
      conversation.filter((_, index) => index % 2 === 0).map((m) => m.role)
    ).toEqual(['system', 'user', 'assistant', 'system', 'developer']);
  });

  it('lastReply accessors enforce expected types', () => {
    const conversation = new GptConversation();

    // NOTE: This is something we should never do in practice.
    // We should never directly set lastReply in a GptConversation object.
    conversation.addAssistantMessage('hello');

    expect(conversation.getLastReplyStr()).toBe('hello');
    expect(conversation.getLastReplyDict()).toEqual({});

    conversation.addAssistantMessage({ x: 10, nested: { y: 1 } });

    expect(conversation.getLastReplyStr()).toBe('');
    const cloned = conversation.getLastReplyDict();
    expect(cloned).toEqual({ x: 10, nested: { y: 1 } });
    (cloned.nested as any).y = 2;
    expect((conversation.lastReply as any).nested.y).toBe(1);
    expect(conversation.getLastReplyDictField('x')).toBe(10);
    expect(conversation.getLastReplyDictField('missing', 99)).toBe(99);

    conversation.addAssistantMessage('not dict');
    expect(conversation.getLastReplyDictField('x', 99)).toBeNull();
  });
});
