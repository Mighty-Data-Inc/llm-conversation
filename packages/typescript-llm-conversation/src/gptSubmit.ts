import {
  currentDatetimeSystemMessage,
  isRecord,
  parseFirstJsonValue,
} from './helpers.js';

export const GPT_MODEL_CHEAP = 'gpt-4.1-nano';
export const GPT_MODEL_SMART = 'gpt-4.1';
export const GPT_MODEL_VISION = 'gpt-4.1';

const GPT_RETRY_LIMIT_DEFAULT = 5;
const GPT_RETRY_BACKOFF_TIME_SECONDS_DEFAULT = 30;

/**
 * Minimal interface describing an OpenAI client that can submit requests via
 * `responses.create`. Accepting this interface instead of the concrete SDK
 * class keeps callers decoupled from the SDK version and makes testing easier.
 */
export interface OpenAIClientLike {
  responses: {
    create: (args: any, options?: any) => Promise<any> | any;
  };
}

/**
 * Options controlling the behaviour of {@link gptSubmit}.
 *
 * @property model - The OpenAI model ID to use. Defaults to {@link GPT_MODEL_SMART}.
 * @property jsonResponse - When `true`, requests a plain JSON object response.
 *   When a `Record` or JSON string, that value is forwarded as the `text` format
 *   parameter (i.e. a JSON schema). Defaults to `undefined` (plain text).
 * @property systemAnnouncementMessage - An optional system message prepended to
 *   every request, before all other messages.
 * @property retryLimit - Maximum number of retry attempts on recoverable errors.
 *   Defaults to {@link GPT_RETRY_LIMIT_DEFAULT}.
 * @property retryBackoffTimeSeconds - Seconds to wait between retries for
 *   OpenAI API errors. Defaults to {@link GPT_RETRY_BACKOFF_TIME_SECONDS_DEFAULT}.
 * @property shotgun - When greater than 1, the request is sent to this many
 *   parallel worker calls and the results are reconciled by a final call.
 * @property warningCallback - Optional callback invoked with a human-readable
 *   warning string on recoverable errors and incomplete responses.
 */
export interface GptSubmitOptions {
  model?: string;
  jsonResponse?: boolean | Record<string, unknown> | string;
  systemAnnouncementMessage?: string;
  retryLimit?: number;
  retryBackoffTimeSeconds?: number;
  shotgun?: number;
  warningCallback?: (message: string) => void;
}

/**
 * Returns `true` when `error` is an OpenAI SDK error that is worth retrying
 * (e.g. rate-limit or transient server errors). Detection is based on the
 * error's `name` property containing `"OpenAI"` or `"APIError"`.
 *
 * @param error - The caught value to inspect.
 * @returns `true` if the error is a retryable OpenAI API error.
 */
function isRetryableOpenAIError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }

  const name = error.name || '';
  return name.includes('OpenAI') || name.includes('APIError');
}

/**
 * Implements the "shotgun" strategy for {@link gptSubmit}: sends `numBarrels`
 * parallel requests with identical inputs, then asks the model to examine all
 * responses and reconcile them into a single authoritative answer.
 *
 * This improves output quality for tasks where the model benefits from
 * exploring multiple reasoning paths simultaneously.
 *
 * @param messages - The conversation history to send to each worker.
 * @param openaiClient - The OpenAI client instance.
 * @param options - Submit options (the `shotgun` field is stripped before
 *   forwarding to avoid infinite recursion).
 * @param numBarrels - Number of parallel worker requests to fire.
 * @returns The reconciled response from the model.
 */
const gptSubmitShotgun = async (
  messages: unknown[],
  openaiClient: OpenAIClientLike,
  options: GptSubmitOptions,
  numBarrels: number
): Promise<
  string | Record<string, unknown> | unknown[] | number | boolean | null
> => {
  messages = JSON.parse(JSON.stringify(messages));
  options = JSON.parse(JSON.stringify(options));

  // Delete the shotgun option from the options passed to gptSubmitShotgun to avoid infinite recursion.
  delete options.shotgun;

  if (numBarrels <= 1) {
    // No need for shotgun logic if only 1 barrel requested.
    return gptSubmit(messages, openaiClient, options);
  }

  // Remove the shotgun option before passing to gptSubmit to avoid
  // infinite recursion!
  delete options.shotgun;

  const convoShotgun: unknown[][] = [];
  for (let i = 0; i < numBarrels; i += 1) {
    const convoBarrel = JSON.parse(JSON.stringify(messages));
    convoShotgun.push(convoBarrel);
  }
  const resultsRaw: unknown[] = await Promise.all(
    convoShotgun.map((convoBarrel) =>
      gptSubmit(convoBarrel, openaiClient, options)
    )
  );
  const resultStrings = resultsRaw.map((result) => JSON.stringify(result));

  // Build the reconciliation conversation on top of a fresh copy of the original messages.
  const reconcileMessages: unknown[] = JSON.parse(JSON.stringify(messages));

  reconcileMessages.push({
    role: 'system',
    content: `
SYSTEM MESSAGE:
In order to produce better results, we submitted this request/question/command/etc.
to ${numBarrels} worker threads in parallel.
The system will now present each of their responses, wrapped in JSON.
The user or developer will not see these responses -- they are only for you, the assistant, 
to examine and reconcile. Think of them as brainstorming or scratchpads.
`,
  });
  resultStrings.forEach((resultString, index) => {
    reconcileMessages.push({
      role: 'system',
      content: `WORKER ${index + 1} RESPONSE:\n\n\n${resultString}`,
    });
  });

  reconcileMessages.push({
    role: 'system',
    content: `
Focus on the differences and discrepancies between the workers' responses. Where do they agree?
Where do they disagree? In the areas where they disagree, which worker's argument is most
consistent with the data you've been shown?

Remember, this is an adjudication, not a democracy -- you should carefully examine the data
presented in the conversation and use your best judgment to determine which worker is most
likely to be correct, even if they're in the minority. Evaluate their answers carefully against
the source data. If multiple workers produced different answers, then clearly there is something
subtle, deceptive, or misleading about the question or the data, and you should be especially
careful to scrutinize the workers' reasoning and the evidence they present for their answers.
At least one of them must be wrong; don't fall into the same trap he did.
`,
  });

  // This is a chain-of-thought ponderance. We specifically do not want a JSON
  // response here, because we want the model to be able to freely reason and
  // draw conclusions without being constrained by JSON syntax. The final answer
  // will be produced in the next step, where we will ask the model to produce
  // the same format as it was originally given (text or JSON).
  const sPonderReply = await gptSubmit(reconcileMessages, openaiClient, {
    ...options,
    jsonResponse: undefined,
  });
  reconcileMessages.push({ role: 'assistant', content: sPonderReply });

  reconcileMessages.push({
    role: 'system',
    content: `
Having seen and reconciled the workers' responses, you are now ready to craft a proper reply to
the question/request/command/etc. This response that you craft now is the one that will be
presented to the user or developer -- it should not directly reference the workers' responses,
but should instead be a fully self-contained and complete answer that draws on the insights
you've gained from examining the workers' responses.
`,
  });

  return gptSubmit(reconcileMessages, openaiClient, options);
};

/**
 * Submits a conversation to the OpenAI Responses API and returns the model's
 * reply.
 *
 * - Prepends a current-datetime system message to every request.
 * - Optionally enforces a JSON response format (plain JSON or a full JSON
 *   schema via `options.jsonResponse`).
 * - Automatically retries on JSON parse errors and retryable OpenAI API errors
 *   up to `options.retryLimit` times.
 * - Delegates to {@link gptSubmitShotgun} when `options.shotgun > 1`.
 *
 * @param messages - The conversation history, including any prior assistant
 *   turns. Each element should be a message object with at minimum `role` and
 *   `content` fields.
 * @param openaiClient - An {@link OpenAIClientLike} instance used to call the
 *   API.
 * @param options - Optional settings controlling model, JSON mode, retries,
 *   and shotgun parallelism.
 * @returns The model's response. A plain `string` when `jsonResponse` is
 *   falsy; otherwise a parsed JSON value (`Record`, `unknown[]`, `number`,
 *   `boolean`, or `null`).
 * @throws The last encountered error after all retry attempts are exhausted,
 *   or immediately for non-retryable errors.
 */
export async function gptSubmit(
  messages: unknown[],
  openaiClient: OpenAIClientLike,
  options: GptSubmitOptions = {}
): Promise<
  string | Record<string, unknown> | unknown[] | number | boolean | null
> {
  if (options.shotgun && options.shotgun > 1) {
    return await gptSubmitShotgun(
      messages,
      openaiClient,
      options,
      options.shotgun
    );
  }

  const model = options.model || GPT_MODEL_SMART;
  const retryLimit = options.retryLimit ?? GPT_RETRY_LIMIT_DEFAULT;
  const retryBackoffTimeSeconds =
    options.retryBackoffTimeSeconds ?? GPT_RETRY_BACKOFF_TIME_SECONDS_DEFAULT;

  let failedError: unknown = null;

  let openaiTextParam: Record<string, unknown> | undefined;
  if (options.jsonResponse) {
    if (typeof options.jsonResponse === 'boolean') {
      openaiTextParam = { format: { type: 'json_object' } };
    } else if (typeof options.jsonResponse === 'string') {
      openaiTextParam = JSON.parse(options.jsonResponse) as Record<
        string,
        unknown
      >;
    } else if (isRecord(options.jsonResponse)) {
      openaiTextParam = JSON.parse(
        JSON.stringify(options.jsonResponse)
      ) as Record<string, unknown>;

      const format = openaiTextParam.format;
      if (isRecord(format) && typeof format.description === 'string') {
        format.description =
          `${format.description}\n\nABSOLUTELY NO UNICODE ALLOWED. ` +
          `Only use typeable keyboard characters. Do not try to circumvent this rule ` +
          `with escape sequences, backslashes, or other tricks. Use double dashes (--), ` +
          `straight quotes (") and single quotes (') instead of em-dashes, en-dashes, ` +
          `and curly versions.`.trim();
      }
    }
  }

  const filteredMessages = messages.filter((message) => {
    if (!isRecord(message)) {
      return true;
    }
    const role = message.role;
    const content = message.content;
    return !(
      role === 'system' &&
      typeof content === 'string' &&
      content.startsWith('!DATETIME:')
    );
  });

  let preparedMessages: unknown[] = [
    currentDatetimeSystemMessage(),
    ...filteredMessages,
  ];

  if (
    options.systemAnnouncementMessage &&
    options.systemAnnouncementMessage.trim()
  ) {
    preparedMessages = [
      { role: 'system', content: options.systemAnnouncementMessage.trim() },
      ...preparedMessages,
    ];
  }

  for (let index = 0; index < retryLimit; index += 1) {
    let llmReply = '';

    try {
      const payload: {
        model: string;
        input: unknown[];
        text?: Record<string, unknown>;
      } = {
        model,
        input: preparedMessages,
      };
      if (openaiTextParam) {
        payload.text = openaiTextParam;
      }

      const llmResponse = await openaiClient.responses.create(payload);

      if (llmResponse.error && options.warningCallback) {
        options.warningCallback(
          `ERROR: OpenAI API returned an error: ${llmResponse.error}`
        );
      }
      if (llmResponse.incomplete_details && options.warningCallback) {
        options.warningCallback(
          `ERROR: OpenAI API returned incomplete details: ${llmResponse.incomplete_details}`
        );
      }

      llmReply = llmResponse.output_text.trim();

      if (!options.jsonResponse) {
        return `${llmReply}`;
      }

      return parseFirstJsonValue(llmReply);
    } catch (error) {
      if (error instanceof SyntaxError) {
        failedError = error;
        if (options.warningCallback) {
          options.warningCallback(
            `JSON decode error:\n\n${error}.\n\nRaw text of LLM Reply:\n${llmReply}\n\nRetrying (attempt ${index + 1} of ${retryLimit}) immediately...`
          );
        }
        continue;
      }

      if (isRetryableOpenAIError(error)) {
        failedError = error;
        if (options.warningCallback) {
          options.warningCallback(
            `OpenAI API error:\n\n${error}.\n\nRetrying (attempt ${index + 1} of ${retryLimit}) in ${retryBackoffTimeSeconds} seconds...`
          );
        }
        // Sleep for retryBackoffTimeSeconds before the next retry attempt.
        await new Promise((resolve) =>
          setTimeout(resolve, retryBackoffTimeSeconds * 1000)
        );
        continue;
      }

      throw error;
    }
  }

  if (failedError) {
    throw failedError;
  }

  throw new Error('Unknown error occurred in gptSubmit');
}
