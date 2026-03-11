import {
  currentDatetimeSystemMessage,
  isRecord,
  parseFirstJsonValue,
} from './helpers.js';
import {
  AIClientLike,
  getModelName,
  identifyLLMProvider,
  LLM_RETRY_BACKOFF_TIME_SECONDS_DEFAULT,
  LLM_RETRY_LIMIT_DEFAULT,
} from './llmProviders.js';

/**
 * Options controlling the behaviour of {@link llmSubmit}.
 *
 * @property model - The model ID to use. If not specified, defaults to the "smart"
 *   tier model for the detected LLM provider (e.g. `gpt-4.1` for OpenAI).
 * @property jsonResponse - When `true`, requests a plain JSON object response.
 *   When a `Record` or JSON string, that value is forwarded as the `text` format
 *   parameter (i.e. a JSON schema). Defaults to `undefined` (plain text).
 * @property systemAnnouncementMessage - An optional system message prepended to
 *   every request, before all other messages.
 * @property retryLimit - Maximum number of retry attempts on recoverable errors.
 *   Defaults to {@link GPT_RETRY_LIMIT_DEFAULT}.
 * @property retryBackoffTimeSeconds - Seconds to wait between retries for
 *   recoverable API errors. Defaults to {@link GPT_RETRY_BACKOFF_TIME_SECONDS_DEFAULT}.
 * @property shotgun - When greater than 1, the request is sent to this many
 *   parallel worker calls and the results are reconciled by a final call.
 * @property warningCallback - Optional callback invoked with a human-readable
 *   warning string on recoverable errors and incomplete responses.
 */
export interface LLMSubmitOptions {
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
 * Implements the "shotgun" strategy for {@link llmSubmit}: sends `numBarrels`
 * parallel requests with identical inputs, then asks the model to examine all
 * responses and reconcile them into a single authoritative answer.
 *
 * This improves output quality for tasks where the model benefits from
 * exploring multiple reasoning paths simultaneously.
 *
 * @param messages - The conversation history to send to each worker.
 * @param aiClient - The LLM provider's client instance.
 * @param options - Submit options (the `shotgun` field is stripped before
 *   forwarding to avoid infinite recursion).
 * @param numBarrels - Number of parallel worker requests to fire.
 * @returns The reconciled response from the model.
 */
const llmSubmitShotgun = async (
  messages: unknown[],
  aiClient: AIClientLike,
  options: LLMSubmitOptions,
  numBarrels: number
): Promise<
  string | Record<string, unknown> | unknown[] | number | boolean | null
> => {
  messages = JSON.parse(JSON.stringify(messages));
  options = JSON.parse(JSON.stringify(options));

  // Delete the shotgun option from the options passed to
  // llmSubmitShotgun to avoid infinite recursion.
  delete options.shotgun;

  if (numBarrels <= 1) {
    // No need for shotgun logic if only 1 barrel requested.
    return llmSubmit(messages, aiClient, options);
  }

  // Remove the shotgun option before passing to llmSubmit to avoid
  // infinite recursion!
  delete options.shotgun;

  const convoShotgun: unknown[][] = [];
  for (let i = 0; i < numBarrels; i += 1) {
    const convoBarrel = JSON.parse(JSON.stringify(messages));
    convoShotgun.push(convoBarrel);
  }
  const resultsRaw: unknown[] = await Promise.all(
    convoShotgun.map((convoBarrel) => llmSubmit(convoBarrel, aiClient, options))
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
  const sPonderReply = await llmSubmit(reconcileMessages, aiClient, {
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

  return llmSubmit(reconcileMessages, aiClient, options);
};

// Prompt to prohibit unicode characters in model output.
// Currently not used, but may be useful in the future.
const PROMPT_UNICODE_PROHIBITION = `
ABSOLUTELY NO UNICODE ALLOWED. 
Only use typeable keyboard characters. Do not try to circumvent this rule 
with escape sequences, backslashes, or other tricks. Use double dashes (--)
instead of em-dashes or en-dashes; use straight quotes (") and single quotes (')
 instead of curly versions, use hyphens instead of bullets, etc.
`.trim();

/**
 * Submits a conversation to the AI Responses API and returns the model's
 * reply.
 *
 * - Prepends a current-datetime system message to every request.
 * - Optionally enforces a JSON response format (plain JSON or a full JSON
 *   schema via `options.jsonResponse`).
 * - Automatically retries on JSON parse errors and retryable errors up to
 *       `options.retryLimit` times.
 * - Delegates to {@link llmSubmitShotgun} when `options.shotgun > 1`.
 *
 * @param messages - The conversation history, including any prior assistant
 *   turns. Each element should be a message object with at minimum `role` and
 *   `content` fields.
 * @param aiClient - An {@link AIClientLike} instance used to call the API.
 * @param options - Optional settings controlling model, JSON mode, retries,
 *   and shotgun parallelism.
 * @returns The model's response. A plain `string` when `jsonResponse` is
 *   falsy; otherwise a parsed JSON value (`Record`, `unknown[]`, `number`,
 *   `boolean`, or `null`).
 * @throws The last encountered error after all retry attempts are exhausted,
 *   or immediately for non-retryable errors.
 */
export async function llmSubmit(
  messages: unknown[],
  aiClient: AIClientLike,
  options: LLMSubmitOptions = {}
): Promise<
  string | Record<string, unknown> | unknown[] | number | boolean | null
> {
  if (options.shotgun && options.shotgun > 1) {
    return await llmSubmitShotgun(messages, aiClient, options, options.shotgun);
  }

  const llmProviderName = identifyLLMProvider(aiClient);

  const model = options.model || getModelName(llmProviderName, 'smart');
  const retryLimit = options.retryLimit ?? LLM_RETRY_LIMIT_DEFAULT;
  const retryBackoffTimeSeconds =
    options.retryBackoffTimeSeconds ?? LLM_RETRY_BACKOFF_TIME_SECONDS_DEFAULT;

  const systemAnnouncement = `${(options.systemAnnouncementMessage || '').trim()}`;

  let failedError: unknown = null;

  // Create a prepared messages payload.
  // We must retain our reference to the original `messages` array for appending the
  // response later. Therefore, we can't simply re-use the messages array.

  // Remove any previous datetime system messages from the conversation, since we'll be
  // prepending a fresh one with the current datetime.
  let preparedMessages = messages.filter((message: any) => {
    const role = message?.role;
    const content = message?.content;
    return !(role === 'system' && `${content}`.startsWith('!DATETIME:'));
  });

  // Insert a new datetime system message at the beginning of the conversation
  // to provide the model with current temporal context.
  preparedMessages.unshift(currentDatetimeSystemMessage());

  // If a system announcement message is provided, insert it at the start.
  if (systemAnnouncement) {
    preparedMessages.unshift({ role: 'system', content: systemAnnouncement });
  }

  for (let index = 0; index < retryLimit; index += 1) {
    let llmReply = '';

    try {
      if (llmProviderName === 'openai') {
        const openaiClient = aiClient as AIClientLike;
        const payloadBody: any = {
          model,
          input: preparedMessages,
        };
        if (options.jsonResponse) {
          if (typeof options.jsonResponse === 'boolean') {
            // Freeform JSON response requested, with no schema enforcement.
            payloadBody.text = { format: { type: 'json_object' } };
          } else {
            // JSON response requested with a provided JSON schema for format enforcement.
            // It has to be the full text object, with a `format` field and everything.
            payloadBody.text = options.jsonResponse;
          }
        }
        let llmResponse = await openaiClient.responses!.create(payloadBody);
        llmResponse = llmResponse.output_text.trim();

        if (options.warningCallback) {
          if (llmResponse.error) {
            options.warningCallback(
              `ERROR: OpenAI API returned an error: ${llmResponse.error}`
            );
          }
          if (llmResponse.incomplete_details) {
            options.warningCallback(
              `ERROR: OpenAI API returned incomplete details: ${llmResponse.incomplete_details}`
            );
          }
        }
      } else if (llmProviderName === 'anthropic') {
        const anthropicClient = aiClient as AIClientLike;
        // payloadBody.max_tokens = 8192;
        // llmResponse = await anthropicClient.messages!.create(payloadBody);
        throw new Error(
          'Anthropic client support not yet implemented in llmSubmit'
        );
      } else {
        throw new Error(`Unsupported LLM provider: ${llmProviderName}`);
      }

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
            `LLM Provider (${llmProviderName}) API error:\n\n${error}.\n\nRetrying (attempt ${index + 1} of ${retryLimit}) in ${retryBackoffTimeSeconds} seconds...`
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
