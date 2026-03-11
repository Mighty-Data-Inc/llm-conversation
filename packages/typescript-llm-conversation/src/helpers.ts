export const GPT_MODEL_CHEAP = 'gpt-4.1-nano';
export const GPT_MODEL_SMART = 'gpt-4.1';
export const GPT_MODEL_VISION = 'gpt-4.1';

const GPT_RETRY_LIMIT_DEFAULT = 5;
const GPT_RETRY_BACKOFF_TIME_SECONDS_DEFAULT = 30;

/**
 * Type guard that returns `true` when `value` is a plain, non-null, non-array
 * object — i.e. a `Record<string, unknown>`.
 *
 * @param value - The value to test.
 * @returns `true` if `value` is a plain object record.
 */
export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * OpenAI's API, even when called with a JSON schema, will often return text that is not
 * valid JSON. It's often because the model will add extra text after the end of its valid
 * JSON response. E.g. instead of `{"foo":"bar"}`, it will sometimes return
 * `{"foo":"bar"}{"baz":"quux"}`.
 *
 * We intentionally do not implement a custom JSON parser here. Instead:
 * 1) Try to parse the full text first (fast path, most responses).
 * 2) If that fails, scan prefixes from start to end and let JSON.parse decide validity.
 *
 * This keeps JSON semantics delegated to the platform parser while still recovering from
 * trailing junk in model output.
 *
 * @param input The text to parse.
 * @returns The first valid JSON value found at the start of the input text.
 * @throws {SyntaxError} If no valid JSON prefix exists.
 */
export function parseFirstJsonValue(input: string): any {
  const text = input.trimStart();
  if (!text) {
    throw new SyntaxError('Unexpected end of JSON input');
  }

  try {
    return JSON.parse(text);
  } catch {
    for (let end = 1; end <= text.length; end += 1) {
      try {
        return JSON.parse(text.slice(0, end));
      } catch {
        // Keep scanning until we find a valid JSON prefix.
      }
    }
  }

  throw new SyntaxError('Unexpected token in JSON input');
}

/**
 * Builds a message containing the current local date and time
 * formatted as `YYYY-MM-DD HH:MM:SS`. This message is automatically prepended
 * to every request so the model is aware of the current datetime.
 *
 * @returns A system message with the current timestamp.
 */
export function currentDatetimeSystemMessage(): {
  role: string;
  content: string;
} {
  const now = new Date();
  const pad = (value: number): string => value.toString().padStart(2, '0');
  const timestamp =
    `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ` +
    `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;

  return {
    role: 'system',
    content: `!DATETIME: The current date and time is ${timestamp}`,
  };
}
