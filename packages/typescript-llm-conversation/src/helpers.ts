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
    // Continue to the prefix-scanning logic below.
  }

  for (let start = 0; start < text.length; start += 1) {
    // We're not only just looking for valid JSON, but specifically
    // for a valid JSON dict. We'll even allow an array, just to be
    // generous. But this means that the first character must
    // be either { or [.
    const firstChar = text[start];
    if (firstChar !== '{' && firstChar !== '[') {
      continue;
    }

    for (let end = start + 1; end <= text.length; end += 1) {
      try {
        return JSON.parse(text.slice(start, end));
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
