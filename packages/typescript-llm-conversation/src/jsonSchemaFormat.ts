/**
 * Recursively converts a schema DSL node into a standard JSON Schema object.
 *
 * The DSL supports several shorthand notations:
 * - A plain object `{}` becomes a JSON Schema `object` with all keys required.
 * - An array with one element becomes a JSON Schema `array` whose `items` is
 *   derived from that element.
 * - A string array with two or more elements becomes an `enum`.
 * - A metadata tuple (detected by {@link isTupleMetadataArray}) can bundle a
 *   description string, an enum list, and/or a numeric range alongside the
 *   actual type value.
 * - Primitive types (or string representations of primitive types) map directly to
 *   JSON Schema primitive types.
 *
 * @param subschema - A DSL node to convert.
 * @returns A JSON Schema-compatible object.
 * @throws {Error} If a schema value cannot be mapped to a known type.
 */
function convertSchemaRecursive(subschema: unknown): Record<string, unknown> {
  // First we handle the easy cases: primitive types and their string aliases.
  if (subschema === 'integer') {
    // JavaScript doesn't distinguish between integers and floats,
    // so this is equivalent to 'number' in JSON Schema. Versions in other languages
    // that support this differentiation will detect Integer vs Float directly,
    // and map them to 'integer' and 'number' respectively in the output JSON Schema.
    return { type: 'number' };
  }
  if (subschema === 'number' || subschema === 'float' || subschema === Number) {
    return { type: 'number' };
  }
  if (subschema === 'string' || subschema === String) {
    return { type: 'string' };
  }
  if (subschema === 'boolean' || subschema === Boolean) {
    return { type: 'boolean' };
  }
  if (subschema === 'null' || subschema === null) {
    return { type: 'null' };
  }

  // If it's an *instance* of a string, and hasn't already been caught by the above,
  // then assume that the string contains a description.
  if (typeof subschema === 'string') {
    return { type: 'string', description: `${subschema}` };
  }

  // We've already handled the case where it's null.
  // Now let's make sure it's an object. It can't really be anything else.
  // In JavaScript arrays are technically objects too, so the only cases we
  // haven't handled yet are where subschema={...} or subschema=[...].
  // Either way, it has to pass the object test. It's not clear to me how
  // it could possibly fail this test at this point, but never underestimate
  // the creativity of bad inputs!
  if (typeof subschema !== 'object') {
    throw new Error(`Invalid schema value: ${JSON.stringify(subschema)}`);
  }

  // If subschema={...}, then we create a schema object for it,
  // with each of its properties being recursively converted.
  // All keys are required by default.
  if (!Array.isArray(subschema)) {
    const retval = {
      type: 'object',
      additionalProperties: false,
      required: [...Object.keys(subschema)],
      properties: {} as Record<string, unknown>,
    };
    for (const [key, value] of Object.entries(subschema)) {
      retval.properties[key] = convertSchemaRecursive(value);
    }
    return retval;
  }

  // If we're here, then subschema=[...].
  // This begins a set of interesting cases.

  // It has to have at least one element, otherwise we don't know what type of array to make.
  if (subschema.length === 0) {
    throw new Error(`Invalid schema array: ${JSON.stringify(subschema)}`);
  }

  // If it's a single-element array, then it's an array schema whose items
  // are the type of that element.
  if (subschema.length === 1) {
    return {
      type: 'array',
      items: convertSchemaRecursive(subschema[0]),
    };
  }

  // All of the cases that remain are multi-element arrays.

  const firstElem = subschema[0];
  const secondElem = subschema[1];

  // If it's an array whose members are all strings, then we treat it as an enum.
  // NOTE: This could create a subtle ambiguous case, in the situation in which an array
  // is of length 2, the first element is a string intending to designate a primitive type,
  // and the second element is a string intending to designate a description.
  // To guard against this case, we check to make sure that the first string isn't
  // a primitive type name. If it is, then we move on to other cases.
  // We'll very carefully build a check for enums.
  const areAllStrings = subschema.every((elem) => typeof elem === 'string');
  let isEnum = areAllStrings;
  if (isEnum && subschema.length == 2) {
    // We suspect it might be an enum, but if it's all strings and exactly length 2
    // and the first string is a primitive type, then it's more likely that this is a case of
    // a type + description tuple. So we check for that and if we find it, we reject the
    // enum hypothesis.
    if (
      ['string', 'number', 'integer', 'float', 'boolean', 'null'].includes(
        firstElem
      )
    ) {
      isEnum = false;
    }
  }
  if (isEnum) {
    return {
      type: 'string',
      enum: subschema,
    };
  }

  // It's *not* an enum. Then it must be a tuple that includes a description
  // and possibly min/max values.
  // It *must* have exactly 2 elements, unless it's a numeric type, in which case it
  // can have 3 elements if the last one is a min/max specifier.

  const retval = convertSchemaRecursive(firstElem);

  // In either case, the second element *must* be a string.
  // (It can be null or undefined.)
  if (secondElem !== null && secondElem !== undefined) {
    if (typeof secondElem !== 'string') {
      throw new Error(
        `Invalid schema tuple: ${JSON.stringify(subschema)}. ` +
          `Second element needs to be a description string.`
      );
    }

    // We've now determined that the second element is a string.
    const description = secondElem.trim();
    if (description.length > 0) {
      retval.description = description;
    }
  }

  // If retval.type is 'number', 'integer', or 'float', then we check for the presence
  // of a third element in the tuple, which we interpret as a [min, max] range specifier.
  // If it's there, it needs to conform to a certain structure. Otherwise, we throw an error.
  // Either way, we do this piecemeal because the logic is delicate.
  const isTypeNumeric =
    retval.type === 'number' ||
    retval.type === 'integer' ||
    retval.type === 'float';

  if (!isTypeNumeric) {
    if (subschema.length > 2) {
      throw new Error(
        `Invalid schema tuple: ${JSON.stringify(subschema)}. ` +
          `Non-numeric types should only have 2 elements (type + description).`
      );
    }
    // It's not numeric, and it's of valid length, so we're done.
    return retval;
  }

  // It *is* numeric.
  // If it's length 2, then it's just a type + description tuple, and we're done.
  if (subschema.length === 2) {
    return retval;
  }

  // It is numeric, and it has a third element.
  const thirdElem = subschema[2];

  // If the third element is null or undefined, then we treat it as if it weren't there at all, and we're done.
  if (thirdElem === null || thirdElem === undefined) {
    return retval;
  }

  // Make sure that the third element is a length-2 array of numbers, representing [min, max].
  // Either value can be null (or undefined).
  if (!Array.isArray(thirdElem) || thirdElem.length !== 2) {
    throw new Error(
      `Invalid schema tuple: ${JSON.stringify(subschema)}. ` +
        `Numeric types with a third element need to have that element be a length-2 array representing [min, max].`
    );
  }
  // Make sure that every element of the third element is either a number, null, or undefined.
  const areMinMaxValuesValid = thirdElem.every(
    (elem) => typeof elem === 'number' || elem === null || elem === undefined
  );
  if (!areMinMaxValuesValid) {
    throw new Error(
      `Invalid schema tuple: ${JSON.stringify(subschema)}. ` +
        `Numeric types with a third element need to have that element be a length-2 array of numbers (or null/undefined).`
    );
  }

  // If we've reached this point, then the third element is valid.
  const [minValue, maxValue] = thirdElem;
  if (minValue !== null && minValue !== undefined) {
    retval.minValue = minValue;
  }
  if (maxValue !== null && maxValue !== undefined) {
    retval.maxValue = maxValue;
  }

  return retval;
}

/**
 * Converts a schema DSL object into a {@link JSONSchemaFormatResult} suitable
 * for use as the `text` parameter of an OpenAI Responses API call, or as the
 * output_config of an Anthropic Messages call (after sanitization), etc.
 *
 * The DSL is a lightweight notation where plain JS objects, arrays, and type
 * constants describe the desired response shape without having to write raw
 * JSON Schema manually. See {@link convertSchemaRecursive} for supported
 * forms.
 *
 * @param schema - The schema DSL value describing the desired response shape.
 *   This *must* be a non-array object at the top level, but can contain nested
 *   arrays, objects, and primitive values in the schema definition.
 * @param name - Optional name for the JSON schema (used as the `name` field
 *   in the OpenAI format and as the wrapper key for non-object schemas).
 * @param description - Optional human-readable description of the schema,
 *   forwarded to the model to guide its output.
 * @returns A {@link JSONSchemaFormatResult} ready to be passed to
 *   {@link gptSubmit} via `options.jsonResponse`.
 */
export function JSONSchemaFormat(
  schema: Record<string, unknown>,
  name?: string,
  description?: string
): Record<string, unknown> {
  if (!name) {
    name = 'json_schema_for_structured_response';
  }

  const result: Record<string, any> = {
    format: {
      type: 'json_schema',
      strict: true,
      name,
      schema: {},
    },
  };

  if (description) {
    result.format.description = description;
  }

  const converted = convertSchemaRecursive(schema);

  result.format.schema = converted;
  return result;
}
