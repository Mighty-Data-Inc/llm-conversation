export const JSON_INTEGER = Symbol('JSON_INTEGER');
export const JSON_NUMBER = Symbol('JSON_NUMBER');
export const JSON_STRING = String;
export const JSON_BOOLEAN = Boolean;

/**
 * The object shape returned by {@link JSONSchemaFormat}. It is passed directly
 * as the `text` parameter of an OpenAI Responses API call to enforce a
 * structured JSON schema response.
 */
export interface JSONSchemaFormatResult extends Record<string, unknown> {
  format: {
    type: 'json_schema';
    strict: true;
    name?: string;
    description?: string;
    schema: Record<string, unknown>;
  };
}

const TYPEMAP = new Map<unknown, string>([
  [JSON_STRING, 'string'],
  [JSON_INTEGER, 'integer'],
  [JSON_NUMBER, 'number'],
  [JSON_BOOLEAN, 'boolean'],
  [String, 'string'],
  [Boolean, 'boolean'],
  [BigInt, 'integer'],
  [Number, 'number'],
]);

/**
 * Type guard that returns `true` when `value` is a plain, non-null, non-array
 * object.
 *
 * @param value - The value to test.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Type guard that returns `true` when `value` is an array where every element
 * is a `string`.
 *
 * @param value - The value to test.
 */
function isStringArray(value: unknown): value is string[] {
  return (
    Array.isArray(value) && value.every((item) => typeof item === 'string')
  );
}

/**
 * Type guard that returns `true` when `value` is a two-element tuple
 * `[min, max]` where each element is either a `number` or `null`, and at
 * least one element is a `number`. Used to represent optional inclusive
 * numeric bounds in a schema definition.
 *
 * @param value - The value to test.
 */
function isNumericRangeArray(
  value: unknown
): value is [number | null, number | null] {
  if (!Array.isArray(value) || value.length !== 2) {
    return false;
  }
  const [min, max] = value;
  const minValid = min === null || typeof min === 'number';
  const maxValid = max === null || typeof max === 'number';

  return (
    minValid && maxValid && (typeof min === 'number' || typeof max === 'number')
  );
}

/**
 * Type guard that returns `true` when `value` is an array that should be
 * interpreted as a metadata tuple: a mixed array with at least two elements
 * that is not a plain string array, and contains at least one string
 * (description) or numeric-range tuple (bounds). This form is used in the
 * schema DSL to attach descriptions and constraints to a field inline.
 *
 * @param value - The value to test.
 */
function isTupleMetadataArray(value: unknown): value is unknown[] {
  if (!Array.isArray(value) || value.length < 2) {
    return false;
  }
  if (isStringArray(value)) {
    return false;
  }

  return value.some(
    (item) => typeof item === 'string' || isNumericRangeArray(item)
  );
}

/**
 * Attempts to map a schema DSL value to a JSON Schema primitive type string
 * (`"string"`, `"boolean"`, `"integer"`, or `"number"`).
 *
 * Resolution order:
 * 1. Look up the value in {@link TYPEMAP} (handles the exported type constants
 *    and the JS built-in constructors).
 * 2. Infer from the JavaScript `typeof` of the value itself (a literal default
 *    value signals the type of the field).
 *
 * @param schemaValue - A DSL value representing the type of a schema field.
 * @returns The JSON Schema type string, or `null` if the type cannot be
 *   determined.
 */
function inferPrimitiveType(schemaValue: unknown): string | null {
  const direct = TYPEMAP.get(schemaValue);
  if (direct) {
    return direct;
  }

  if (typeof schemaValue === 'string') {
    return 'string';
  }
  if (typeof schemaValue === 'boolean') {
    return 'boolean';
  }
  if (typeof schemaValue === 'bigint') {
    return 'integer';
  }
  if (typeof schemaValue === 'number') {
    return Number.isInteger(schemaValue) ? 'integer' : 'number';
  }

  return null;
}

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
 * - Primitive type constants (`JSON_STRING`, `JSON_INTEGER`, `JSON_NUMBER`,
 *   `JSON_BOOLEAN`, and their JS built-in equivalents) map directly to JSON
 *   Schema primitive types.
 *
 * @param subschema - A DSL node to convert.
 * @returns A JSON Schema-compatible object.
 * @throws {Error} If a schema value cannot be mapped to a known type.
 */
function convertSchemaRecursive(subschema: unknown): Record<string, unknown> {
  let subschemaDescription = '';
  let subschemaEnum: string[] = [];
  let subschemaNumrange: [number | null, number | null] = [null, null];
  let subschemaValue: unknown = subschema;

  if (isTupleMetadataArray(subschema)) {
    for (const item of subschema) {
      if (!item) {
        subschemaValue = item;
        continue;
      }

      if (typeof item === 'string') {
        subschemaDescription = item;
        continue;
      }

      if (isStringArray(item) && item.length >= 2) {
        subschemaEnum = item;
        continue;
      }

      if (isNumericRangeArray(item)) {
        subschemaNumrange = item;
        continue;
      }

      subschemaValue = item;
    }
  }

  if (
    (Array.isArray(subschemaValue) && isTupleMetadataArray(subschemaValue)) ||
    (Array.isArray(subschemaValue) && subschemaValue.length === 0)
  ) {
    if (subschemaEnum.length > 0) {
      subschemaValue = JSON_STRING;
    }

    const [nr0, nr1] = subschemaNumrange;
    if (nr0 !== null || nr1 !== null) {
      if (
        (typeof nr0 === 'number' && !Number.isInteger(nr0)) ||
        (typeof nr1 === 'number' && !Number.isInteger(nr1))
      ) {
        subschemaValue = JSON_NUMBER;
      } else {
        subschemaValue = JSON_INTEGER;
      }
    }
  }

  const result: Record<string, unknown> = {};

  if (isRecord(subschemaValue)) {
    result.type = 'object';
    if (subschemaDescription) {
      result.description = subschemaDescription;
    }
    result.additionalProperties = false;

    const keys = Object.keys(subschemaValue);
    result.required = keys;

    const properties: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(subschemaValue)) {
      if (typeof value === 'string') {
        properties[key] = { type: 'string', description: value };
      } else {
        properties[key] = convertSchemaRecursive(value);
      }
    }

    result.properties = properties;
  } else if (Array.isArray(subschemaValue)) {
    if (subschemaValue.length >= 2 && isStringArray(subschemaValue)) {
      result.type = 'string';
      subschemaEnum = subschemaValue;
    } else {
      result.type = 'array';
      if (subschemaDescription) {
        result.description = subschemaDescription;
      }
      if (subschemaNumrange[0] !== null) {
        result.minItems = subschemaNumrange[0];
      }
      if (subschemaNumrange[1] !== null) {
        result.maxItems = subschemaNumrange[1];
      }

      const arrayExemplar = subschemaValue[0];
      if (typeof arrayExemplar === 'string') {
        result.items = { type: 'string', description: arrayExemplar };
      } else {
        result.items = convertSchemaRecursive(arrayExemplar);
      }
    }
  } else {
    const primitiveType = inferPrimitiveType(subschemaValue);
    if (!primitiveType) {
      throw new Error(
        `Unrecognized type for schema value: ${String(subschemaValue)}`
      );
    }
    result.type = primitiveType;
    if (subschemaDescription) {
      result.description = subschemaDescription;
    }
  }

  if (subschemaEnum.length) {
    result.enum = subschemaEnum;
  }

  if (result.type === 'integer' || result.type === 'number') {
    if (subschemaNumrange[0] !== null) {
      result.minimum = subschemaNumrange[0];
    }
    if (subschemaNumrange[1] !== null) {
      result.maximum = subschemaNumrange[1];
    }
  }

  return result;
}

/**
 * Converts a schema DSL object into a {@link JSONSchemaFormatResult} suitable
 * for use as the `text` parameter of an OpenAI Responses API call.
 *
 * The DSL is a lightweight notation where plain JS objects, arrays, and type
 * constants describe the desired response shape without having to write raw
 * JSON Schema manually. See {@link convertSchemaRecursive} for supported
 * forms.
 *
 * If the converted schema is not already an `object` type, it is automatically
 * wrapped in an object with a single required property named `name` (or
 * `"schema"` when `name` is omitted).
 *
 * @param schema - The schema DSL value describing the desired response shape.
 * @param name - Optional name for the JSON schema (used as the `name` field
 *   in the OpenAI format and as the wrapper key for non-object schemas).
 * @param description - Optional human-readable description of the schema,
 *   forwarded to the model to guide its output.
 * @returns A {@link JSONSchemaFormatResult} ready to be passed to
 *   {@link gptSubmit} via `options.jsonResponse`.
 */
export function JSONSchemaFormat(
  schema: unknown,
  name?: string,
  description?: string
): JSONSchemaFormatResult {
  if (!name) {
    name = 'json_schema_for_structured_response';
  }

  const result: JSONSchemaFormatResult = {
    format: {
      type: 'json_schema',
      strict: true,
      name,
      schema: {
        type: 'object',
        properties: {},
        required: [],
        additionalProperties: false,
      },
    },
  };

  if (description) {
    result.format.description = description;
  }

  let converted = convertSchemaRecursive(schema);
  if (converted.type !== 'object') {
    converted = {
      type: 'object',
      required: [name],
      additionalProperties: false,
      properties: {
        [name]: converted,
      },
    };
  }

  result.format.schema = converted;
  return result;
}
