import { describe, expect, it } from 'vitest';

import { JSONSchemaFormat } from '../src/jsonSchemaFormat.js';

describe('JSONSchemaFormat', () => {
  describe('format envelope behavior', () => {
    it('uses default format scaffold and default name', () => {
      const result = JSONSchemaFormat({ value: String });

      expect(result).toMatchObject({
        format: {
          type: 'json_schema',
          strict: true,
          name: 'json_schema_for_structured_response',
        },
      });
    });

    it('forwards custom name and description', () => {
      const result = JSONSchemaFormat(
        { value: String },
        'custom_name',
        'Schema description'
      );

      expect(result).toMatchObject({
        format: {
          type: 'json_schema',
          strict: true,
          name: 'custom_name',
          description: 'Schema description',
        },
      });
    });
  });

  describe('primitive schema conversion', () => {
    it('maps constructor and literal primitives', () => {
      const result = JSONSchemaFormat({
        sCtor: String,
        nCtor: Number,
        bCtor: Boolean,
        nullLiteral: null,
      });

      expect(result).toMatchObject({
        format: {
          schema: {
            type: 'object',
            properties: {
              sCtor: { type: 'string' },
              nCtor: { type: 'number' },
              bCtor: { type: 'boolean' },
              nullLiteral: { type: 'null' },
            },
          },
        },
      });
    });

    it('treats plain strings as descriptions for string fields', () => {
      const result = JSONSchemaFormat({ title: 'Human-readable title' });

      expect(result).toMatchObject({
        format: {
          schema: {
            properties: {
              title: {
                type: 'string',
                description: 'Human-readable title',
              },
            },
          },
        },
      });
    });
  });

  describe('object schema conversion', () => {
    it('converts nested objects recursively and marks all keys as required', () => {
      const result = JSONSchemaFormat({
        user: {
          id: Number,
          profile: {
            display_name: String,
            active: Boolean,
          },
        },
      });

      expect(result).toEqual({
        format: {
          type: 'json_schema',
          strict: true,
          name: 'json_schema_for_structured_response',
          schema: {
            type: 'object',
            additionalProperties: false,
            required: ['user'],
            properties: {
              user: {
                type: 'object',
                additionalProperties: false,
                required: ['id', 'profile'],
                properties: {
                  id: { type: 'number' },
                  profile: {
                    type: 'object',
                    additionalProperties: false,
                    required: ['display_name', 'active'],
                    properties: {
                      display_name: { type: 'string' },
                      active: { type: 'boolean' },
                    },
                  },
                },
              },
            },
          },
        },
      });
    });
  });

  describe('array schema conversion', () => {
    describe('single-element arrays', () => {
      it('uses first element as array item schema', () => {
        const result = JSONSchemaFormat({ tags: [String] });

        expect(result).toMatchObject({
          format: {
            schema: {
              properties: {
                tags: {
                  type: 'array',
                  items: { type: 'string' },
                },
              },
            },
          },
        });
      });
    });

    describe('multi-string arrays as enums', () => {
      it('treats multi-string arrays as enums', () => {
        const result = JSONSchemaFormat({ mode: ['fast', 'safe', 'balanced'] });

        expect(result).toMatchObject({
          format: {
            schema: {
              properties: {
                mode: {
                  type: 'string',
                  enum: ['fast', 'safe', 'balanced'],
                },
              },
            },
          },
        });
      });

      it('disambiguates [primitiveTypeName, description] as tuple, not enum', () => {
        const result = JSONSchemaFormat({
          mode: ['string', 'Mode description'],
        });

        expect(result).toMatchObject({
          format: {
            schema: {
              properties: {
                mode: {
                  type: 'string',
                  description: 'Mode description',
                },
              },
            },
          },
        });
      });
    });

    describe('tuple arrays (type + description)', () => {
      it('requires second tuple element to be a description string', () => {
        expect(() =>
          JSONSchemaFormat({ bad: [Number, [0, 10]] as unknown[] })
        ).toThrow('Second element needs to be a description string');
      });

      it('allows null second element and omits description', () => {
        const result = JSONSchemaFormat({ value: [String, null] as unknown[] });

        expect(result).toMatchObject({
          format: {
            schema: {
              properties: {
                value: { type: 'string' },
              },
            },
          },
        });

        const props = (result.format as any).schema.properties;
        expect(props.value).not.toHaveProperty('description');
      });

      it('allows undefined second element and omits description', () => {
        const result = JSONSchemaFormat({
          value: [Number, undefined, [0, 10]] as unknown[],
        });

        expect(result).toMatchObject({
          format: {
            schema: {
              properties: {
                value: {
                  type: 'number',
                  minimum: 0,
                  maximum: 10,
                },
              },
            },
          },
        });

        const props = (result.format as any).schema.properties;
        expect(props.value).not.toHaveProperty('description');
      });

      it('allows empty-string second element and omits description', () => {
        const result = JSONSchemaFormat({ value: [Boolean, ''] });

        expect(result).toMatchObject({
          format: {
            schema: {
              properties: {
                value: { type: 'boolean' },
              },
            },
          },
        });

        const props = (result.format as any).schema.properties;
        expect(props.value).not.toHaveProperty('description');
      });

      it('rejects non-numeric tuples longer than two elements', () => {
        expect(() =>
          JSONSchemaFormat({ bad: [String, 'Label', [0, 1]] as unknown[] })
        ).toThrow('Non-numeric types should only have 2 elements');
      });
    });

    describe('numeric tuple ranges', () => {
      it('applies numeric bounds from [min, max] tuple', () => {
        const result = JSONSchemaFormat({
          confidence: [Number, 'Confidence score', [0.0, 1.0]],
        });

        expect(result).toMatchObject({
          format: {
            schema: {
              properties: {
                confidence: {
                  type: 'number',
                  description: 'Confidence score',
                  minimum: 0.0,
                  maximum: 1.0,
                },
              },
            },
          },
        });
      });

      it('supports one-sided bounds with null/undefined', () => {
        const result = JSONSchemaFormat({
          min_only: [Number, 'Minimum only', [0, null]],
          max_only: [Number, 'Maximum only', [undefined, 10]],
          no_bounds: [Number, 'No bounds', [undefined, null]],
        });

        expect(result).toMatchObject({
          format: {
            schema: {
              properties: {
                min_only: {
                  type: 'number',
                  description: 'Minimum only',
                  minimum: 0,
                },
                max_only: {
                  type: 'number',
                  description: 'Maximum only',
                  maximum: 10,
                },
                no_bounds: {
                  type: 'number',
                  description: 'No bounds',
                },
              },
            },
          },
        });

        const props = (result.format as any).schema.properties;
        expect(props.no_bounds).not.toHaveProperty('minimum');
        expect(props.no_bounds).not.toHaveProperty('maximum');
      });

      it('rejects malformed numeric third element shape', () => {
        expect(() =>
          JSONSchemaFormat({ bad: [Number, 'Score', 5] as unknown[] })
        ).toThrow(
          'Numeric types with a third element need to have that element be a length-2 array representing [min, max].'
        );
      });

      it('rejects malformed numeric third element value types', () => {
        expect(() =>
          JSONSchemaFormat({
            bad: [Number, 'Score', [0, 'x']] as unknown[],
          })
        ).toThrow(
          'Numeric types with a third element need to have that element be a length-2 array of numbers (or null/undefined).'
        );
      });
    });
  });

  describe('invalid input handling', () => {
    it('throws for empty arrays', () => {
      expect(() => JSONSchemaFormat({ bad: [] })).toThrow(
        'Invalid schema array'
      );
    });

    it('throws for unsupported schema values', () => {
      expect(() =>
        JSONSchemaFormat({ bad: Symbol('x') as unknown as string })
      ).toThrow('Invalid schema value');
    });
  });

  describe('regression and branch guards', () => {
    it('currently allows array roots at runtime when cast, returning array schema', () => {
      const result = JSONSchemaFormat(['string'] as unknown as Record<
        string,
        unknown
      >);

      expect(result).toMatchObject({
        format: {
          schema: {
            type: 'array',
            items: { type: 'string' },
          },
        },
      });
    });

    it('treats integer alias as numeric type in current JS implementation', () => {
      const result = JSONSchemaFormat({ count: ['integer', 'Count', [1, 3]] });

      expect(result).toMatchObject({
        format: {
          schema: {
            properties: {
              count: {
                type: 'number',
                description: 'Count',
                minimum: 1,
                maximum: 3,
              },
            },
          },
        },
      });
    });
  });

  describe('string representations of primitive types', () => {
    it('maps primitive string tokens to JSON schema primitive types', () => {
      const result = JSONSchemaFormat({
        s: 'string',
        n: 'number',
        i: 'integer',
        f: 'float',
        b: 'boolean',
        z: 'null',
      });

      expect(result).toMatchObject({
        format: {
          schema: {
            properties: {
              s: { type: 'string' },
              n: { type: 'number' },
              i: { type: 'number' },
              f: { type: 'number' },
              b: { type: 'boolean' },
              z: { type: 'null' },
            },
          },
        },
      });
    });

    it('disambiguates ["string", description] as tuple, not enum', () => {
      const result = JSONSchemaFormat({
        mode: ['string', 'Mode description'],
      });

      expect(result).toMatchObject({
        format: {
          schema: {
            properties: {
              mode: {
                type: 'string',
                description: 'Mode description',
              },
            },
          },
        },
      });
    });
  });
});
