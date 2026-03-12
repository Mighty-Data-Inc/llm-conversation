import unittest

from mightydatainc_llm_conversation.json_schema_format import (
    JSON_BOOLEAN,
    JSON_INTEGER,
    JSON_NUMBER,
    JSON_STRING,
    JSONSchemaFormat,
)


class TestJSONSchemaFormat(unittest.TestCase):
    def test_expands_object_schema_with_primitive_fields(self):
        result = JSONSchemaFormat(
            {
                "title": "Human-readable title",
                "age": JSON_INTEGER,
                "score": JSON_NUMBER,
                "enabled": JSON_BOOLEAN,
            },
            "response",
            "Structured response payload",
        )

        self.assertEqual(
            result,
            {
                "format": {
                    "type": "json_schema",
                    "strict": True,
                    "name": "response",
                    "description": "Structured response payload",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["title", "age", "score", "enabled"],
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Human-readable title",
                            },
                            "age": {"type": "integer"},
                            "score": {"type": "number"},
                            "enabled": {"type": "boolean"},
                        },
                    },
                }
            },
        )

    def test_wraps_non_object_root_schema_with_provided_name(self):
        result = JSONSchemaFormat(JSON_STRING, "answer")

        self.assertEqual(
            result,
            {
                "format": {
                    "type": "json_schema",
                    "strict": True,
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["answer"],
                        "properties": {
                            "answer": {"type": "string"},
                        },
                    },
                }
            },
        )

    def test_supports_enum_shorthand_from_string_list(self):
        result = JSONSchemaFormat(
            {
                "mode": ["fast", "safe", "balanced"],
            },
            "answer_enum",
        )

        self.assertEqual(
            result,
            {
                "format": {
                    "type": "json_schema",
                    "strict": True,
                    "name": "answer_enum",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["mode"],
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": ["fast", "safe", "balanced"],
                            },
                        },
                    },
                }
            },
        )

    def test_supports_metadata_tuple_style_for_array_bounds_and_item_description(self):
        result = JSONSchemaFormat(
            {
                "tags": ["Tag collection", [1, 5], ["Single tag"]],
            },
            "test_schema",
        )

        self.assertEqual(
            result,
            {
                "format": {
                    "type": "json_schema",
                    "strict": True,
                    "name": "test_schema",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["tags"],
                        "properties": {
                            "tags": {
                                "type": "array",
                                "description": "Tag collection",
                                "minItems": 1,
                                "maxItems": 5,
                                "items": {
                                    "type": "string",
                                    "description": "Single tag",
                                },
                            },
                        },
                    },
                }
            },
        )

    def test_infers_integer_and_enum_via_tuple_metadata(self):
        result = JSONSchemaFormat(
            {
                "age": ["Age in years", [0, 120], []],
                "color": ["Preferred color", ["red", "green", "blue"], []],
            },
            "test_schema",
        )

        self.assertEqual(
            result,
            {
                "format": {
                    "type": "json_schema",
                    "strict": True,
                    "name": "test_schema",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["age", "color"],
                        "properties": {
                            "age": {
                                "type": "integer",
                                "description": "Age in years",
                                "minimum": 0,
                                "maximum": 120,
                            },
                            "color": {
                                "type": "string",
                                "description": "Preferred color",
                                "enum": ["red", "green", "blue"],
                            },
                        },
                    },
                }
            },
        )

    def test_supports_number_type_with_range_metadata_when_explicitly_marked(self):
        result = JSONSchemaFormat(
            {
                "confidence": ["Confidence score", [0.0, 1.0], JSON_NUMBER],
            },
            "test_schema",
        )

        self.assertEqual(
            result,
            {
                "format": {
                    "type": "json_schema",
                    "strict": True,
                    "name": "test_schema",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["confidence"],
                        "properties": {
                            "confidence": {
                                "type": "number",
                                "description": "Confidence score",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                    },
                }
            },
        )

    def test_supports_one_sided_numeric_bounds(self):
        result = JSONSchemaFormat(
            {
                "min_only": ["Minimum only", [0, None], []],
                "max_only": ["Maximum only", [None, 10], []],
            },
            "test_schema",
        )

        self.assertEqual(
            result,
            {
                "format": {
                    "type": "json_schema",
                    "strict": True,
                    "name": "test_schema",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["min_only", "max_only"],
                        "properties": {
                            "min_only": {
                                "type": "integer",
                                "description": "Minimum only",
                                "minimum": 0,
                            },
                            "max_only": {
                                "type": "integer",
                                "description": "Maximum only",
                                "maximum": 10,
                            },
                        },
                    },
                }
            },
        )

    def test_supports_nested_recursive_schemas(self):
        result = JSONSchemaFormat(
            {
                "groups": [
                    {
                        "name": "Group name",
                        "members": [
                            {
                                "id": JSON_INTEGER,
                                "roles": ["admin", "viewer"],
                                "tags": ["Tag label"],
                                "profile": {
                                    "active": JSON_BOOLEAN,
                                    "scores": [JSON_NUMBER],
                                },
                            }
                        ],
                    }
                ],
            },
            "nested_schema",
        )

        self.assertEqual(
            result,
            {
                "format": {
                    "type": "json_schema",
                    "strict": True,
                    "name": "nested_schema",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["groups"],
                        "properties": {
                            "groups": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["name", "members"],
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Group name",
                                        },
                                        "members": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "additionalProperties": False,
                                                "required": [
                                                    "id",
                                                    "roles",
                                                    "tags",
                                                    "profile",
                                                ],
                                                "properties": {
                                                    "id": {"type": "integer"},
                                                    "roles": {
                                                        "type": "string",
                                                        "enum": ["admin", "viewer"],
                                                    },
                                                    "tags": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "string",
                                                            "description": "Tag label",
                                                        },
                                                    },
                                                    "profile": {
                                                        "type": "object",
                                                        "additionalProperties": False,
                                                        "required": [
                                                            "active",
                                                            "scores",
                                                        ],
                                                        "properties": {
                                                            "active": {
                                                                "type": "boolean"
                                                            },
                                                            "scores": {
                                                                "type": "array",
                                                                "items": {
                                                                    "type": "number"
                                                                },
                                                            },
                                                        },
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                }
            },
        )

    def test_supports_nested_recursive_schemas_with_inner_tuple_metadata(self):
        result = JSONSchemaFormat(
            {
                "groups": [
                    {
                        "name": "Group name",
                        "members": [
                            "Members list",
                            [1, None],
                            [
                                {
                                    "id": JSON_INTEGER,
                                    "score": ["Member score", [0.0, 1.0], JSON_NUMBER],
                                    "aliases": [
                                        "Alias list",
                                        [0, 3],
                                        ["Alias text"],
                                    ],
                                    "history": [
                                        {
                                            "year": [
                                                "Year",
                                                [1900, 2100],
                                                JSON_INTEGER,
                                            ],
                                            "tags": [
                                                "History tags",
                                                [0, 5],
                                                ["Tag text"],
                                            ],
                                        }
                                    ],
                                }
                            ],
                        ],
                    }
                ],
            },
            "nested_schema_with_metadata",
        )

        self.assertEqual(
            result,
            {
                "format": {
                    "type": "json_schema",
                    "strict": True,
                    "name": "nested_schema_with_metadata",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["groups"],
                        "properties": {
                            "groups": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["name", "members"],
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Group name",
                                        },
                                        "members": {
                                            "type": "array",
                                            "description": "Members list",
                                            "minItems": 1,
                                            "items": {
                                                "type": "object",
                                                "additionalProperties": False,
                                                "required": [
                                                    "id",
                                                    "score",
                                                    "aliases",
                                                    "history",
                                                ],
                                                "properties": {
                                                    "id": {"type": "integer"},
                                                    "score": {
                                                        "type": "number",
                                                        "description": "Member score",
                                                        "minimum": 0.0,
                                                        "maximum": 1.0,
                                                    },
                                                    "aliases": {
                                                        "type": "array",
                                                        "description": "Alias list",
                                                        "minItems": 0,
                                                        "maxItems": 3,
                                                        "items": {
                                                            "type": "string",
                                                            "description": "Alias text",
                                                        },
                                                    },
                                                    "history": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "additionalProperties": False,
                                                            "required": [
                                                                "year",
                                                                "tags",
                                                            ],
                                                            "properties": {
                                                                "year": {
                                                                    "type": "integer",
                                                                    "description": "Year",
                                                                    "minimum": 1900,
                                                                    "maximum": 2100,
                                                                },
                                                                "tags": {
                                                                    "type": "array",
                                                                    "description": "History tags",
                                                                    "minItems": 0,
                                                                    "maxItems": 5,
                                                                    "items": {
                                                                        "type": "string",
                                                                        "description": "Tag text",
                                                                    },
                                                                },
                                                            },
                                                        },
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                }
            },
        )

    def test_uses_default_name_when_name_not_provided_for_object_schema(self):
        result = JSONSchemaFormat({"value": JSON_STRING})

        self.assertEqual(
            result["format"]["name"],
            "json_schema_for_structured_response",
        )
        self.assertEqual(result["format"]["type"], "json_schema")
        self.assertTrue(result["format"]["strict"])

    def test_uses_default_name_as_wrapper_key_when_name_not_provided_for_non_object_schema(
        self,
    ):
        result = JSONSchemaFormat(JSON_INTEGER)

        self.assertEqual(
            result,
            {
                "format": {
                    "type": "json_schema",
                    "strict": True,
                    "name": "json_schema_for_structured_response",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["json_schema_for_structured_response"],
                        "properties": {
                            "json_schema_for_structured_response": {"type": "integer"},
                        },
                    },
                }
            },
        )

    def test_throws_for_unsupported_schema_values(self):
        with self.assertRaisesRegex(Exception, "Unrecognized type for schema value"):
            JSONSchemaFormat({"bad": object()}, "test_schema")


if __name__ == "__main__":
    unittest.main()
