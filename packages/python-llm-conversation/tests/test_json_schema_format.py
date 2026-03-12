import unittest
from importlib import import_module


JSONSchemaFormat = getattr(
    import_module("mightydatainc_llm_conversation"),
    "JSONSchemaFormat",
)


class TestJSONSchemaFormat(unittest.TestCase):
    # format envelope behavior
    def test_uses_default_format_scaffold_and_default_name(self):
        result = JSONSchemaFormat({"value": str})

        self.assertIn("format", result)
        self.assertIsInstance(result["format"], dict)
        self.assertEqual(result["format"].get("type"), "json_schema")
        self.assertEqual(result["format"].get("strict"), True)
        self.assertEqual(
            result["format"].get("name"), "json_schema_for_structured_response"
        )
        self.assertIn("schema", result["format"])
        self.assertIsInstance(result["format"]["schema"], dict)

        self.assertIn("schema", result["format"])
        self.assertIsInstance(result["format"]["schema"], dict)

        resultSchema = result["format"]["schema"]
        self.assertEqual(resultSchema.get("type"), "object")
        self.assertEqual(resultSchema.get("additionalProperties"), False)
        self.assertIn("properties", resultSchema)
        self.assertIsInstance(resultSchema["properties"], dict)
        self.assertIn("value", resultSchema["properties"])
        self.assertEqual(resultSchema["properties"]["value"], {"type": "string"})
        self.assertIn("required", resultSchema)
        self.assertEqual(resultSchema["required"], ["value"])

    def test_forwards_custom_name_and_description(self):
        result = JSONSchemaFormat({"value": str}, "custom_name", "Schema description")

        self.assertEqual(result["format"].get("name"), "custom_name")
        self.assertEqual(result["format"].get("description"), "Schema description")

    # primitive schema conversion
    def test_maps_constructor_and_literal_primitives(self):
        result = JSONSchemaFormat(
            {
                "sCtor": str,
                "nCtor": float,
                "bCtor": bool,
                "nullLiteral": None,
            }
        )

        self.assertIn("format", result)
        self.assertIsInstance(result["format"], dict)
        self.assertIn("schema", result["format"])
        self.assertIsInstance(result["format"]["schema"], dict)

        resultSchema = result["format"]["schema"]

        self.assertEqual(resultSchema["type"], "object")
        self.assertEqual(resultSchema["additionalProperties"], False)
        self.assertListEqual(
            resultSchema["required"],
            ["sCtor", "nCtor", "bCtor", "nullLiteral"],
        )

        self.assertDictEqual(
            resultSchema["properties"],
            {
                "sCtor": {"type": "string"},
                "nCtor": {"type": "number"},
                "bCtor": {"type": "boolean"},
                "nullLiteral": {"type": "null"},
            },
        )

    def test_treats_plain_strings_as_descriptions_for_string_fields(self):
        result = JSONSchemaFormat({"title": "Human-readable title"})

        self.assertDictEqual(
            result["format"]["schema"]["properties"]["title"],
            {
                "type": "string",
                "description": "Human-readable title",
            },
        )

    # object schema conversion
    def test_converts_nested_objects_recursively_and_marks_all_keys_required(
        self,
    ):
        result = JSONSchemaFormat(
            {
                "user": {
                    "id": float,
                    "profile": {
                        "display_name": str,
                        "active": bool,
                    },
                }
            }
        )

        resultSchema = result["format"]["schema"]

        self.assertListEqual(resultSchema["required"], ["user"])
        self.assertEqual(resultSchema["additionalProperties"], False)

        # Assert that "properties" has the "user" property, and nothing else.
        self.assertIn("user", resultSchema["properties"])
        self.assertEqual(len(resultSchema["properties"]), 1)

        self.assertListEqual(
            resultSchema["properties"]["user"]["required"],
            ["id", "profile"],
        )
        self.assertEqual(
            resultSchema["properties"]["user"]["additionalProperties"],
            False,
        )
        self.assertDictEqual(
            resultSchema["properties"]["user"]["properties"]["id"],
            {"type": "number"},
        )
        self.assertListEqual(
            resultSchema["properties"]["user"]["properties"]["profile"]["required"],
            ["display_name", "active"],
        )
        self.assertEqual(
            resultSchema["properties"]["user"]["properties"]["profile"][
                "additionalProperties"
            ],
            False,
        )
        self.assertEqual(
            resultSchema["properties"]["user"]["properties"]["profile"]["properties"],
            {
                "display_name": {"type": "string"},
                "active": {"type": "boolean"},
            },
        )

    # array schema conversion - single-element arrays
    def test_uses_first_element_as_array_item_schema(self):
        result = JSONSchemaFormat({"tags": [str]})
        resultSchema = result["format"]["schema"]

        self.assertEqual(resultSchema["properties"]["tags"]["type"], "array")
        self.assertEqual(
            resultSchema["properties"]["tags"]["items"], {"type": "string"}
        )

    # array schema conversion - multi-string arrays as enums
    def test_treats_multi_string_arrays_as_enums(self):
        result = JSONSchemaFormat({"mode": ["fast", "safe", "balanced"]})
        resultSchema = result["format"]["schema"]

        self.assertEqual(resultSchema["properties"]["mode"]["type"], "string")
        self.assertEqual(
            resultSchema["properties"]["mode"]["enum"], ["fast", "safe", "balanced"]
        )

    def test_disambiguates_string_description_tuple_not_enum(self):
        result = JSONSchemaFormat({"mode": ["string", "Mode description"]})
        resultSchema = result["format"]["schema"]

        self.assertEqual(resultSchema["properties"]["mode"]["type"], "string")
        self.assertEqual(
            resultSchema["properties"]["mode"]["description"], "Mode description"
        )

    # array schema conversion - tuple arrays (type + description)
    def test_requires_second_tuple_element_to_be_description_string(self):
        with self.assertRaises(Exception):
            JSONSchemaFormat({"bad": [float, [0, 10]]})

    def test_allows_null_second_element_and_omits_description(self):
        result = JSONSchemaFormat({"value": [str, None]})
        resultSchema = result["format"]["schema"]

        self.assertEqual(resultSchema["properties"]["value"]["type"], "string")
        self.assertNotIn("description", resultSchema["properties"]["value"])

    def test_allows_min_and_max_values_with_omitted_description(self):
        result = JSONSchemaFormat({"value": [float, None, [0, 10]]})
        resultSchema = result["format"]["schema"]

        self.assertEqual(resultSchema["properties"]["value"]["type"], "string")
        self.assertNotIn("description", resultSchema["properties"]["value"])
        self.assertEqual(resultSchema["properties"]["value"]["minValue"], "string")
        self.assertEqual(resultSchema["properties"]["value"]["type"], "string")

    def test_allows_empty_string_second_element_and_omits_description(self):
        result = JSONSchemaFormat({"value": [bool, ""]})
        resultSchema = result["format"]["schema"]

        self.assertEqual(resultSchema["properties"]["value"]["type"], "boolean")
        self.assertNotIn("description", resultSchema["properties"]["value"])

    def test_rejects_non_numeric_tuples_longer_than_two_elements(self):
        with self.assertRaises(Exception):
            JSONSchemaFormat({"bad": [str, "Label", [0, 1]]})

    # array schema conversion - numeric tuple ranges
    def test_applies_numeric_bounds_from_min_max_tuple(self):
        result = JSONSchemaFormat(
            {
                "confidence": [
                    float,
                    "Confidence score",
                    [0.0, 1.0],
                ]
            }
        )
        resultSchema = result["format"]["schema"]
        self.assertEqual(resultSchema["properties"]["confidence"]["type"], "number")
        self.assertEqual(
            resultSchema["properties"]["confidence"]["description"], "Confidence score"
        )
        self.assertEqual(resultSchema["properties"]["confidence"]["minimum"], 0.0)
        self.assertEqual(resultSchema["properties"]["confidence"]["maximum"], 1.0)

    def test_supports_one_sided_bounds_with_null(self):
        result = JSONSchemaFormat(
            {
                "min_only": [float, "Minimum only", [0, None]],
                "max_only": [float, "Maximum only", [None, 10]],
                "no_bounds": [float, "No bounds", [None, None]],
                "no_bounds_null": [float, "No bounds null", None],
            }
        )
        resultSchemaProps = result["format"]["schema"]["properties"]

        self.assertEqual(resultSchemaProps["min_only"]["type"], "number")
        self.assertEqual(resultSchemaProps["min_only"]["description"], "Minimum only")
        self.assertEqual(resultSchemaProps["min_only"]["minimum"], 0)
        self.assertNotIn("maximum", resultSchemaProps["min_only"])

        self.assertEqual(resultSchemaProps["max_only"]["type"], "number")
        self.assertEqual(resultSchemaProps["max_only"]["description"], "Maximum only")
        self.assertEqual(resultSchemaProps["max_only"]["maximum"], 10)
        self.assertNotIn("minimum", resultSchemaProps["max_only"])

        self.assertEqual(resultSchemaProps["no_bounds"]["type"], "number")
        self.assertEqual(resultSchemaProps["no_bounds"]["description"], "No bounds")
        self.assertNotIn("minimum", resultSchemaProps["no_bounds"])
        self.assertNotIn("maximum", resultSchemaProps["no_bounds"])

        self.assertEqual(resultSchemaProps["no_bounds_null"]["type"], "number")
        self.assertEqual(
            resultSchemaProps["no_bounds_null"]["description"], "No bounds null"
        )
        self.assertNotIn("minimum", resultSchemaProps["no_bounds_null"])
        self.assertNotIn("maximum", resultSchemaProps["no_bounds_null"])

    def test_rejects_malformed_numeric_third_element_shape(self):
        with self.assertRaises(Exception):
            JSONSchemaFormat({"bad": [float, "Score", 5]})

    def test_rejects_malformed_numeric_third_element_value_types(self):
        with self.assertRaises(Exception):
            JSONSchemaFormat({"bad": [float, "Score", [0, "x"]]})

    # invalid input handling
    def test_throws_for_empty_arrays(self):
        with self.assertRaises(Exception):
            JSONSchemaFormat({"bad": []})

    def test_throws_for_unsupported_schema_values(self):
        with self.assertRaises(Exception):
            JSONSchemaFormat({"bad": object()})

    # regression and branch guards
    def test_currently_allows_array_roots_returning_array_schema(self):
        result = JSONSchemaFormat(["string"])
        resultSchema = result["format"]["schema"]
        self.assertEqual(resultSchema.get("type"), "array")
        self.assertEqual(resultSchema.get("items"), {"type": "string"})

    def test_treats_integer_alias_as_numeric_type(self):
        result = JSONSchemaFormat({"count": ["integer", "Count", [1, 3]]})
        resultSchemaProps = result["format"]["schema"]["properties"]

        self.assertEqual(resultSchemaProps["count"]["type"], "number")
        self.assertEqual(resultSchemaProps["count"]["description"], "Count")
        self.assertEqual(resultSchemaProps["count"]["minimum"], 1)
        self.assertEqual(resultSchemaProps["count"]["maximum"], 3)

    # string representations of primitive types
    def test_maps_primitive_string_tokens_to_json_schema_types(self):
        result = JSONSchemaFormat(
            {
                "s": "string",
                "n": "number",
                "i": "integer",
                "f": "float",
                "b": "boolean",
                "z": "null",
            }
        )
        resultSchemaProps = result["format"]["schema"]["properties"]

        self.assertEqual(resultSchemaProps["s"]["type"], "string")
        self.assertEqual(resultSchemaProps["n"]["type"], "number")
        self.assertEqual(resultSchemaProps["i"]["type"], "number")
        self.assertEqual(resultSchemaProps["f"]["type"], "number")
        self.assertEqual(resultSchemaProps["b"]["type"], "boolean")
        self.assertEqual(resultSchemaProps["z"]["type"], "null")

    def test_disambiguates_string_description_tuple_again_not_enum(self):
        result = JSONSchemaFormat({"mode": ["string", "Mode description"]})
        resultSchemaProps = result["format"]["schema"]["properties"]

        self.assertEqual(resultSchemaProps["mode"]["type"], "string")
        self.assertEqual(resultSchemaProps["mode"]["description"], "Mode description")


if __name__ == "__main__":
    unittest.main()
