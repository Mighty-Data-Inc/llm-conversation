import json
from typing import Any


def _json_stringify(value: Any) -> str:
    try:
        return json.dumps(value)
    except TypeError:
        return json.dumps(repr(value))


def _is_number_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def convert_schema_recursive(subschema: Any) -> dict[str, Any]:
    # First we handle the easy cases: primitive types and their string aliases.
    if subschema == "integer" or subschema is int:
        # Python can distinguish integer and float at runtime.
        return {"type": "integer"}

    if subschema == "number" or subschema == "float" or subschema is float:
        return {"type": "number"}

    if subschema == "string" or subschema is str:
        return {"type": "string"}

    if subschema == "boolean" or subschema is bool:
        return {"type": "boolean"}

    if subschema == "null" or subschema is None:
        return {"type": "null"}

    # If it's a string and hasn't already been caught above, assume it's a
    # string field description.
    if isinstance(subschema, str):
        return {"type": "string", "description": f"{subschema}"}

    # Coerce tuples into lists.
    if isinstance(subschema, tuple):
        subschema = list(subschema)

    # If it's not a dict or list at this point, it's invalid.
    if not isinstance(subschema, (dict, list)):
        raise ValueError(f"Invalid schema value: {_json_stringify(subschema)}")

    # If subschema={...}, recursively convert each property and require all keys.
    if isinstance(subschema, dict):
        retval: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "required": [*subschema.keys()],
            "properties": {},
        }
        for key, value in subschema.items():
            retval["properties"][key] = convert_schema_recursive(value)
        return retval

    # If we're here, then subschema=[...].
    if len(subschema) == 0:
        raise ValueError(f"Invalid schema array: {_json_stringify(subschema)}")

    if len(subschema) == 1:
        return {
            "type": "array",
            "items": convert_schema_recursive(subschema[0]),
        }

    first_elem = subschema[0]
    second_elem = subschema[1]

    # Detect enum arrays (all strings), but disambiguate [primitive, description].
    are_all_strings = all(isinstance(elem, str) for elem in subschema)
    is_enum = are_all_strings
    if is_enum and len(subschema) == 2:
        if first_elem in ["string", "number", "integer", "float", "boolean", "null"]:
            is_enum = False

    if is_enum:
        return {
            "type": "string",
            "enum": subschema,
        }

    # It's not an enum; treat as tuple metadata format.
    retval = convert_schema_recursive(first_elem)

    # The second element must be a string (or None in Python, representing
    # null/undefined parity from TypeScript).
    if second_elem is not None:
        if not isinstance(second_elem, str):
            raise ValueError(
                f"Invalid schema tuple: {_json_stringify(subschema)}. "
                + "Second element needs to be a description string."
            )

        description = second_elem.strip()
        if len(description) > 0:
            retval["description"] = description

    is_type_numeric = retval.get("type") in ["number", "integer", "float"]

    if not is_type_numeric:
        if len(subschema) > 2:
            raise ValueError(
                f"Invalid schema tuple: {_json_stringify(subschema)}. "
                + "Non-numeric types should only have 2 elements (type + description)."
            )
        return retval

    if len(subschema) == 2:
        return retval

    third_elem = subschema[2]

    # In Python, None represents both null/undefined parity from TypeScript.
    if third_elem is None:
        return retval

    # If it's a tuple, coerce it into a list.
    if isinstance(third_elem, tuple):
        third_elem = list(third_elem)

    if not isinstance(third_elem, list) or len(third_elem) != 2:
        raise ValueError(
            f"Invalid schema tuple: {_json_stringify(subschema)}. "
            + "Numeric types with a third element need to have that element be a length-2 array representing [min, max]."
        )

    are_min_max_values_valid = all(
        _is_number_value(elem) or elem is None for elem in third_elem
    )
    if not are_min_max_values_valid:
        raise ValueError(
            f"Invalid schema tuple: {_json_stringify(subschema)}. "
            + "Numeric types with a third element need to have that element be a length-2 array of numbers (or null/undefined)."
        )

    min_value, max_value = third_elem
    if min_value is not None:
        retval["minValue"] = min_value
    if max_value is not None:
        retval["maxValue"] = max_value

    return retval


def JSONSchemaFormat(
    schema: dict[str, Any],
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    if not name:
        name = "json_schema_for_structured_response"

    result: dict[str, Any] = {
        "format": {
            "type": "json_schema",
            "strict": True,
            "name": name,
            "schema": {},
        }
    }

    if description:
        result["format"]["description"] = description

    converted = convert_schema_recursive(schema)
    result["format"]["schema"] = converted

    return result


__all__ = ["JSONSchemaFormat", "convert_schema_recursive"]
