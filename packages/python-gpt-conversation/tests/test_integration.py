"""Integration tests that exercise GptConversation against the live OpenAI API.

These tests require a valid OPENAI_API_KEY in the .env file (or environment).
They are intentionally slow and make real API calls.
"""

import base64
import os
import sys
import unittest
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

load_dotenv(ROOT / ".env")

from openai import OpenAI

from gpt_conversation.functions import GPT_MODEL_VISION
from gpt_conversation.gpt_conversation import GptConversation
from gpt_conversation.json_schema_format import JSONSchemaFormat

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is required for live API tests. "
        "Configure your .env file to provide it."
    )

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def make_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def load_image_data_url(filename: str) -> str:
    data = (FIXTURES_DIR / filename).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


IMAGE_IDENTIFICATION_SCHEMA = JSONSchemaFormat(
    {
        "image_subject_enum": [
            "house",
            "chair",
            "boat",
            "car",
            "cat",
            "dog",
            "telephone",
            "duck",
            "city_skyline",
            "still_life",
            "bed",
            "headphones",
            "skull",
            "photo_camera",
            "unknown",
            "none",
            "error",
        ],
    },
    name="ImageIdentification",
    description="A test schema for image identification response",
)


class TestIntegration(unittest.TestCase):

    def test_should_repeat_hello_world(self):
        convo = GptConversation([], openai_client=make_client())

        convo.add_user_message(
            "This is a test to see if I'm correctly calling the OpenAI API to invoke GPT.\n"
            'If you can see this, please respond with "Hello World" -- just like that,\n'
            "with no additional text or explanation. Do not include punctuation or quotation\n"
            'marks. Emit only the words "Hello World", capitalized as shown.'
        )
        convo.submit()

        self.assertEqual(convo.get_last_reply_str(), "Hello World")

    def test_should_invoke_an_llm_with_some_nominal_intelligence(self):
        convo = GptConversation([], openai_client=make_client())

        # Use the submit_user_message convenience method.
        convo.submit_user_message(
            "I'm conducting a test of my REST API response parsing systems.\n"
            "If you can see this, please reply with the capital of France.\n"
            "Reply *only* with the name of the city, with no additional text, punctuation,\n"
            "or explanation. I'll be comparing your output string to a standard known\n"
            "value, so it's important to the integrity of my system that the *only*\n"
            "response you produce be *just* the name of the city. Standard capitalization\n"
            "please -- first letter capitalized, all other letters lower-case."
        )

        self.assertEqual(convo.get_last_reply_str(), "Paris")

    def test_should_reply_with_a_general_form_json_object(self):
        convo = GptConversation([], openai_client=make_client())

        convo.add_user_message(
            "This is a test to see if I'm correctly calling the OpenAI API to invoke GPT.\n\n"
            "Please reply with the following JSON object, exactly as shown:\n\n"
            "{\n"
            '  "text": "Hello World",\n'
            '  "success": true,\n'
            '  "sample_array_data": [1, 2, {"nested_key": "nested_value"}]\n'
            "}"
        )
        convo.submit(json_response=True)

        reply = convo.get_last_reply_dict()

        self.assertEqual(reply.get("text"), "Hello World")
        self.assertEqual(reply.get("success"), True)
        self.assertIn("sample_array_data", reply)
        arr = reply["sample_array_data"]
        self.assertIsInstance(arr, list)
        self.assertEqual(len(arr), 3)
        self.assertEqual(arr[0], 1)
        self.assertEqual(arr[1], 2)
        self.assertEqual(arr[2].get("nested_key"), "nested_value")

        # Also verify the shortcut accessors work.
        self.assertEqual(convo.get_last_reply_dict_field("text"), "Hello World")
        self.assertEqual(convo.get_last_reply_dict_field("success"), True)
        self.assertEqual(len(convo.get_last_reply_dict_field("sample_array_data")), 3)

    def test_should_reply_with_structured_json_using_raw_schema(self):
        convo = GptConversation([], openai_client=make_client())

        schema = {
            "format": {
                "type": "json_schema",
                "strict": True,
                "name": "TestSchema",
                "description": "A test schema for structured JSON response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "success": {"type": "boolean"},
                        "sample_array_data": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "nested_dict": {
                            "type": "object",
                            "properties": {
                                "nested_key": {"type": "string"},
                            },
                            "required": ["nested_key"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["text", "success", "sample_array_data", "nested_dict"],
                    "additionalProperties": False,
                },
            },
        }

        convo.add_user_message(
            "Please reply with a JSON object that contains the following data:\n\n"
            "Success flag: true\n"
            'Text: "Hello World"\n'
            "Sample array data (2 elements long):\n"
            "    Element 0: 5\n"
            "    Element 1: 33\n"
            "Nested dict (1 item long):\n"
            '    Value under "nested_key": "foobar"'
        )
        convo.submit(json_response=schema)

        self.assertEqual(convo.get_last_reply_dict_field("success"), True)
        self.assertEqual(convo.get_last_reply_dict_field("text"), "Hello World")

        nested_dict = convo.get_last_reply_dict_field("nested_dict")
        self.assertEqual(nested_dict.get("nested_key"), "foobar")
        self.assertEqual(len(nested_dict), 1)

        sample_array = convo.get_last_reply_dict_field("sample_array_data")
        self.assertEqual(len(sample_array), 2)
        self.assertEqual(sample_array[0], 5)
        self.assertEqual(sample_array[1], 33)

    def test_should_reply_with_structured_json_using_json_schema_format_shorthand(self):
        convo = GptConversation([], openai_client=make_client())

        schema = JSONSchemaFormat(
                "success": bool,
                "sample_array_data": [int],
                "nested_dict": {
                    "nested_key": str,
                },
            },
            name="TestSchema",
            description="A test schema for structured JSON response",
        )

        convo.add_user_message(
            "Please reply with a JSON object that contains the following data:\n\n"
            "Success flag: true\n"
            'Text: "Hello World"\n'
            "Sample array data (2 elements long):\n"
            "    Element 0: 5\n"
            "    Element 1: 33\n"
            "Nested dict (1 item long):\n"
            '    Value under "nested_key": "foobar"'
        )
        convo.submit(json_response=schema)

        self.assertEqual(convo.get_last_reply_dict_field("success"), True)
        self.assertEqual(convo.get_last_reply_dict_field("text"), "Hello World")

        nested_dict = convo.get_last_reply_dict_field("nested_dict")
        self.assertEqual(nested_dict.get("nested_key"), "foobar")
        self.assertEqual(len(nested_dict), 1)

        sample_array = convo.get_last_reply_dict_field("sample_array_data")
        self.assertEqual(len(sample_array), 2)
        self.assertEqual(sample_array[0], 5)
        self.assertEqual(sample_array[1], 33)

    def test_should_perform_image_recognition_with_manual_content_message(self):
        convo = GptConversation([], openai_client=make_client(), model=GPT_MODEL_VISION)

        img_data_url = load_image_data_url("phoenix.png")

        # Build the multi-modal message manually (without convenience methods).
        gpt_msg_with_image = {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "An image submitted by a user, needing identification",
                },
                {
                    "type": "input_image",
                    "image_url": img_data_url,
                    "detail": "high",
                },
            ],
        }
        convo.append(gpt_msg_with_image)
        convo.add_user_message("What is this a picture of?")

        convo.submit(json_response=IMAGE_IDENTIFICATION_SCHEMA)

        self.assertEqual(convo.get_last_reply_dict_field("image_subject_enum"), "cat")

    def test_should_perform_image_recognition_with_convenience_methods(self):
        convo = GptConversation([], openai_client=make_client(), model=GPT_MODEL_VISION)

        img_data_url = load_image_data_url("phoenix.png")

        convo.add_image(
            "user",
            "An image submitted by a user, needing identification",
            img_data_url,
        )
        convo.add_user_message("What is this a picture of?")

        convo.submit(json_response=IMAGE_IDENTIFICATION_SCHEMA)

        self.assertEqual(convo.get_last_reply_dict_field("image_subject_enum"), "cat")


if __name__ == "__main__":
    unittest.main()
