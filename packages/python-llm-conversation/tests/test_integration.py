"""Integration tests that exercise GptConversation against the live OpenAI API.

These tests require a valid OPENAI_API_KEY in the .env file (or environment).
They are intentionally slow and make real API calls.
"""

import base64
import concurrent.futures
import json
import os
import sys
from typing import Any, Dict, List
import unittest
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

load_dotenv(ROOT / ".env")

from openai import OpenAI

from mightydatainc_gpt_conversation.functions import GPT_MODEL_VISION
from mightydatainc_gpt_conversation.gpt_conversation import GptConversation
from mightydatainc_gpt_conversation.json_schema_format import JSONSchemaFormat

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
        convo = GptConversation(make_client())

        convo.add_user_message(
            """
This is a test to see if I'm correctly calling the OpenAI API to invoke GPT.
If you can see this, please respond with "Hello World" -- just like that,
with no additional text or explanation. Do not include punctuation or quotation
marks. Emit only the words "Hello World", capitalized as shown.
"""
        )
        convo.submit()

        self.assertEqual(convo.get_last_reply_str(), "Hello World")

    def test_should_invoke_an_llm_with_some_nominal_intelligence(self):
        convo = GptConversation(make_client())

        # Use the submit_user_message convenience method.
        convo.submit_user_message(
            """
I'm conducting a test of my REST API response parsing systems.
If you can see this, please reply with the capital of France.
Reply *only* with the name of the city, with no additional text, punctuation,
or explanation. I'll be comparing your output string to a standard known
value, so it's important to the integrity of my system that the *only*
response you produce be *just* the name of the city. Standard capitalization
please -- first letter capitalized, all other letters lower-case.
"""
        )

        self.assertEqual(convo.get_last_reply_str(), "Paris")

    def test_should_reply_with_a_general_form_json_object(self):
        convo = GptConversation(make_client())

        convo.add_user_message(
            """
This is a test to see if I'm correctly calling the OpenAI API to invoke GPT.

Please reply with the following JSON object, exactly as shown:

{
  "text": "Hello World",
  "success": true,
  "sample_array_data": [1, 2, {"nested_key": "nested_value"}]
}
"""
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
        convo = GptConversation(make_client())

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
            """
Please reply with a JSON object that contains the following data:

Success flag: true
Text: "Hello World"
Sample array data (2 elements long):
    Element 0: 5
    Element 1: 33
Nested dict (1 item long):
    Value under "nested_key": "foobar"
"""
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
        convo = GptConversation(make_client())

        schema = JSONSchemaFormat(
            {
                "success": bool,
                "text": str,
                "sample_array_data": [int],
                "nested_dict": {
                    "nested_key": str,
                },
            },
            name="TestSchema",
            description="A test schema for structured JSON response",
        )

        convo.add_user_message(
            """
Please reply with a JSON object that contains the following data:

Success flag: true
Text: "Hello World"
Sample array data (2 elements long):
    Element 0: 5
    Element 1: 33
Nested dict (1 item long):
    Value under "nested_key": "foobar"
"""
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
        convo = GptConversation(make_client(), model=GPT_MODEL_VISION)

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
        convo = GptConversation(make_client(), model=GPT_MODEL_VISION)

        img_data_url = load_image_data_url("phoenix.png")

        convo.add_image(
            "user",
            "An image submitted by a user, needing identification",
            img_data_url,
        )
        convo.add_user_message("What is this a picture of?")

        convo.submit(json_response=IMAGE_IDENTIFICATION_SCHEMA)

        self.assertEqual(convo.get_last_reply_dict_field("image_subject_enum"), "cat")

    def test_shotgun_to_get_reliable_answer_on_unreliable_prompt(self):
        """Intentionally flaky without shotgun: LLMs miscount repeated letters.."""
        # Adjust this number as needed to achieve a reliable pass rate.
        # A ridiculous number of barrels but this test needs to be reliable.
        NUM_SHOTGUN_BARRELS = 10

        convo = GptConversation(make_client())

        convo.add_developer_message(
            """
Count the number of times each letter of the alphabet appears in a key phrase
that the user will give you.

Ignore spaces, and treat all letters as lowercase for counting purposes.
Do not count any characters other than the 26 letters of the English alphabet.

Return a JSON object where each key is a lowercase letter and each
value is the integer count of that letter. Include only letters that appear at least
once. Emit nothing except the JSON object. E.g. it should look like this:

{
  "a": 99,
  "b": 99,
  "c": 99,
  ...
}

Except, of course, with the correct counts for the letters instead of "99".
Your response should include all 26 keys, appearing in order from "a" to "z",
even if the count for some letters is zero.
"""
        )
        convo.add_user_message("strawberry milkshake")

        formatparam: Dict[str, Any] = {
            "scratchpad": (
                str,
                (
                    "An internal deliberation you can have with yourself about how to best answer "
                    "the question. Use this as a whiteboard to work through your reasoning process. "
                    "PRO TIP: Be very careful to not count any position twice. If you find that "
                    "you're counting one letter for position n, and then counting another letter "
                    "for position n, then one or both must be wrong."
                ),
            ),
        }

        # Iterate a-z and add each letter as a key with the value "int".
        for letter in "abcdefghijklmnopqrstuvwxyz":
            formatparam[letter] = {
                "count": int,
                "locations": (
                    str,
                    (
                        "An explicit list of the places where you found this letter. "
                        "It should describe <count> distinct locations in the key phrase "
                        "where this letter appears. Actually write out the text at the "
                        "locations to prove that you found them, like this: "
                        '[position 2, the first "o" in "foo": f *o* o], '
                        '[position 3, the second "o" in "foo": f o *o*], '
                    ),
                ),
            }

        # Clone the convo to a "savepoint". This will allow us to validate its performance
        # against different run modalities.
        convo_before_submit = convo.clone()

        # Without shotgunning, this fails >90% of the time.
        # Notably, it fails a *different* way each time.
        # Because of this multivariate leverage, the shotgun approach
        # is astronomically more likely to get a perfect answer than
        # any single attempt is on its own.
        # It's a lot of barrels, but we can't let this test be flaky.
        convo.submit(
            shotgun=NUM_SHOTGUN_BARRELS,
            json_response=JSONSchemaFormat(formatparam),
        )

        reply = convo.get_last_reply_dict()

        # Make sure it has all 26 letters plus the discussion field.
        self.assertEqual(len(reply), 27)

        # strawberry milkshake
        # s(2) t(1) r(3) a(2) w(1) b(1) e(2) y(1) m(1) i(1) l(1) k(2) h(1)
        expected_counts = {
            "a": 2,
            "b": 1,
            "c": 0,
            "d": 0,
            "e": 2,
            "f": 0,
            "g": 0,
            "h": 1,
            "i": 1,
            "j": 0,
            "k": 2,
            "l": 1,
            "m": 1,
            "n": 0,
            "o": 0,
            "p": 0,
            "q": 0,
            "r": 3,
            "s": 2,
            "t": 1,
            "u": 0,
            "v": 0,
            "w": 1,
            "x": 0,
            "y": 1,
            "z": 0,
        }
        observed_counts = {
            letter: reply[letter]["count"] for letter in expected_counts.keys()
        }
        self.assertEqual(observed_counts, expected_counts)

        # We need to validate that it reliably fails without shotgunning to confirm that the shotgun
        # approach is actually necessary to get a consistent pass on this test. If it passes 100%
        # of the time without shotgunning, then this test isn't actually doing anything useful.
        # Use concurrent futures to run NUM_SHOTGUN_BARRELS attempts in parallel, and assert that
        # at least one of them fails.
        convo = convo_before_submit.clone()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    convo.submit,
                    json_response=JSONSchemaFormat(formatparam),
                )
                for _ in range(NUM_SHOTGUN_BARRELS)
            ]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        does_each_result_equal_expected = []
        for result in results:
            observed_counts = {
                letter: result[letter]["count"] for letter in expected_counts.keys()
            }
            does_each_result_equal_expected.append(observed_counts == expected_counts)

        # Assert that at least one of the attempts failed to get the correct answer.
        self.assertFalse(all(does_each_result_equal_expected))


if __name__ == "__main__":
    unittest.main()
