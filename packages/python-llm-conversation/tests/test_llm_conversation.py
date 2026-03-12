import sys
import unittest
from importlib import import_module
from pathlib import Path
from typing import Any


# Ensure local src layout is importable when running:
# python -m unittest discover tests
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

pkg = import_module("mightydatainc_llm_conversation")
LLMConversation = getattr(pkg, "LLMConversation")


class FakeResponse:
    def __init__(
        self,
        output_text: Any = "",
        error: Any = None,
        incomplete_details: Any = None,
    ):
        self.output_text = output_text
        self.error = error
        self.incomplete_details = incomplete_details


class FakeResponsesAPI:
    def __init__(self, side_effects: list[Any] | None = None):
        self.side_effects = list(side_effects or [])
        self.create_calls: list[dict[str, Any]] = []

    def create(self, kwargs: dict[str, Any]):
        self.create_calls.append(kwargs)

        if not self.side_effects:
            return FakeResponse()

        next_effect = self.side_effects.pop(0)
        if isinstance(next_effect, BaseException):
            raise next_effect
        return next_effect


class FakeOpenAIClient:
    def __init__(self, side_effects: list[Any] | None = None):
        self.responses = FakeResponsesAPI(side_effects)


class TestLLMConversation(unittest.TestCase):
    def test_defaults_to_empty_state_with_no_client_or_model(self):
        conversation = LLMConversation()

        self.assertEqual(conversation, [])
        self.assertIsNone(conversation.ai_client)
        self.assertIsNone(conversation.model)
        self.assertIsNone(conversation.last_reply)

    def test_assign_messages_replaces_contents_and_returns_self(self):
        conversation = LLMConversation(
            None,
            [
                {
                    "role": "user",
                    "content": "old",
                }
            ],
        )
        next_messages = [
            {
                "role": "system",
                "content": "new",
            },
            {
                "role": "user",
                "content": "hello",
            },
        ]

        returned = conversation.assign_messages(next_messages)

        self.assertIs(returned, conversation)
        self.assertEqual(conversation, next_messages)

    def test_clone_deep_copies_messages_not_message_references(self):
        client = FakeOpenAIClient()
        conversation = LLMConversation(
            client,
            [
                {
                    "role": "user",
                    "content": "foo",
                }
            ],
        )

        cloned = conversation.clone()

        self.assertIsNot(cloned, conversation)
        self.assertEqual(cloned, conversation)

        conversation[0]["content"] = "bar"
        self.assertEqual(cloned[0]["content"], "foo")

    def test_clone_copies_client_and_model(self):
        client = FakeOpenAIClient()
        conversation = LLMConversation(client, None, "gpt-custom")

        cloned = conversation.clone()

        self.assertIs(cloned.ai_client, client)
        self.assertEqual(cloned.model, "gpt-custom")

    def test_clone_deep_copies_last_reply(self):
        client = FakeOpenAIClient()
        conversation = LLMConversation(
            client,
            [
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
        )
        conversation.last_reply = {"result": "ok"}

        cloned = conversation.clone()

        self.assertEqual(cloned.last_reply, {"result": "ok"})
        self.assertIsNot(cloned.last_reply, conversation.last_reply)
        conversation.last_reply["result"] = "changed"
        self.assertEqual(cloned.last_reply, {"result": "ok"})

    def test_add_message_serializes_object_content_as_pretty_json(self):
        conversation = LLMConversation()

        conversation.add_message("user", {"a": 1})

        self.assertEqual(
            conversation[0],
            {
                "role": "user",
                "content": '{\n  "a": 1\n}',
            },
        )

    def test_role_specific_helpers_add_expected_role_labels(self):
        conversation = LLMConversation()

        conversation.add_user_message("u")
        conversation.add_assistant_message("a")
        conversation.add_system_message("s")
        conversation.add_developer_message("d")

        self.assertEqual(
            [message["role"] for message in conversation],
            ["user", "assistant", "system", "developer"],
        )

    def test_add_image_throws_if_no_llm_provider_is_specified(self):
        conversation = LLMConversation()
        img_data_url = "data:image/png;base64,abc123"

        with self.assertRaises(Exception):
            conversation.add_image("user", "This is an image", img_data_url)

    def test_submit_throws_if_no_client_is_configured(self):
        conversation = LLMConversation(
            None,
            [
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
        )

        with self.assertRaises(Exception) as ctx:
            conversation.submit()

        self.assertIn(
            "AI client is not set. Please provide an AI client.",
            str(ctx.exception),
        )

    def test_submit_appends_user_message_then_assistant_reply(self):
        client = FakeOpenAIClient([FakeResponse("assistant reply")])
        conversation = LLMConversation(client)

        result = conversation.submit("hello")

        self.assertEqual(result, "assistant reply")
        self.assertEqual(
            conversation,
            [
                {
                    "role": "user",
                    "content": "hello",
                },
                {
                    "role": "assistant",
                    "content": "assistant reply",
                },
            ],
        )
        self.assertEqual(conversation.last_reply, "assistant reply")

    def test_submit_passes_through_array_content_for_multimodal_messages(self):
        client = FakeOpenAIClient([FakeResponse("Nice image!")])
        conversation = LLMConversation(client)

        img_data_url = "data:image/png;base64,abc123"
        message = {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Hello World. Here's an image.",
                },
                {
                    "type": "input_image",
                    "image_url": img_data_url,
                    "detail": "high",
                },
            ],
        }

        result = conversation.submit(message)

        self.assertEqual(result, "Nice image!")
        self.assertEqual(
            conversation[0],
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Hello World. Here's an image.",
                    },
                    {
                        "type": "input_image",
                        "image_url": img_data_url,
                        "detail": "high",
                    },
                ],
            },
        )
        self.assertEqual(
            conversation[1],
            {
                "role": "assistant",
                "content": "Nice image!",
            },
        )

    def test_submit_wrapper_methods_add_role_and_return_reply(self):
        client = FakeOpenAIClient(
            [
                FakeResponse("r1"),
                FakeResponse("r2"),
                FakeResponse("r3"),
                FakeResponse("r4"),
                FakeResponse("r5"),
            ]
        )
        conversation = LLMConversation(client)

        self.assertEqual(conversation.submit_message("system", "m1"), "r1")
        self.assertEqual(conversation.submit_user_message("m2"), "r2")
        self.assertEqual(conversation.submit_assistant_message("m3"), "r3")
        self.assertEqual(conversation.submit_system_message("m4"), "r4")
        self.assertEqual(conversation.submit_developer_message("m5"), "r5")

        self.assertEqual(
            [
                message["role"]
                for index, message in enumerate(conversation)
                if index % 2 == 0
            ],
            ["system", "user", "assistant", "system", "developer"],
        )

    def test_last_reply_accessors_enforce_expected_types(self):
        conversation = LLMConversation()

        conversation.add_assistant_message("hello")

        self.assertEqual(conversation.get_last_reply_str(), "hello")
        self.assertEqual(conversation.get_last_reply_dict(), {})

        conversation.add_assistant_message({"x": 10, "nested": {"y": 1}})

        self.assertEqual(conversation.get_last_reply_str(), "")
        cloned = conversation.get_last_reply_dict()
        self.assertEqual(cloned, {"x": 10, "nested": {"y": 1}})
        cloned["nested"]["y"] = 2
        self.assertEqual(conversation.last_reply["nested"]["y"], 1)
        self.assertEqual(conversation.get_last_reply_dict_field("x"), 10)
        self.assertEqual(conversation.get_last_reply_dict_field("missing", 99), 99)

        conversation.add_assistant_message("not dict")
        self.assertIsNone(conversation.get_last_reply_dict_field("x"))


if __name__ == "__main__":
    unittest.main()
