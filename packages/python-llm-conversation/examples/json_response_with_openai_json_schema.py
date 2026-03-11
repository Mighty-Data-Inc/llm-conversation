from openai import OpenAI
from mightydatainc_gpt_conversation import GptConversation

client = OpenAI()
conversation = GptConversation(openai_client=client)

story = conversation.submit_user_message(
    "Write a short story about a raccoon who steals the Mona Lisa."
)

# We *could* add a separate user message telling the AI to answer a few questions,
# but honestly just submitting this JSON query is enough to make the AI "understand"
# what we want it to do.
conversation.submit(
    json_response={
        "format": {
            "type": "json_schema",
            "name": "raccoon_story_questionnaire",
            "description": "Answer a few questions about this raccoon story.",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "protagonist_name": {
                        "type": "string",
                        "description": "What is the name of the protagonist?",
                    },
                    "city": {
                        "type": "string",
                        "description": "Where does the story take place?",
                        "enum": ["Paris", "New York", "Tokyo", "Other"],
                    },
                    "number_of_theft_attempts": {
                        "type": "integer",
                        "description": "How many attempts at theft does the protagonist make during the course of the story?",
                        "minValue": 0,
                        "maxValue": 10,
                    },
                    "do_they_ultimately_succeed": {
                        "type": "boolean",
                        "description": "Does the protagonist ultimately succeed?",
                    },
                },
                "required": [
                    "protagonist_name",
                    "city",
                    "number_of_theft_attempts",
                    "do_they_ultimately_succeed",
                ],
                "additionalProperties": False,
            },
        }
    }
)

print(story)

# Use the helper method `get_last_reply_dict_field(...)` to get the parsed JSON responses.
print(
    "Protagonist: ",
    conversation.get_last_reply_dict_field("protagonist_name"),
)
print(
    "City where the story takes place: ",
    conversation.get_last_reply_dict_field("city"),
)
print(
    "Number of theft attempts: ",
    conversation.get_last_reply_dict_field("number_of_theft_attempts"),
)
print(
    "Ultimately successful? ",
    conversation.get_last_reply_dict_field("do_they_ultimately_succeed"),
)
