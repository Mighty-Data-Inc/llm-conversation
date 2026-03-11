from openai import OpenAI
from mightydatainc_gpt_conversation import GptConversation
from mightydatainc_gpt_conversation.json_schema_format import JSONSchemaFormat

client = OpenAI()
conversation = GptConversation(openai_client=client)

story = conversation.submit_user_message(
    "Write a short story about a raccoon who steals the Mona Lisa. "
    "Give the raccoon between 0 and 4 accomplices who help them in their heist, "
    "each of a different species."
)

# We *could* add a separate user message telling the AI to answer a few questions,
# but honestly just submitting this JSON query is enough to make the AI "understand"
# what we want it to do.
conversation.submit(
    json_response=JSONSchemaFormat(
        {
            "protagonist_name": str,
            "city": (
                str,
                "Where does this story take place?",
            ),
            "number_of_theft_attempts": (
                int,
                "How many attempts do they make during the course of the story?",
                (0, 10),
            ),
            "do_they_ultimately_succeed": bool,
            "accomplices": [
                {
                    "name": str,
                    "species": (
                        str,
                        "What species is this accomplice?",
                        [
                            "raccoon",
                            "cat",
                            "dog",
                            "ferret",
                            "squirrel",
                            "pigeon",
                            "human",
                            "other",
                        ],
                    ),
                    "species_unusual": (
                        str,
                        "If the species is 'other', please specify it here. Otherwise, leave this blank.",
                    ),
                }
            ],
        }
    )
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

for accomplice in conversation.get_last_reply_dict_field("accomplices"):
    species = accomplice["species"]
    if species == "other":
        species = accomplice["species_unusual"] + " (unusual)"
    print(
        f"Accomplice {accomplice['name']} is a {species}.",
    )
