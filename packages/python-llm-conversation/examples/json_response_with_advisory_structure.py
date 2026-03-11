from openai import OpenAI
from mightydatainc_gpt_conversation import GptConversation

client = OpenAI()
conversation = GptConversation(openai_client=client)

# The `submit*` methods actually send a real call to the LLM, and will
# take a few seconds to run. When it's finished, the story will implicitly
# get stored as part of the conversation history.
story = conversation.submit_user_message(
    "Write a short story about a raccoon who steals the Mona Lisa."
)

# We'll use an `add*` method, because the `submit*` methods are primarily
# convenience methods. The real workhorse is the `submit(...)` method itself, which
# exposes a finer array of options and variants. We'll call that next.
conversation.add_user_message(
    """
Answer a few questions about the story you just wrote.
- What is the name of the protagonist?
- Where does the story take place?
- How many attempts at theft does the protagonist make during the course of the story?
- Does the protagonist ultimately succeed?

Provide your response as a JSON object using the following structure:
{
    "protagonist_name": (string)
    "city": (string, one of: "Paris", "New York", "Tokyo", "Other")
    "number_of_theft_attempts": (int, between 0 and 10)
    "do_they_ultimately_succeed": (boolean)
}
"""
)
conversation.submit(json_response=True)

# In this example, we get the entire response object as a dict.
# Alternatively, we could use the helper method `get_last_reply_dict_field(...)`.
questionnaire = conversation.get_last_reply_dict()

print(story)
print("Protagonist: ", questionnaire["protagonist_name"])
print("City where the story takes place: ", questionnaire["city"])
print("Number of theft attempts: ", questionnaire["number_of_theft_attempts"])
print("Ultimately successful? ", questionnaire["do_they_ultimately_succeed"])
