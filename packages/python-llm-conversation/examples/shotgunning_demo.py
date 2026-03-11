from openai import OpenAI
from mightydatainc_gpt_conversation import GptConversation

client = OpenAI()
conversation = GptConversation(openai_client=client)

from openai import OpenAI
from mightydatainc_gpt_conversation import GptConversation

client = OpenAI()
conversation = GptConversation(openai_client=client)

conversation.submit_user_message(
    "Write a short story about a raccoon who steals the Mona Lisa."
)
conversation.add_user_message(
    "Think of one improvement that can be made to the story, and write a revised version of the story that incorporates that improvement."
)
conversation.submit(shotgun=3)
story = conversation.get_last_reply_str()

print(story)
