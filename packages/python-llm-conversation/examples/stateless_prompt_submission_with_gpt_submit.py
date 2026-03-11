from openai import OpenAI
from mightydatainc_gpt_conversation import gpt_submit

client = OpenAI()

story = gpt_submit(
    messages=[
        {
            "role": "user",
            "content": "Write a short story about a raccoon who steals the Mona Lisa.",
        }
    ],
    openai_client=client,
)
print(story)
