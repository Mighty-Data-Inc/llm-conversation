from openai import OpenAI
from mightydatainc_gpt_conversation import GptConversation

client = OpenAI()
conversation = GptConversation(openai_client=client)

should_include_sidekick = True
should_emphasize_character_development = True

# The `submit*` methods make a call to the LLM, and take a few seconds to run.
# They'll return the string reply produced by the LLM.
story = conversation.submit_user_message(
    "Write a short story about a raccoon who steals the Mona Lisa."
)

if should_include_sidekick or should_emphasize_character_development:
    # The `add*` methods append messages to the conversation without actually
    # sending a request to the LLM, allowing us to queue multiple instructions
    # or conditionally adjust conversation topology.
    conversation.add_user_message(
        "That's a good first draft. But I'd now like you to enhance your story as follows."
    )
    if should_include_sidekick:
        # Note that, in this example, we don't know if the LLM did or didn't include
        # a sidekick in its original draft. We're performing a blind multi-shot.
        # We're essentially saying, "Look, I don't know what you just wrote, but
        # we're willing to bet that, whatever it was, it needs a sidekick."
        conversation.add_user_message(
            "- If the story doesn't already have a sidekick, add one."
        )
    if should_emphasize_character_development:
        # Again, this blind multi-shot conversation is essentially structured to
        # implicitly flow with the understanding that we don't know what story
        # the LLM originally wrote -- but whatever it was, we bet it needs more
        # character development.
        conversation.add_user_message(
            "- Focus more on the protagonist's character development."
        )

    # Use a chain-of-thought stage to let the LLM talk through its intended changes.
    # This is what "thinking" models actually do under-the-hood. Here, you can get
    # specialized "thinking" performance for your own specific needs, by telling the
    # LLM exactly what it needs to deliberate with itself about.
    # This is a `submit*` method, which will actually send a call to the LLM.
    # Its reply will be implicitly added to the conversation history.
    conversation.submit_user_message(
        "Discuss how you'd go about revising your story to integrate these suggestions. "
        + "Don't actually write a new draft yet. Just talk about it for now."
    )

    story = conversation.submit_user_message(
        "Now emit your final draft of the story, starting with the title."
    )

print(story)
