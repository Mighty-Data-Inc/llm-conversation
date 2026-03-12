import json
from typing import Any


def llm_submit_shotgun(
    messages: list[Any],
    ai_client: Any,
    options: dict[str, Any],
    num_barrels: int,
):
    """Run shotgun parallel submissions and reconcile to one final answer."""
    from .llm_submit import llm_submit

    # Deep-copy messages so worker conversations cannot cross-mutate.
    messages = json.loads(json.dumps(messages))

    if num_barrels <= 1:
        return llm_submit(messages=messages, ai_client=ai_client, **options)

    # Remove shotgun before nested submits to prevent recursion.
    options_to_pass_to_workers = {**options}
    options_to_pass_to_workers.pop("shotgun", None)

    convo_shotgun: list[list[Any]] = []
    for _ in range(num_barrels):
        convo_barrel = json.loads(json.dumps(messages))
        convo_shotgun.append(convo_barrel)

    results_raw: list[Any] = [
        llm_submit(
            messages=convo_barrel,
            ai_client=ai_client,
            **options_to_pass_to_workers,
        )
        for convo_barrel in convo_shotgun
    ]
    result_strings = [json.dumps(result) for result in results_raw]

    reconcile_messages: list[Any] = json.loads(json.dumps(messages))

    reconcile_messages.append(
        {
            "role": "system",
            "content": f"""
SYSTEM MESSAGE:
In order to produce better results, we submitted this request/question/command/etc.
to {num_barrels} worker threads in parallel.
The system will now present each of their responses, wrapped in JSON.
The user or developer will not see these responses -- they are only for you, the assistant,
to examine and reconcile. Think of them as brainstorming or scratchpads.
""",
        }
    )

    for index, result_string in enumerate(result_strings):
        reconcile_messages.append(
            {
                "role": "system",
                "content": f"WORKER {index + 1} RESPONSE:\n\n\n{result_string}",
            }
        )

    reconcile_messages.append(
        {
            "role": "system",
            "content": """
Focus on the differences and discrepancies between the workers' responses. Where do they agree?
Where do they disagree? In the areas where they disagree, which worker's argument is most
consistent with the data you've been shown?

Remember, this is an adjudication, not a democracy -- you should carefully examine the data
presented in the conversation and use your best judgment to determine which worker is most
likely to be correct, even if they're in the minority. Evaluate their answers carefully against
the source data. If multiple workers produced different answers, then clearly there is something
subtle, deceptive, or misleading about the question or the data, and you should be especially
careful to scrutinize the workers' reasoning and the evidence they present for their answers.
At least one of them must be wrong; don't fall into the same trap he did.
""",
        }
    )

    s_ponder_reply = llm_submit(
        messages=reconcile_messages,
        ai_client=ai_client,
        **{**options_to_pass_to_workers, "json_response": None},
    )
    reconcile_messages.append({"role": "assistant", "content": s_ponder_reply})

    reconcile_messages.append(
        {
            "role": "system",
            "content": """
Having seen and reconciled the workers' responses, you are now ready to craft a proper reply to
the question/request/command/etc. This response that you craft now is the one that will be
presented to the user or developer -- it should not directly reference the workers' responses,
but should instead be a fully self-contained and complete answer that draws on the insights
you've gained from examining the workers' responses.
""",
        }
    )

    return llm_submit(
        messages=reconcile_messages,
        ai_client=ai_client,
        **options_to_pass_to_workers,
    )
