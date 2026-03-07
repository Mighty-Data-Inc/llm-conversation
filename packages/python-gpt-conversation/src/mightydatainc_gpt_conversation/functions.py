"""Core GPT API helpers for message submission and response parsing.

This module centralizes OpenAI request/response handling used by the package,
including:
- model defaults,
- datetime/system message injection,
- optional JSON-response configuration,
- retry behavior for recoverable failures, and
- optional warning emission via callback.

The primary entry point is ``gpt_submit``.
"""

import concurrent.futures
import copy
import datetime
import json
import openai
import time

import httpx

from typing import Any, Callable, Protocol, cast, Dict, List, Optional, Tuple, Union

from openai._types import Omit, omit
from openai.types.responses import EasyInputMessage, ResponseTextConfigParam


GPT_MODEL_CHEAP = "gpt-4.1-nano"
GPT_MODEL_SMART = "gpt-4.1"
GPT_MODEL_VISION = "gpt-4.1"

_GPT_RETRY_LIMIT_DEFAULT = 5
_GPT_RETRY_BACKOFF_TIME_SECONDS_DEFAULT = 30  # seconds


class _ResponsesAPI(Protocol):
    create: Callable[..., Any]


class OpenAIClientLike(Protocol):
    @property
    def responses(self) -> _ResponsesAPI: ...


def current_datetime_system_message() -> Dict[str, str]:
    """Build a system message containing the current local date and time.

    This helper returns a message object in the OpenAI chat format
    (``{"role": ..., "content": ...}``) so it can be prepended to a
    conversation and provide the model with temporal context.

    Returns:
        Dict[str, str]: A system message with ``role`` set to ``"system"`` and
        ``content`` set to a !DATETIME string formatted as
        ``YYYY-MM-DD HH:MM:SS``.
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    retval = {
        "role": "system",
        "content": f"!DATETIME: The current date and time is {current_time}",
    }
    return retval


def gpt_submit(
    messages: list,
    openai_client: OpenAIClientLike,
    *,
    model: Optional[str] = None,
    json_response: Optional[Union[bool, dict, str]] = None,
    system_announcement_message: Optional[str] = None,
    retry_limit: Optional[int] = None,
    retry_backoff_time_seconds: Optional[int] = None,
    shotgun: Optional[int] = None,
    warning_callback: Optional[Callable[[str], None]] = None,
) -> Union[str, dict, list]:
    """Submit messages to an OpenAI Responses client with retry handling.

    The function prepends a fresh ``!DATETIME`` system message on every call,
    optionally prepends a caller-provided system announcement, and then submits
    the request through ``openai_client.responses.create``.

    Args:
        messages: Message objects in OpenAI-style chat format.
        openai_client: Client-like object exposing ``responses.create``.
        model: Model name override. Defaults to ``GPT_MODEL_SMART``.
        json_response: JSON mode selector.
            - ``True``: requests ``{"format": {"type": "json_object"}}``.
            - ``dict``: uses the provided text format config (deep-copied).
            - ``str``: parsed as JSON and used as text format config.
            - ``None``/``False``: plain text mode.
        system_announcement_message: Optional additional system message placed
            before the auto-generated datetime system message.
        retry_limit: Maximum number of attempts for retryable failures.
        retry_backoff_time_seconds: Delay between retries for OpenAI API errors.
        shotgun: When greater than 1, the request is sent this many times in
            parallel and the results are reconciled into a single reply. Use
            this when output quality matters more than latency or cost —
            higher values yield better answers at the expense of proportionally
            more API calls. Values of ``None`` or ``1`` behave normally.
        warning_callback: Optional callback for recoverable warning strings
            (non-fatal API warnings and retry notices).

    Returns:
        In text mode, returns a stripped ``str``.
        In JSON mode, returns the first JSON value parsed from the model output,
        typically a ``dict`` or ``list``.

    Raises:
        openai.AuthenticationError: Immediately, without retrying, when the API
            key is missing or invalid.
        openai.PermissionDeniedError: Immediately, without retrying, when the
            request is forbidden.
        openai.OpenAIError: When other API errors persist through all retries.
        json.JSONDecodeError: When JSON parsing fails through all retries in
            JSON mode, or when ``json_response`` is an invalid JSON string.
        ValueError: If no attempt runs or no terminal failure is captured.
        AttributeError: If the response object is missing expected attributes.
    """
    if shotgun and shotgun > 1:
        return _gpt_submit_shotgun(
            messages,
            openai_client,
            num_barrels=shotgun,
            model=model,
            json_response=json_response,
            system_announcement_message=system_announcement_message,
            retry_limit=retry_limit,
            retry_backoff_time_seconds=retry_backoff_time_seconds,
            warning_callback=warning_callback,
        )

    if not model:
        model = GPT_MODEL_SMART

    if retry_limit is None:
        retry_limit = _GPT_RETRY_LIMIT_DEFAULT
    if retry_backoff_time_seconds is None:
        retry_backoff_time_seconds = _GPT_RETRY_BACKOFF_TIME_SECONDS_DEFAULT

    efail = None

    openai_text_param: ResponseTextConfigParam | Omit = omit
    if json_response:
        if isinstance(json_response, bool):
            openai_text_param = {"format": {"type": "json_object"}}
        elif isinstance(json_response, dict):
            # Deep copy to avoid modifying caller's object
            json_response = json.loads(json.dumps(json_response))

            openai_text_param = cast(ResponseTextConfigParam, json_response)
            # Check if format exists and has description before modifying
            if (
                "format" in openai_text_param
                and "description" in openai_text_param["format"]
            ):
                # Append instructions to the description to ensure JSON output.
                format_dict = openai_text_param["format"]
                if isinstance(format_dict, dict):
                    format_dict["description"] = format_dict.get("description", "")
                    format_dict["description"] += (
                        "\n\nABSOLUTELY NO UNICODE ALLOWED. Only use typeable keyboard characters. "
                        "Do not try to circumvent this rule with escape sequences, "
                        'backslashes, or other tricks. Use double dashes (--), straight quotes ("), '
                        "and single quotes (') instead of em-dashes, en-dashes, and curly versions."
                    )
                    format_dict["description"] = format_dict["description"].strip()

        elif isinstance(json_response, str):
            openai_text_param = json.loads(json_response)

    # Clear any existing datetime system message and add a fresh one.
    messages = [
        m
        for m in messages
        if not (
            type(m) is dict
            and m.get("role") == "system"
            and type(m.get("content")) is str
            and f"{m['content']}".startswith("!DATETIME:")
        )
    ]
    messages = [current_datetime_system_message()] + messages
    if system_announcement_message and system_announcement_message.strip():
        messages = [
            {
                "role": "system",
                "content": system_announcement_message.strip(),
            }
        ] + messages

    for iretry in range(retry_limit):
        llmreply = ""
        try:
            # Attempt to get a response from the OpenAI API
            llmresponse = openai_client.responses.create(
                model=model,
                input=messages,
                text=openai_text_param,
            )
            if llmresponse.error:
                if warning_callback:
                    warning_callback(
                        f"ERROR: OpenAI API returned an error: {llmresponse.error}"
                    )
            if llmresponse.incomplete_details:
                if warning_callback:
                    warning_callback(
                        "ERROR: OpenAI API returned incomplete details: "
                        f"{llmresponse.incomplete_details}"
                    )
            llmreply = llmresponse.output_text.strip()
            if not json_response:
                return f"{llmreply}"

            # If we got here, then we expect a JSON response,
            # which will be a dictionary or a list.
            # We'll use raw_decode rather than loads to parse it, because
            # GPT has a habit of concatenating multiple JSON objects
            # for some reason (raw_decode will stop at the end of the first object,
            # whereas loads will raise an error if there's any trailing text).
            (llmobj, _) = json.JSONDecoder().raw_decode(llmreply)
            llmobj: Union[dict, list] = llmobj
            return llmobj
        except openai.OpenAIError as e:
            # Non-retryable errors should propagate immediately — retrying
            # with a backoff would only waste time and obscure the problem.
            if isinstance(
                e, (openai.AuthenticationError, openai.PermissionDeniedError)
            ):
                raise
            # A local protocol error (e.g. illegal header value caused by a
            # malformed API key) means the request can't be sent at all.
            # It will fail identically on every retry.
            if isinstance(e, openai.APIConnectionError) and isinstance(
                e.__cause__, httpx.LocalProtocolError
            ):
                raise
            efail = e
            if warning_callback:
                warning_callback(
                    f"OpenAI API error:\n\n{e}.\n\n"
                    f"Retrying (attempt {iretry + 1} of {retry_limit}) "
                    f"in {retry_backoff_time_seconds} seconds..."
                )
            time.sleep(retry_backoff_time_seconds)
        except json.JSONDecodeError as e:
            efail = e
            if warning_callback:
                warning_callback(
                    f"JSON decode error:\n\n{e}.\n\n"
                    f"Raw text of LLM Reply:\n{llmreply}\n\n"
                    f"Retrying (attempt {iretry + 1} of {retry_limit}) immediately..."
                )

    # Propagate the last error after all retries
    if efail:
        raise efail
    raise ValueError("Unknown error occurred in _gpt_helpers")


def _gpt_submit_shotgun(
    messages: list,
    openai_client: OpenAIClientLike,
    num_barrels: int,
    *,
    model: Optional[str] = None,
    json_response: Optional[Union[bool, dict, str]] = None,
    system_announcement_message: Optional[str] = None,
    retry_limit: Optional[int] = None,
    retry_backoff_time_seconds: Optional[int] = None,
    warning_callback: Optional[Callable[[str], None]] = None,
) -> Union[str, dict, list]:
    """Submit a conversation using the shotgun strategy for higher-quality replies.

    Fires ``num_barrels`` parallel requests with identical inputs using a thread
    pool, then asks the model to examine all responses and reconcile them into a
    single authoritative answer.  This improves output quality for tasks where
    the model benefits from exploring multiple reasoning paths simultaneously.

    If ``num_barrels`` is 1 or fewer, the function falls back to a single
    ``gpt_submit`` call with no overhead.

    Args:
        messages: Message objects in OpenAI-style chat format.
        openai_client: Client-like object exposing ``responses.create``.
        num_barrels: Number of parallel worker requests to fire.
        model: Model name override. Defaults to ``GPT_MODEL_SMART``.
        json_response: JSON mode selector (same semantics as ``gpt_submit``).
        system_announcement_message: Optional system message placed before the
            auto-generated datetime system message.
        retry_limit: Maximum number of attempts for retryable failures per call.
        retry_backoff_time_seconds: Delay between retries for OpenAI API errors.
        warning_callback: Optional callback for recoverable warning strings.

    Returns:
        The reconciled response from the model — a plain ``str`` in text mode,
        or a parsed JSON value (``dict`` / ``list``) in JSON mode.
    """
    # Work with a copy of the messages list to avoid modifying the caller's object
    messages = copy.deepcopy(messages)

    def _internal_gpt_submit(
        messages: list,
    ) -> Union[str, dict, list]:
        return gpt_submit(
            messages,
            openai_client,
            model=model,
            json_response=json_response,
            system_announcement_message=system_announcement_message,
            retry_limit=retry_limit,
            retry_backoff_time_seconds=retry_backoff_time_seconds,
            warning_callback=warning_callback,
        )

    if num_barrels <= 1:
        return _internal_gpt_submit(messages)

    # Fire num_barrels identical requests in parallel.
    barrel_messages = [copy.deepcopy(messages) for _ in range(num_barrels)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_barrels) as executor:
        futures = [
            executor.submit(_internal_gpt_submit, barrel) for barrel in barrel_messages
        ]
        results_raw = [f.result() for f in futures]

    result_strings = [json.dumps(r) for r in results_raw]

    # Build the reconciliation conversation on top of the original messages.
    reconcile_messages = copy.deepcopy(messages)

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
consistent with the data you've been shown? Remember, this is an adjudication, not a democracy --
you should carefully examine the data presented in the conversation and use your best judgment
to determine which worker is most likely to be correct.
""",
        }
    )

    ponder_reply = _internal_gpt_submit(reconcile_messages)
    reconcile_messages.append({"role": "assistant", "content": ponder_reply})

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

    return _internal_gpt_submit(reconcile_messages)
