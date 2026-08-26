"""The one chat deployment this fork runs against.

Every rollout goes to `gpt-5.6-luna` on Azure AI Foundry, so the call shape is
hard-coded here rather than assembled from the config. The GPT-5 generation
renamed `max_tokens` to `max_completion_tokens` and accepts no `temperature`
other than its default, and the deployment answers a 400 for each parameter it
does not take -- one for the name, one for the value, before any of the agent's
own work happens. A config knob for either is a knob that can only be turned to
a rejected call, so there isn't one.
"""

import logging
import os
import time
from abc import ABC, abstractmethod

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

# The output-length ceiling under the name this deployment gives it. Reasoning
# models spend it on thinking before emitting anything, so it has to clear
# reasoning *plus* the answer or the response arrives truncated.
TOKEN_PARAM = "max_completion_tokens"
MAX_COMPLETION_TOKENS = 32768


def make_system_message(content: str) -> dict:
    return dict(role="system", content=content)


def make_user_message(content: str) -> dict:
    return dict(role="user", content=content)


def make_assistant_message(content: str) -> dict:
    return dict(role="assistant", content=content)


class RetryError(RuntimeError):
    """Raised when the chat API could not be reached within the retry budget."""


class AbstractChatModel(ABC):
    @abstractmethod
    def __call__(self, messages: list[dict]) -> dict:
        pass

    def get_stats(self):
        return {}


class ChatModel(AbstractChatModel):
    """`gpt-5.6-luna` on Azure AI Foundry.

    Foundry's `/openai/v1` speaks plain OpenAI rather than Azure's
    `deployments/<name>?api-version=` routing, so this is the vanilla client with
    a `base_url` and there is no deployment name to configure.
    """

    def __init__(
        self,
        model_name,
        max_tokens=MAX_COMPLETION_TOKENS,
        max_retry=4,
        min_retry_wait_time=10,
    ):
        assert max_retry > 0, "max_retry should be greater than 0"

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_retry = max_retry
        self.min_retry_wait_time = min_retry_wait_time

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        assert endpoint, "AZURE_OPENAI_ENDPOINT has to be defined in the environment"
        # Accept either the base (".../openai/v1") or the full completions URL,
        # which is what the Foundry portal hands you.
        base_url = endpoint.split("/chat/completions")[0].rstrip("/")
        self.client = OpenAI(api_key=os.getenv("AZURE_API_KEY"), base_url=base_url)

    def _completion_kwargs(self, messages):
        # No `temperature`: this deployment takes only its default and 400s on
        # anything else, including the 0.0 a benchmark run would want. No
        # `logprobs` either -- nothing in the harness reads them back.
        return {
            "model": self.model_name,
            "messages": messages,
            TOKEN_PARAM: self.max_tokens,
        }

    def __call__(self, messages: list[dict]) -> dict:
        # Initialize retry tracking attributes
        self.retries = 0
        self.success = False
        self.error_types = []

        completion = None
        last_error = None
        itr = 0
        while itr < self.max_retry:
            self.retries += 1
            try:
                completion = self.client.chat.completions.create(
                    **self._completion_kwargs(messages)
                )
                self.success = True
                break
            except openai.BadRequestError as e:
                # The request itself is wrong -- a parameter this deployment does
                # not take, or a conversation past its context window. Waiting
                # will not change that, so surface it now instead of spending the
                # backoff budget reserved for rate limits and outages.
                raise RetryError(f"{self.model_name} rejected the request: {e}") from e
            except openai.OpenAIError as e:
                itr += 1
                last_error = e
                self.error_types.append(type(e).__name__)
                wait_time = self.min_retry_wait_time * (2 ** (itr - 1))
                logger.warning(
                    f"{type(e).__name__} from {self.model_name} "
                    f"(attempt {itr}/{self.max_retry}), retrying in {wait_time}s: {e}"
                )
                if itr < self.max_retry:
                    time.sleep(wait_time)
        if not completion:
            raise RetryError(
                f"Failed to get a response from the API after {self.max_retry} retries\n"
                f"Last error: {last_error}"
            )
        if completion.choices[0].finish_reason == "length":
            # Worth its own line in the log: a truncated response reaches the
            # parser as malformed JSON, so without this the budget looks like a
            # model that cannot follow the output format. Reasoning models spend
            # `max_completion_tokens` on thinking before emitting anything, so
            # the ceiling has to clear reasoning *plus* the answer.
            logger.warning(
                f"{self.model_name} hit the {self.max_tokens}-token ceiling "
                f"({completion.usage.completion_tokens} used); the response is truncated."
            )
        cost = {
            "input_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
        }
        return completion.choices[0].message, cost

    def get_stats(self):
        return {
            "n_retry_llm": self.retries,
        }


def make_chat_model(config: dict) -> AbstractChatModel:
    """Build the chat model a config block describes.

    Both the agent and the environment's debugger go through here so they cannot
    drift apart. The only thing a config still chooses is which model to name and
    how long its answer may be -- everything else about the call is fixed above,
    so no config can describe a request the deployment will reject.
    """
    return ChatModel(
        model_name=config["model_name"],
        max_tokens=config.get("max_new_tokens") or MAX_COMPLETION_TOKENS,
    )
