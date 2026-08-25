import dataclasses
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import openai
from openai import AzureOpenAI, OpenAI

logger = logging.getLogger(__name__)

# Codes the OpenAI-shaped APIs use to say "that parameter is not for this model".
_PARAM_REJECTION_CODES = ("unsupported_parameter", "unsupported_value")
_REPLACEMENT_RE = re.compile(r"use '(\w+)' instead", re.I)


def make_system_message(content: str) -> dict:
    return dict(role="system", content=content)


def make_user_message(content: str) -> dict:
    return dict(role="user", content=content)


def make_assistant_message(content: str) -> dict:
    return dict(role="assistant", content=content)


class RetryError(RuntimeError):
    """Raised when the chat API could not be reached within the retry budget."""


@dataclass
class ParamQuirks:
    """What a deployment has told us it will not accept.

    Deployments disagree about sampling parameters -- the GPT-5 generation
    rejects any `temperature` other than the default and renamed `max_tokens` to
    `max_completion_tokens`. Sniffing model names to guess this ages badly, so we
    send the full set once, read the 400 back, and remember it for the rest of
    the process instead.
    """

    drop: set[str] = field(default_factory=set)
    rename: dict[str, str] = field(default_factory=dict)

    def apply(self, kwargs: dict) -> dict:
        for old, new in self.rename.items():
            if old in kwargs:
                kwargs[new] = kwargs.pop(old)
        for name in self.drop:
            kwargs.pop(name, None)
        return kwargs

    def learn(self, error: openai.BadRequestError) -> bool:
        """Record the rejection. False if it taught us nothing new.

        The caller retries only while this returns True, so every True has to add
        information -- otherwise a deployment that keeps rejecting the same
        parameter would spin forever.
        """
        body = error.body if isinstance(error.body, dict) else {}
        param = body.get("param")
        if body.get("code") not in _PARAM_REJECTION_CODES or not param:
            return False
        replacement = _REPLACEMENT_RE.search(body.get("message") or "")
        if replacement:
            if self.rename.get(param) == replacement.group(1):
                return False
            self.rename[param] = replacement.group(1)
        else:
            if param in self.drop:
                return False
            self.drop.add(param)
        return True


# Keyed by model name so each worker process pays the discovery cost once.
_QUIRKS: dict[str, ParamQuirks] = {}


class AbstractChatModel(ABC):
    @abstractmethod
    def __call__(self, messages: list[dict]) -> dict:
        pass

    def get_stats(self):
        return {}


@dataclass
class BaseModelArgs(ABC):
    """Base class for all model arguments"""

    model_name: str
    max_total_tokens: int | None = None
    max_input_tokens: int | None = None
    max_new_tokens: int | None = None
    temperature: float = 0.6
    vision_support: bool = False
    log_probs: bool = False

    @abstractmethod
    def make_model(self) -> AbstractChatModel:
        pass


@dataclass
class OpenAIModelArgs(BaseModelArgs):
    """Serializable object for instantiating a chat model backed by the OpenAI API."""

    def make_model(self):
        return OpenAIChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            log_probs=self.log_probs,
        )


@dataclass
class AzureModelArgs(BaseModelArgs):
    """Serializable object for instantiating a chat model backed by Azure OpenAI."""

    deployment_name: str | None = None

    def make_model(self):
        return AzureChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            deployment_name=self.deployment_name,
            log_probs=self.log_probs,
        )


@dataclass
class FoundryModelArgs(BaseModelArgs):
    """Serializable object for instantiating a chat model backed by Azure AI Foundry."""

    def make_model(self):
        return FoundryChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            log_probs=self.log_probs,
        )


class ChatModel(AbstractChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=10,
        api_key_env_var=None,
        client_class=OpenAI,
        client_args=None,
        log_probs=False,
    ):
        assert max_retry > 0, "max_retry should be greater than 0"

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry
        self.min_retry_wait_time = min_retry_wait_time
        self.log_probs = log_probs

        # Get the API key from the environment variable if not provided
        if api_key_env_var:
            api_key = api_key or os.getenv(api_key_env_var)
        self.api_key = api_key

        self.client = client_class(api_key=api_key, **(client_args or {}))

    def _completion_kwargs(self, messages, n_samples, temperature):
        return {
            "model": self.model_name,
            "messages": messages,
            "n": n_samples,
            "logprobs": self.log_probs,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }

    def __call__(self, messages: list[dict], n_samples: int = 1, temperature: float = None) -> dict:
        # Initialize retry tracking attributes
        self.retries = 0
        self.success = False
        self.error_types = []

        completion = None
        last_error = None
        temperature = temperature if temperature is not None else self.temperature
        quirks = _QUIRKS.setdefault(self.model_name, ParamQuirks())
        itr = 0
        while itr < self.max_retry:
            self.retries += 1
            try:
                completion = self.client.chat.completions.create(
                    **quirks.apply(self._completion_kwargs(messages, n_samples, temperature))
                )
                self.success = True
                break
            except openai.OpenAIError as e:
                if isinstance(e, openai.BadRequestError) and quirks.learn(e):
                    # We sent a parameter this deployment does not take. That is
                    # our bug, not a service failure, so retry at once and do not
                    # spend the budget reserved for rate limits and outages.
                    logger.info(f"{self.model_name} rejected a parameter, retrying without it: {e}")
                    continue
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
        if n_samples == 1:
            return completion.choices[0].message, cost
        return [c.message for c in completion.choices], cost

    def get_stats(self):
        return {
            "n_retry_llm": self.retries,
        }


class OpenAIChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=10,
        log_probs=False,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var="OPENAI_API_KEY",
            client_class=OpenAI,
            log_probs=log_probs,
        )


class AzureChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        deployment_name=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=10,
        log_probs=False,
    ):
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("API_VERSION")
        assert endpoint, "AZURE_OPENAI_ENDPOINT has to be defined in the environment"

        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            client_class=AzureOpenAI,
            client_args={
                "azure_deployment": deployment_name,
                "azure_endpoint": endpoint,
                "api_version": api_version,
            },
            log_probs=log_probs,
        )


class FoundryChatModel(ChatModel):
    """Azure AI Foundry's `/openai/v1` API.

    It speaks plain OpenAI rather than Azure's `deployments/<name>?api-version=`
    routing, so it takes the vanilla `OpenAI` client with a `base_url` and there
    is no deployment name to configure.
    """

    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=10,
        log_probs=False,
    ):
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        assert endpoint, "AZURE_OPENAI_ENDPOINT has to be defined in the environment"
        # Accept either the base (".../openai/v1") or the full completions URL,
        # which is what the Foundry portal hands you.
        base_url = endpoint.split("/chat/completions")[0].rstrip("/")

        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("AZURE_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var="AZURE_OPENAI_API_KEY",
            client_class=OpenAI,
            client_args={"base_url": base_url},
            log_probs=log_probs,
        )


MODEL_ARGS = {
    "OpenAI": OpenAIModelArgs,
    "Azure": AzureModelArgs,
    "Foundry": FoundryModelArgs,
}


def make_chat_model(config: dict) -> AbstractChatModel:
    """Build the chat model a config block describes.

    Both the agent and the environment's debugger go through here so they cannot
    drift apart. Keys the chosen `*ModelArgs` does not declare are ignored, which
    is what lets one config shape serve all three backends.
    """
    model_type = config.get("model_type")
    if model_type not in MODEL_ARGS:
        raise ValueError(
            f"Model type {model_type!r} not supported. Choose from {sorted(MODEL_ARGS)}."
        )
    args_class = MODEL_ARGS[model_type]
    declared = {f.name for f in dataclasses.fields(args_class)}
    return args_class(**{k: v for k, v in config.items() if k in declared}).make_model()
