import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import openai
from openai import AzureOpenAI, OpenAI

logger = logging.getLogger(__name__)


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
    """Serializable object for instantiating a chat model backed by Azure OpenAI / AI Foundry."""

    deployment_name: str | None = None

    def make_model(self):
        return AzureChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            deployment_name=self.deployment_name,
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
        """Reasoning models reject `temperature` and `max_tokens`."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "n": n_samples,
            "logprobs": self.log_probs,
        }
        if not any(tag in self.model_name.lower() for tag in ("o3", "o4")):
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = self.max_tokens
        return kwargs

    def __call__(self, messages: list[dict], n_samples: int = 1, temperature: float = None) -> dict:
        # Initialize retry tracking attributes
        self.retries = 0
        self.success = False
        self.error_types = []

        completion = None
        last_error = None
        temperature = temperature if temperature is not None else self.temperature
        for itr in range(self.max_retry):
            self.retries += 1
            try:
                completion = self.client.chat.completions.create(
                    **self._completion_kwargs(messages, n_samples, temperature)
                )
                self.success = True
                break
            except openai.OpenAIError as e:
                last_error = e
                self.error_types.append(type(e).__name__)
                wait_time = self.min_retry_wait_time * (2**itr)
                logger.warning(
                    f"{type(e).__name__} from {self.model_name} "
                    f"(attempt {itr + 1}/{self.max_retry}), retrying in {wait_time}s: {e}"
                )
                if itr < self.max_retry - 1:
                    time.sleep(wait_time)
        if not completion:
            raise RetryError(
                f"Failed to get a response from the API after {self.max_retry} retries\n"
                f"Last error: {last_error}"
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
