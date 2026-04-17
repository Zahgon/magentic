import os
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
from typing import Any, Literal, cast

import openai
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionStreamOptionsParam,
)

from magentic._parsing import contains_parallel_function_call_type, contains_string_type
from magentic.chat_model.base import ChatModel, OutputT, aparse_stream, parse_stream
from magentic.chat_model.function_schema import (
    get_async_function_schemas,
    get_function_schemas,
)
from magentic.chat_model.message import AssistantMessage, ContentT, Message, Usage
from magentic.chat_model.openai_chat_model import (
    BaseFunctionToolSchema,
    OpenaiChatModel,
    OpenaiStreamParser,
    OpenaiStreamState,
    _add_missing_tool_calls_responses,
    _if_given,
    async_message_to_openai_message,
    message_to_openai_message,
)
from magentic.chat_model.stream import AsyncOutputStream, OutputStream


class OpenRouterStreamState(OpenaiStreamState):
    """State for OpenRouter stream parsing."""

    reasoning: str

    def __init__(self) -> None:
        super().__init__()
        self.reasoning = ""

    def update(self, item: ChatCompletionChunk) -> None:
        pass


class OpenRouterAssistantMessage(AssistantMessage[ContentT]):
    """An assistant message from OpenRouter that includes reasoning tokens."""

    reasoning: str = ""

    def __init__(self, content: ContentT, reasoning: str = "", **data: Any):
        super().__init__(content=content, **data)
        self.reasoning = reasoning

    @classmethod
    def _with_usage(
        cls,
        content: ContentT,  # type: ignore[misc]
        usage_ref: list[Usage],
        reasoning: str = "",
    ) -> "OpenRouterAssistantMessage[ContentT]":
        """Create a message with usage statistics."""
        pass


class _OpenRouterOpenaiChatModel(OpenaiChatModel):
    """Modified OpenaiChatModel to be compatible with OpenRouter API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = "https://openrouter.ai/api/v1",
        max_tokens: int | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        # Routing options
        route: Literal["fallback"] | None = None,
        models: list[str] | None = None,
        # Reasoner model options
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        reasoning_exclude: bool | None = None,
        # Provider options
        provider_order: list[str] | None = None,
        allow_fallbacks: bool | None = None,
        data_collection: Literal["allow", "deny"] | None = None,
        provider_only: list[str] | None = None,
        provider_ignore: list[str] | None = None,
        quantizations: list[str] | None = None,
        provider_sort: Literal["price", "throughput", "latency"] | None = None,
        max_price: dict[str, float] | None = None,
    ):
        super().__init__(
            model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            seed=seed,
            temperature=temperature,
        )
        self._route = route
        self._models = models
        self._reasoning_effort = reasoning_effort
        self._reasoning_exclude = reasoning_exclude
        # Provider options
        self._provider_order = provider_order
        self._allow_fallbacks = allow_fallbacks
        self._data_collection = data_collection
        self._provider_only = provider_only
        self._provider_ignore = provider_ignore
        self._quantizations = quantizations
        self._provider_sort = provider_sort
        self._max_price = max_price

    def _get_stream_options(self) -> ChatCompletionStreamOptionsParam | openai.Omit:
        pass

    @staticmethod
    def _get_tool_choice(
        *,
        tool_schemas: Sequence[BaseFunctionToolSchema[Any]],
        output_types: Iterable[type],
    ) -> (
        Literal["none", "auto", "required"]
        | openai.Omit
        | ChatCompletionNamedToolChoiceParam
    ):
        pass

    def _get_parallel_tool_calls(
        self, *, tools_specified: bool, output_types: Iterable[type]
    ) -> bool | openai.Omit:
        pass

    def _get_extra_body(self) -> dict[str, Any] | None:
        """Get extra body parameters for OpenRouter API."""
        pass

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> OpenRouterAssistantMessage[OutputT]:
        """Request an LLM message."""
        pass

    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> OpenRouterAssistantMessage[OutputT]:
        """Async version of `complete`."""
        pass


class OpenRouterChatModel(ChatModel):
    """An LLM chat model that uses OpenRouter's API via the `openai` python package."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = "https://openrouter.ai/api/v1",
        max_tokens: int | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        # Routing options
        route: Literal["fallback"] | None = None,
        models: list[str] | None = None,
        # Reasoner model options
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        reasoning_exclude: bool | None = None,
        # Provider options
        provider_order: list[str] | None = None,
        allow_fallbacks: bool | None = None,
        data_collection: Literal["allow", "deny"] | None = None,
        provider_only: list[str] | None = None,
        provider_ignore: list[str] | None = None,
        quantizations: list[str] | None = None,
        provider_sort: Literal["price", "throughput", "latency"] | None = None,
        max_price: dict[str, float] | None = None,
    ):
        if not (api_key or os.getenv("OPENROUTER_API_KEY")):
            exception_string = "OPENROUTER_API_KEY variable or api_key required."
            raise openai.OpenAIError(exception_string)
        self._openrouter_openai_chat_model = _OpenRouterOpenaiChatModel(
            model,
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=base_url,
            max_tokens=max_tokens,
            seed=seed,
            temperature=temperature,
            route=route,
            models=models,
            reasoning_effort=reasoning_effort,
            reasoning_exclude=reasoning_exclude,
            provider_order=provider_order,
            allow_fallbacks=allow_fallbacks,
            data_collection=data_collection,
            provider_only=provider_only,
            provider_ignore=provider_ignore,
            quantizations=quantizations,
            provider_sort=provider_sort,
            max_price=max_price,
        )

    def _get_extra_body(self) -> dict[str, Any] | None:
        """Get extra body parameters for OpenRouter API."""
        pass

    @property
    def model(self) -> str:
        pass

    @property
    def api_key(self) -> str | None:
        pass

    @property
    def base_url(self) -> str | None:
        pass

    @property
    def max_tokens(self) -> int | None:
        pass

    @property
    def seed(self) -> int | None:
        pass

    @property
    def temperature(self) -> float | None:
        pass

    @property
    def route(self) -> Literal["fallback"] | None:
        pass

    @property
    def models(self) -> list[str] | None:
        pass

    @property
    def reasoning(self) -> dict[str, Any] | None:
        pass

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> OpenRouterAssistantMessage[OutputT]:
        """Request an LLM message."""
        pass

    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> OpenRouterAssistantMessage[OutputT]:
        """Async version of `complete`."""
        pass
