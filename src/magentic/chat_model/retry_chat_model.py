from collections.abc import Callable, Iterable
from functools import singledispatchmethod
from typing import Any

from magentic.chat_model.base import ChatModel, OutputT, ToolSchemaParseError
from magentic.chat_model.message import AssistantMessage, Message, ToolResultMessage
from magentic.logger import logfire


class RetryChatModel(ChatModel):
    """Wraps another ChatModel to add LLM-assisted retries."""

    def __init__(
        self,
        chat_model: ChatModel,
        *,
        max_retries: int,
    ):
        self._chat_model = chat_model
        self._max_retries = max_retries

    # TODO: Make this public to allow modifying error handling behavior
    # User should be able to add handlers to instance using decorator
    # e.g. `@my_retry_chat_model.exception_handler(exc_type)`
    # TODO: Add exception base class for those with output_message attribute
    @singledispatchmethod
    def _make_retry_messages(self, error: Exception) -> list[Message[Any]]:
        raise NotImplementedError

    # TODO: Catch UnknownToolError here
    @_make_retry_messages.register
    def _(self, error: ToolSchemaParseError) -> list[Message[Any]]:
        pass

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        """Request an LLM message."""
        pass

    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        """Async version of `complete`."""
        pass
