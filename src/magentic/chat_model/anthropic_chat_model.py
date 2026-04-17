import json
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
from enum import Enum
from functools import singledispatch
from itertools import groupby
from typing import Any, Generic, cast

from typing_extensions import TypeVar

from magentic._parsing import contains_parallel_function_call_type, contains_string_type
from magentic._streamed_response import AsyncStreamedResponse, StreamedResponse
from magentic.chat_model.base import ChatModel, OutputT, aparse_stream, parse_stream
from magentic.chat_model.function_schema import (
    BaseFunctionSchema,
    FunctionCallFunctionSchema,
    function_schema_for_type,
    get_async_function_schemas,
    get_function_schemas,
)
from magentic.chat_model.message import (
    AssistantMessage,
    DocumentBytes,
    ImageBytes,
    Message,
    SystemMessage,
    ToolResultMessage,
    Usage,
    UserMessage,
    _RawMessage,
)
from magentic.chat_model.stream import (
    AsyncOutputStream,
    FunctionCallChunk,
    OutputStream,
    StreamParser,
    StreamState,
)
from magentic.function_call import (
    AsyncParallelFunctionCall,
    FunctionCall,
    ParallelFunctionCall,
    _create_unique_id,
)
from magentic.streaming import AsyncStreamedStr, StreamedStr
from magentic.vision import UserImageMessage

try:
    import anthropic
    from anthropic.lib.streaming import MessageStreamEvent
    from anthropic.lib.streaming._messages import accumulate_event
    from anthropic.types import (
        DocumentBlockParam,
        ImageBlockParam,
        MessageParam,
        TextBlockParam,
        ToolChoiceParam,
        ToolChoiceToolParam,
        ToolParam,
        ToolUseBlockParam,
    )
except ImportError as error:
    msg = "To use AnthropicChatModel you must install the `anthropic` package using `pip install 'magentic[anthropic]'`."
    raise ImportError(msg) from error


class AnthropicMessageRole(Enum):
    ASSISTANT = "assistant"
    USER = "user"


@singledispatch
def message_to_anthropic_message(message: Message[Any]) -> MessageParam:
    """Convert a Message to an OpenAI message."""
    # TODO: Add instructions for registering new Message type to this error message
    raise NotImplementedError(type(message))


@singledispatch
async def async_message_to_anthropic_message(message: Message[Any]) -> MessageParam:
    """Async version of `message_to_anthropic_message`."""
    pass


@message_to_anthropic_message.register(_RawMessage)
def _(message: _RawMessage[Any]) -> MessageParam:
    # TODO: Validate the message content
    pass


@message_to_anthropic_message.register(UserMessage)
def _(message: UserMessage[Any]) -> MessageParam:
    pass


@message_to_anthropic_message.register(UserImageMessage)
def _(message: UserImageMessage[Any]) -> MessageParam:
    pass


def _function_call_to_tool_call_block(
    function_call: FunctionCall[Any],
) -> ToolUseBlockParam:
    pass


@message_to_anthropic_message.register(AssistantMessage)
def _(message: AssistantMessage[Any]) -> MessageParam:
    pass


@async_message_to_anthropic_message.register(AssistantMessage)
async def _(message: AssistantMessage[Any]) -> MessageParam:
    pass


@message_to_anthropic_message.register(ToolResultMessage)
def _(message: ToolResultMessage[Any]) -> MessageParam:
    pass


# TODO: Move this to the magentic level by allowing `UserMessage` have a list of content
def _combine_messages(messages: Iterable[MessageParam]) -> list[MessageParam]:
    """Combine messages with the same role, to get alternating roles.

    Alternating roles is a requirement of the Anthropic API.
    """
    pass


T = TypeVar("T")
BaseFunctionSchemaT = TypeVar("BaseFunctionSchemaT", bound=BaseFunctionSchema[Any])


class BaseFunctionToolSchema(Generic[BaseFunctionSchemaT]):
    def __init__(self, function_schema: BaseFunctionSchemaT):
        self._function_schema = function_schema

    def to_dict(self) -> ToolParam:
        pass

    def as_tool_choice(self, *, disable_parallel_tool_use: bool) -> ToolChoiceToolParam:
        pass


class AnthropicStreamParser(StreamParser[MessageStreamEvent]):
    def is_content(self, item: MessageStreamEvent) -> bool:
        pass

    def get_content(self, item: MessageStreamEvent) -> str | None:
        pass

    def is_tool_call(self, item: MessageStreamEvent) -> bool:
        pass

    def iter_tool_calls(self, item: MessageStreamEvent) -> Iterable[FunctionCallChunk]:
        pass


class AnthropicStreamState(StreamState[MessageStreamEvent]):
    def __init__(self) -> None:
        self._current_message_snapshot: anthropic.types.Message | None = (
            None  # TODO: type
        )
        self.usage_ref: list[Usage] = []

    def update(self, item: MessageStreamEvent) -> None:
        pass

    @property
    def current_message_snapshot(self) -> Message[Any]:
        pass


def _extract_system_message(
    messages: Iterable[Message[Any]],
) -> tuple[str | anthropic.NotGiven, list[Message[Any]]]:
    pass


def _if_given(value: T | None) -> T | anthropic.NotGiven:
    pass


class AnthropicChatModel(ChatModel):
    """An LLM chat model that uses the `anthropic` python package."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._temperature = temperature

        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        self._async_client = anthropic.AsyncAnthropic(
            api_key=api_key, base_url=base_url
        )

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
    def max_tokens(self) -> int:
        pass

    @property
    def temperature(self) -> float | None:
        pass

    @staticmethod
    def _get_tool_choice(
        *,
        tool_schemas: Sequence[BaseFunctionToolSchema[Any]],
        output_types: Iterable[type],
    ) -> ToolChoiceParam | anthropic.NotGiven:
        """Create the tool choice argument."""
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
