from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
from enum import Enum
from functools import singledispatch
from typing import Any, Generic, Literal, TypeVar, cast

import openai
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)

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
    ImageBytes,
    ImageUrl,
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


class OpenaiMessageRole(Enum):
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    USER = "user"


@singledispatch
def message_to_openai_message(message: Message[Any]) -> ChatCompletionMessageParam:
    """Convert a Message to an OpenAI message."""
    # TODO: Add instructions for registering new Message type to this error message
    raise NotImplementedError(type(message))


@singledispatch
async def async_message_to_openai_message(
    message: Message[Any],
) -> ChatCompletionMessageParam:
    """Async version of `message_to_openai_message`."""
    pass


@message_to_openai_message.register(_RawMessage)
def _(message: _RawMessage[Any]) -> ChatCompletionMessageParam:
    pass


@message_to_openai_message.register
def _(message: SystemMessage) -> ChatCompletionMessageParam:
    pass


@message_to_openai_message.register(UserMessage)
def _(message: UserMessage[Any]) -> ChatCompletionUserMessageParam:
    pass


@message_to_openai_message.register(UserImageMessage)
def _(message: UserImageMessage[Any]) -> ChatCompletionUserMessageParam:
    pass


def _function_call_to_tool_call_block(
    function_call: FunctionCall[Any],
) -> ChatCompletionMessageToolCallParam:
    pass


@message_to_openai_message.register(AssistantMessage)
def _(message: AssistantMessage[Any]) -> ChatCompletionMessageParam:
    pass


@async_message_to_openai_message.register(AssistantMessage)
async def _(message: AssistantMessage[Any]) -> ChatCompletionMessageParam:
    pass


@message_to_openai_message.register(ToolResultMessage)
def _(message: ToolResultMessage[Any]) -> ChatCompletionMessageParam:
    pass


# TODO: Use ToolResultMessage to solve this at magentic level
def _add_missing_tool_calls_responses(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    """Add null responses for tool calls without a response.

    This is required by OpenAI's API.
    "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'."
    """
    pass


T = TypeVar("T")
BaseFunctionSchemaT = TypeVar("BaseFunctionSchemaT", bound=BaseFunctionSchema[Any])


class BaseFunctionToolSchema(Generic[BaseFunctionSchemaT]):
    def __init__(self, function_schema: BaseFunctionSchemaT):
        self._function_schema = function_schema

    def as_tool_choice(self) -> ChatCompletionNamedToolChoiceParam:
        pass

    def to_dict(self) -> ChatCompletionToolParam:
        pass


class OpenaiStreamParser(StreamParser[ChatCompletionChunk]):
    def is_content(self, item: ChatCompletionChunk) -> bool:
        pass

    def get_content(self, item: ChatCompletionChunk) -> str | None:
        pass

    def is_tool_call(self, item: ChatCompletionChunk) -> bool:
        pass

    def iter_tool_calls(self, item: ChatCompletionChunk) -> Iterator[FunctionCallChunk]:
        pass


class OpenaiStreamState(StreamState[ChatCompletionChunk]):
    """Tracks the state of the OpenAI model output stream.

    - message snapshot
    - usage
    - stop reason
    """

    def __init__(self) -> None:
        self._chat_completion_stream_state = ChatCompletionStreamState(
            input_tools=openai.omit,
            response_format=openai.omit,
        )
        self.usage_ref: list[Usage] = []

        # Keep track of tool call index to add this to Mistral tool calls
        self._current_tool_call_index: int = -1
        self._seen_tool_call_ids: set[str] = set()

    def update(self, item: ChatCompletionChunk) -> None:
        # Add tool call index for Mistral tool calls to make compatible with OpenAI
        # TODO: Remove this fix when MistralChatModel switched to mistral python package
        pass

    @property
    def current_message_snapshot(self) -> Message[Any]:
        pass


def _if_given(value: T | None) -> T | openai.Omit:
    pass


class OpenaiChatModel(ChatModel):
    """An LLM chat model that uses the `openai` python package."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_type: Literal["openai", "azure"] = "openai",
        base_url: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        verbosity: Literal["low", "medium", "high"] | None = None,
    ):
        self._model = model
        self._api_key = api_key
        self._api_type = api_type
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._max_completion_tokens = max_completion_tokens
        self._seed = seed
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._verbosity = verbosity

        match api_type:
            case "openai":
                self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
                self._async_client = openai.AsyncOpenAI(
                    api_key=api_key, base_url=base_url
                )
            case "azure":
                self._client = openai.AzureOpenAI(
                    api_key=api_key,
                    base_url=base_url,  # type: ignore[arg-type]
                )
                self._async_client = openai.AsyncAzureOpenAI(
                    api_key=api_key,
                    base_url=base_url,  # type: ignore[arg-type]
                )

    @property
    def model(self) -> str:
        pass

    @property
    def api_key(self) -> str | None:
        pass

    @property
    def api_type(self) -> Literal["openai", "azure"]:
        pass

    @property
    def base_url(self) -> str | None:
        pass

    @property
    def max_tokens(self) -> int | None:
        pass

    @property
    def max_completion_tokens(self) -> int | None:
        pass

    @property
    def seed(self) -> int | None:
        pass

    @property
    def temperature(self) -> float | None:
        pass

    @property
    def reasoning_effort(self) -> Literal["low", "medium", "high"] | None:
        pass

    @property
    def verbosity(self) -> Literal["low", "medium", "high"] | None:
        pass

    def _get_stream_options(self) -> ChatCompletionStreamOptionsParam | openai.Omit:
        pass

    @staticmethod
    def _get_tool_choice(
        *,
        tool_schemas: Sequence[BaseFunctionToolSchema[Any]],
        output_types: Iterable[type],
    ) -> ChatCompletionToolChoiceOptionParam | openai.Omit:
        """Create the tool choice argument."""
        pass

    def _get_parallel_tool_calls(
        self, *, tools_specified: bool, output_types: Iterable[type]
    ) -> bool | openai.Omit:
        pass

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
        # TODO: Add type hint for function call ?
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
