from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, cast

import openai
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletionNamedToolChoiceParam

from magentic._parsing import contains_string_type
from magentic.chat_model.base import ChatModel, OutputT, aparse_stream, parse_stream
from magentic.chat_model.function_schema import (
    get_async_function_schemas,
    get_function_schemas,
)
from magentic.chat_model.message import AssistantMessage, Message, Usage, _RawMessage
from magentic.chat_model.openai_chat_model import (
    BaseFunctionToolSchema,
    message_to_openai_message,
)
from magentic.chat_model.stream import (
    AsyncOutputStream,
    FunctionCallChunk,
    OutputStream,
    StreamParser,
    StreamState,
)

try:
    import litellm
    from litellm.litellm_core_utils.streaming_handler import (  # type: ignore[attr-defined]
        StreamingChoices,
    )
    from litellm.types.utils import ModelResponse
except ImportError as error:
    msg = "To use LitellmChatModel you must install the `litellm` package using `pip install 'magentic[litellm]'`."
    raise ImportError(msg) from error


class LitellmStreamParser(StreamParser[ModelResponse]):
    def is_content(self, item: ModelResponse) -> bool:
        pass

    def get_content(self, item: ModelResponse) -> str | None:
        pass

    def is_tool_call(self, item: ModelResponse) -> bool:
        pass

    def iter_tool_calls(self, item: ModelResponse) -> Iterable[FunctionCallChunk]:
        pass


class LitellmStreamState(StreamState[ModelResponse]):
    def __init__(self) -> None:
        self._chat_completion_stream_state = ChatCompletionStreamState(
            input_tools=openai.omit,
            response_format=openai.omit,
        )
        self.usage_ref: list[Usage] = []

    def update(self, item: ModelResponse) -> None:
        # Patch attributes required inside ChatCompletionStreamState.handle_chunk
        pass

    @property
    def current_message_snapshot(self) -> Message[Any]:
        pass


class LitellmChatModel(ChatModel):
    """An LLM chat model that uses the `litellm` python package."""

    def __init__(
        self,
        model: str,
        *,
        api_base: str | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
        temperature: float | None = None,
        custom_llm_provider: str | None = None,
    ):
        self._model = model
        self._api_base = api_base
        self._extra_headers = extra_headers
        self._max_tokens = max_tokens
        self._metadata = metadata
        self._temperature = temperature
        self._custom_llm_provider = custom_llm_provider

    @property
    def model(self) -> str:
        pass

    @property
    def api_base(self) -> str | None:
        pass

    @property
    def extra_headers(self) -> dict[str, str] | None:
        pass

    @property
    def max_tokens(self) -> int | None:
        pass

    @property
    def metadata(self) -> dict[str, Any] | None:
        pass

    @property
    def temperature(self) -> float | None:
        pass

    @property
    def custom_llm_provider(self) -> str | None:
        pass

    @staticmethod
    def _get_tool_choice(
        *,
        tool_schemas: Sequence[BaseFunctionToolSchema[Any]],
        output_types: Iterable[type[OutputT]],
    ) -> ChatCompletionNamedToolChoiceParam | Literal["required"] | None:
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
