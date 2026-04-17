import inspect
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ParamSpec

from typing_extensions import Self, deprecated

from magentic.backend import get_chat_model
from magentic.chat_model.base import ChatModel
from magentic.chat_model.message import (
    AssistantMessage,
    FunctionResultMessage,
    Message,
    SystemMessage,
    UserMessage,
    UserMessageContentBlock,
)
from magentic.function_call import (
    AsyncParallelFunctionCall,
    FunctionCall,
    ParallelFunctionCall,
)
from magentic.prompt_function import BasePromptFunction
from magentic.streaming import async_iter, azip

P = ParamSpec("P")


class Chat:
    """A chat with an LLM chat model.

    Examples
    --------
    >>> chat = Chat().add_user_message("Hello")
    >>> chat.messages
    [UserMessage('Hello')]
    >>> chat = chat.submit()
    >>> chat.messages
    [UserMessage('Hello'), AssistantMessage('Hello! How can I assist you today?')]
    """

    def __init__(
        self,
        messages: Sequence[Message[Any]] | None = None,
        *,
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[Any]] | None = None,
        model: ChatModel | None = None,
    ):
        self._messages = list(messages) if messages else []
        self._functions = list(functions) if functions else []
        self._output_types = list(output_types) if output_types else [str]
        self._model = model

    @classmethod
    @deprecated(
        "Chat.from_prompt will be removed in a future version."
        " Instead, use the regular init method, `Chat(messages, functions, output_types, model)`."
    )
    def from_prompt(
        cls: type[Self],
        prompt: BasePromptFunction[P, Any],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        """Create a chat from a prompt function."""
        pass

    @property
    def messages(self) -> list[Message[Any]]:
        pass

    @property
    def last_message(self) -> Message[Any]:
        pass

    @property
    def model(self) -> ChatModel:
        pass

    def add_message(self, message: Message[Any]) -> Self:
        """Add a message to the chat."""
        pass

    def add_system_message(self, content: str) -> Self:
        """Add a system message to the chat."""
        pass

    def add_user_message(
        self, content: str | Sequence[str | UserMessageContentBlock]
    ) -> Self:
        """Add a user message to the chat."""
        pass

    def add_assistant_message(self, content: Any) -> Self:
        """Add an assistant message to the chat."""
        pass

    # TODO: Allow restricting functions and/or output types here
    def submit(self) -> Self:
        """Request an LLM message to be added to the chat."""
        pass

    async def asubmit(self) -> Self:
        """Async version of `submit`."""
        pass

    # TODO: Add optional error handling to this method, with param to toggle
    def exec_function_call(self) -> Self:
        """If the last message is a function call, execute it and add the result."""
        pass

    async def aexec_function_call(self) -> Self:
        """Async version of `exec_function_call`."""
        pass
