import inspect
from collections.abc import Awaitable, Callable, Sequence
from functools import update_wrapper
from typing import Any, Generic, ParamSpec, Protocol, TypeVar, cast, overload

from magentic.backend import get_chat_model
from magentic.chat_model.base import ChatModel
from magentic.chat_model.message import Message
from magentic.chat_model.retry_chat_model import RetryChatModel
from magentic.logger import logfire
from magentic.typing import split_union_type

P = ParamSpec("P")
# TODO: Make `R` type Union of all possible return types except FunctionCall ?
# Then `R | FunctionCall[FuncR]` will separate FunctionCall from other return types.
# Can then use `FuncR` to make typechecker check `functions` argument to `chatprompt`
# `Not` type would solve this - https://github.com/python/typing/issues/801
R = TypeVar("R")


def escape_braces(text: str) -> str:
    """Escape curly braces in a string.

    This allows curly braces to be used in a string template without being interpreted
    as format specifiers.
    """
    pass


class BaseChatPromptFunction(Generic[P, R]):
    """Base class for an LLM chat prompt template that is directly callable to query the LLM."""

    def __init__(
        self,
        name: str,
        parameters: Sequence[inspect.Parameter],
        return_type: type[R],
        messages: Sequence[Message[Any]],
        functions: list[Callable[..., Any]] | None = None,
        stop: list[str] | None = None,
        max_retries: int = 0,
        model: ChatModel | None = None,
    ):
        self._name = name
        self._signature = inspect.Signature(
            parameters=parameters,
            return_annotation=return_type,
        )
        self._messages = messages
        self._functions = functions or []
        self._stop = stop
        self._max_retries = max_retries
        self._model = model

        self._return_types = list(split_union_type(return_type))

    @property
    def functions(self) -> list[Callable[..., Any]]:
        pass

    @property
    def model(self) -> ChatModel:
        pass

    @property
    def return_types(self) -> list[type[R]]:
        pass

    def format(self, *args: P.args, **kwargs: P.kwargs) -> list[Message[Any]]:
        """Format the message templates with the given arguments."""
        pass


class ChatPromptFunction(BaseChatPromptFunction[P, R], Generic[P, R]):
    """An LLM chat prompt template that is directly callable to query the LLM."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Query the LLM with the formatted chat prompt template."""
        with logfire.span(
            f"Calling chatprompt-function {self._name}",
            **self._signature.bind(*args, **kwargs).arguments,
        ):
            message = self.model.complete(
                messages=self.format(*args, **kwargs),
                functions=self._functions,
                output_types=self._return_types,
                stop=self._stop,
            )
            return message.content


class AsyncChatPromptFunction(BaseChatPromptFunction[P, R], Generic[P, R]):
    """Async version of `ChatPromptFunction`."""

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Asynchronously query the LLM with the formatted chat prompt template."""
        with logfire.span(
            f"Calling async chatprompt-function {self._name}",
            **self._signature.bind(*args, **kwargs).arguments,
        ):
            message = await self.model.acomplete(
                messages=self.format(*args, **kwargs),
                functions=self._functions,
                output_types=self._return_types,
                stop=self._stop,
            )
            return message.content


class ChatPromptDecorator(Protocol):
    """Protocol for a decorator that returns a `ChatPromptFunction`.

    This allows finer-grain type annotation of the `chatprompt` function
    See https://github.com/microsoft/pyright/issues/5014#issuecomment-1523778421
    """

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, func: Callable[P, Awaitable[R]]
    ) -> AsyncChatPromptFunction[P, R]: ...

    @overload
    def __call__(self, func: Callable[P, R]) -> ChatPromptFunction[P, R]: ...


def chatprompt(
    *messages: Message[Any],
    functions: list[Callable[..., Any]] | None = None,
    stop: list[str] | None = None,
    max_retries: int = 0,
    model: ChatModel | None = None,
) -> ChatPromptDecorator:
    """Convert a function into an LLM chat prompt template.

    The `@chatprompt` decorator allows you to define a prompt template for a chat-based Large Language Model (LLM).

    Examples
    --------
    >>> from magentic import chatprompt, AssistantMessage, SystemMessage, UserMessage
    >>>
    >>> from pydantic import BaseModel
    >>>
    >>>
    >>> class Quote(BaseModel):
    >>>     quote: str
    >>>     character: str
    >>>
    >>>
    >>> @chatprompt(
    >>>     SystemMessage("You are a movie buff."),
    >>>     UserMessage("What is your favorite quote from Harry Potter?"),
    >>>     AssistantMessage(
    >>>         Quote(
    >>>             quote="It does not do to dwell on dreams and forget to live.",
    >>>             character="Albus Dumbledore",
    >>>         )
    >>>     ),
    >>>     UserMessage("What is your favorite quote from {movie}?"),
    >>> )
    >>> def get_movie_quote(movie: str) -> Quote: ...
    >>>
    >>>
    >>> get_movie_quote("Iron Man")
    Quote(quote='I am Iron Man.', character='Tony Stark')
    """
    pass
