import inspect
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

from magentic._chat import Chat
from magentic.chat_model.base import ChatModel
from magentic.chat_model.message import Message, UserMessage
from magentic.chatprompt import AsyncChatPromptFunction, ChatPromptFunction
from magentic.function_call import FunctionCall
from magentic.logger import logfire

P = ParamSpec("P")
R = TypeVar("R")


class MaxFunctionCallsError(Exception):
    """Raised when prompt chain reaches the max number of function calls."""


def prompt_chain(
    template: str | Sequence[Message[Any]],
    functions: list[Callable[..., Any]] | None = None,
    model: ChatModel | None = None,
    max_calls: int | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Convert a Python function to an LLM query, auto-resolving function calls.

    Use `@prompt_chain` when you need the LLM to perform multiple function calls to
    reach a final answer. When a function decorated with `@prompt_chain` is called, the
    LLM is queried, then any function calls are automatically executed and the results
    appended to the list of messages. Then the LLM is queried again and this repeats
    until a final answer is reached.

    Set `max_calls` to limit the number of function calls. If the limit is reached, a
    `MaxFunctionCallsError` will be raised.
    """
    pass
