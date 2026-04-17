import inspect
import typing
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Callable, Iterable
from functools import singledispatch
from typing import Any, Generic, TypeVar, cast, get_args, get_origin

from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel, TypeAdapter, create_model

from magentic._pydantic import ConfigDict, get_pydantic_config, json_schema
from magentic._streamed_response import AsyncStreamedResponse, StreamedResponse
from magentic.function_call import (
    AsyncParallelFunctionCall,
    FunctionCall,
    ParallelFunctionCall,
)
from magentic.streaming import (
    AsyncStreamedStr,
    StreamedStr,
    aiter_streamed_json_array,
    iter_streamed_json_array,
)
from magentic.typing import is_origin_abstract, is_origin_subclass, name_type

T = TypeVar("T")


class BaseFunctionSchema(ABC, Generic[T]):
    """Converts a Python object to the JSON Schema that represents it as a function for the LLM."""

    # Allow any arguments to avoid error passing type to subclasses without __init__
    def __init__(self, *args: Any, **kwargs: Any): ...

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the function.

        Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
        """
        ...

    @property
    def description(self) -> str | None:
        """A description of what the function does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """The parameters the functions accepts as a JSON Schema object."""
        ...

    @property
    def strict(self) -> bool | None:
        """Whether to enable strict schema adherence when generating the function call."""
        pass

    def dict(self) -> FunctionDefinition:
        pass


BaseFunctionSchemaT = TypeVar("BaseFunctionSchemaT", bound=BaseFunctionSchema[Any])


def select_function_schema(
    function_schemas: Iterable[BaseFunctionSchemaT], name: str
) -> BaseFunctionSchemaT | None:
    """Select the function schema with the given name."""
    pass


class AsyncFunctionSchema(BaseFunctionSchema[T], Generic[T]):
    @abstractmethod
    async def aparse_args(self, chunks: AsyncIterable[str]) -> T:
        """Parse an async iterable of string chunks into the function arguments."""
        ...

    @abstractmethod
    async def aserialize_args(self, value: T) -> str:
        """Serialize the function arguments into a JSON string."""
        ...


class FunctionSchema(AsyncFunctionSchema[T], Generic[T]):
    @abstractmethod
    def parse_args(self, chunks: Iterable[str]) -> T:
        """Parse an iterable of string chunks into the function arguments."""
        ...

    @abstractmethod
    def serialize_args(self, value: T) -> str:
        """Serialize the function arguments into a JSON string."""
        ...

    async def aparse_args(self, chunks: AsyncIterable[str]) -> T:
        """Parse an async iterable of string chunks into the function arguments."""
        pass

    async def aserialize_args(self, value: T) -> str:
        """Serialize the function arguments into a JSON string."""
        pass


# Use the singledispatch registry to map classes to FunctionSchemas
# because this handles subclass resolution for us.
@singledispatch
def _async_function_schema_registry(type_: type[T]) -> AsyncFunctionSchema[T]:
    pass


def async_function_schema_for_type(type_: type[T]) -> AsyncFunctionSchema[T]:
    """Create a FunctionSchema for the given type."""
    pass


@singledispatch
def _function_schema_registry(type_: type[T]) -> FunctionSchema[T]:
    pass


def function_schema_for_type(type_: type[T]) -> FunctionSchema[T]:
    """Create a FunctionSchema for the given type."""
    pass


TypeFunctionSchemaT = TypeVar(
    "TypeFunctionSchemaT", bound=type[BaseFunctionSchema[Any]]
)


def register_function_schema(
    type_: type[Any],
) -> Callable[[TypeFunctionSchemaT], TypeFunctionSchemaT]:
    """Register a new FunctionSchema for the given type."""

    def _register(cls: TypeFunctionSchemaT) -> TypeFunctionSchemaT:
        pass

    return _register


@register_function_schema(object)
class AnyFunctionSchema(FunctionSchema[T], Generic[T]):
    """The most generic FunctionSchema that should work for most types supported by pydantic."""

    def __init__(self, output_type: type[T]):
        self._output_type = output_type
        self._model = create_model(
            "Output",
            __config__=get_pydantic_config(output_type),
            value=(output_type, ...),
        )

    @property
    def name(self) -> str:
        pass

    @property
    def parameters(self) -> dict[str, Any]:
        pass

    @property
    def strict(self) -> bool | None:
        pass

    def parse_args(self, chunks: Iterable[str]) -> T:
        pass

    def serialize_args(self, value: T) -> str:
        pass


IterableT = TypeVar("IterableT", bound=Iterable[Any])


@register_function_schema(Iterable)
class IterableFunctionSchema(FunctionSchema[IterableT], Generic[IterableT]):
    """FunctionSchema for types that are iterable. Can parse LLM output as a stream."""

    def __init__(self, output_type: type[IterableT]):
        self._output_type = output_type
        self._item_type_adapter: TypeAdapter[Any] = TypeAdapter(
            args[0] if (args := get_args(output_type)) else Any
        )
        self._model = create_model(
            "Output",
            __config__=get_pydantic_config(output_type),
            value=(output_type, ...),
        )

    @property
    def name(self) -> str:
        pass

    @property
    def parameters(self) -> dict[str, Any]:
        pass

    @property
    def strict(self) -> bool | None:
        pass

    def parse_args(self, chunks: Iterable[str]) -> IterableT:
        pass

    def serialize_args(self, value: IterableT) -> str:
        pass


AsyncIterableT = TypeVar("AsyncIterableT", bound=AsyncIterable[Any])


@register_function_schema(AsyncIterable)
class AsyncIterableFunctionSchema(
    AsyncFunctionSchema[AsyncIterableT], Generic[AsyncIterableT]
):
    """FunctionSchema for types that are async iterable. Can parse LLM output as a stream."""

    def __init__(self, output_type: type[AsyncIterableT]):
        self._output_type = output_type
        item_type: type | Any = args[0] if (args := get_args(output_type)) else Any
        self._item_type_adapter = TypeAdapter(item_type)
        self._model = create_model(
            "Output",
            __config__=get_pydantic_config(output_type),
            # Convert to list so pydantic can handle for schema generation
            value=(list[item_type], ...),  # type: ignore[valid-type]
        )

    @property
    def name(self) -> str:
        pass

    @property
    def parameters(self) -> dict[str, Any]:
        pass

    @property
    def strict(self) -> bool | None:
        pass

    async def aparse_args(self, chunks: AsyncIterable[str]) -> AsyncIterableT:
        pass

    async def aserialize_args(self, value: AsyncIterableT) -> str:
        pass


@register_function_schema(dict)
class DictFunctionSchema(FunctionSchema[T], Generic[T]):
    """FunctionSchema for dict."""

    def __init__(self, output_type: type[T]):
        self._output_type = output_type
        self._type_adapter = TypeAdapter(
            output_type, config=get_pydantic_config(output_type)
        )

    @property
    def name(self) -> str:
        pass

    @property
    def parameters(self) -> dict[str, Any]:
        pass

    def parse_args(self, chunks: Iterable[str]) -> T:
        pass

    def serialize_args(self, value: T) -> str:
        pass


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


@register_function_schema(BaseModel)
class BaseModelFunctionSchema(FunctionSchema[BaseModelT], Generic[BaseModelT]):
    """FunctionSchema for pydantic BaseModel."""

    def __init__(self, model: type[BaseModelT]):
        self._model = model

    @property
    def name(self) -> str:
        pass

    @property
    def parameters(self) -> dict[str, Any]:
        pass

    @property
    def strict(self) -> bool | None:
        pass

    def parse_args(self, chunks: Iterable[str]) -> BaseModelT:
        pass

    def serialize_args(self, value: BaseModelT) -> str:
        pass


def create_model_from_function(func: Callable[..., Any]) -> type[BaseModel]:
    """Create a Pydantic model from a function signature."""
    pass


class FunctionCallFunctionSchema(FunctionSchema[FunctionCall[T]], Generic[T]):
    """FunctionSchema for FunctionCall."""

    def __init__(self, func: Callable[..., T]):
        self._func = func
        self._model = create_model_from_function(func)

    @property
    def name(self) -> str:
        pass

    @property
    def description(self) -> str | None:
        pass

    @property
    def parameters(self) -> dict[str, Any]:
        pass

    @property
    def strict(self) -> bool | None:
        pass

    def parse_args(self, chunks: Iterable[str]) -> FunctionCall[T]:
        # Anthropic message stream returns empty string for function call with no arguments
        pass

    def serialize_args(self, value: FunctionCall[T]) -> str:
        pass


R = TypeVar("R")

_NON_FUNCTION_CALL_TYPES = (
    str,
    StreamedStr,
    AsyncStreamedStr,
    FunctionCall,
    ParallelFunctionCall,
    AsyncParallelFunctionCall,
    StreamedResponse,
    AsyncStreamedResponse,
)


def get_function_schemas(
    functions: Iterable[Callable[..., R]] | None,
    output_types: Iterable[type[T]],
) -> Iterable[FunctionSchema[FunctionCall[R] | T]]:
    pass


def get_async_function_schemas(
    functions: Iterable[Callable[..., R]] | None,
    output_types: Iterable[type[T]],
) -> Iterable[FunctionSchema[FunctionCall[R] | T]]:
    pass
