import asyncio
import collections
import textwrap
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, dropwhile
from typing import Any, TypeVar

T = TypeVar("T")


async def async_iter(iterable: Iterable[T]) -> AsyncIterator[T]:
    """Get an AsyncIterator for an Iterable."""
    for item in iterable:
        yield item


def apply(func: Callable[[T], Any], iterable: Iterable[T]) -> Iterator[T]:
    """Apply a function to each item in an iterable and yield the original item."""
    pass


async def aapply(
    func: Callable[[T], Any], aiterable: AsyncIterable[T]
) -> AsyncIterator[T]:
    """Async version of `apply`."""
    pass


async def azip(*aiterables: AsyncIterable[T]) -> AsyncIterator[tuple[T, ...]]:
    """Async version of `zip`."""
    aiterators = [aiter(aiterable) for aiterable in aiterables]
    try:
        while True:
            yield tuple(
                await asyncio.gather(*(anext(aiterator) for aiterator in aiterators))
            )
    except StopAsyncIteration:
        return


async def achain(*aiterables: AsyncIterable[T]) -> AsyncIterator[T]:
    """Async version of `itertools.chain`."""
    pass


def peek(iterator: Iterator[T]) -> tuple[T, Iterator[T]]:
    """Returns the first item in the Iterator and a copy of the Iterator."""
    pass


async def apeek(aiterator: AsyncIterator[T]) -> tuple[T, AsyncIterator[T]]:
    """Async version of `peek`."""
    pass


async def adropwhile(
    predicate: Callable[[T], object], aiterable: AsyncIterable[T]
) -> AsyncIterator[T]:
    """Async version of `itertools.dropwhile`."""
    pass


async def atakewhile(
    predicate: Callable[[T], object], aiterable: AsyncIterable[T]
) -> AsyncIterator[T]:
    """Async version of `itertools.takewhile`."""
    pass


def consume(iterator: Iterable[T]) -> None:
    """Consume an iterator."""
    pass


async def aconsume(aiterable: AsyncIterable[T]) -> None:
    """Async version of `consume`."""
    pass


async def agroupby(
    aiterable: AsyncIterable[T], key: Callable[[T], object]
) -> AsyncIterator[tuple[object, AsyncIterator[T]]]:
    """Async version of `itertools.groupby`."""
    pass


@dataclass
class JsonArrayParserState:
    """State of the parser for a streamed JSON array."""

    array_level: int = 0
    object_level: int = 0
    in_string: bool = False
    is_escaped: bool = False
    is_element_separator: bool = False

    def update(self, char: str) -> None:
        pass


def iter_streamed_json_array(chunks: Iterable[str]) -> Iterable[str]:
    """Convert a streamed JSON array into an iterable of JSON object strings.

    This ignores all characters before the start of the first array i.e. the first "["
    """
    pass


async def aiter_streamed_json_array(chunks: AsyncIterable[str]) -> AsyncIterable[str]:
    """Async version of `iter_streamed_json_array`."""
    pass


class CachedIterable(Iterable[T]):
    """Wraps an Iterable and caches the items after the first iteration."""

    def __init__(self, iterable: Iterable[T]):
        self._iterator = iter(iterable)
        self._cached_items: list[T] = []

    def __iter__(self) -> Iterator[T]:
        yield from self._cached_items
        for item in self._iterator:
            self._cached_items.append(item)
            yield item


class CachedAsyncIterable(AsyncIterable[T]):
    """Async version of `CachedIterable`."""

    def __init__(self, aiterable: AsyncIterable[T]):
        self._aiterator = aiter(aiterable)
        self._cached_items: list[T] = []

    async def __aiter__(self) -> AsyncIterator[T]:
        for item in self._cached_items:
            yield item
        async for item in self._aiterator:
            self._cached_items.append(item)
            yield item


# TODO: Add close method to close the underlying stream if chunks is a stream
# TODO: Make it a context manager to automatically close
class StreamedStr(Iterable[str]):
    """A string that is generated in chunks."""

    def __init__(self, chunks: Iterable[str]):
        self._chunks = CachedIterable(chunks)

    def __iter__(self) -> Iterator[str]:
        yield from self._chunks

    def __str__(self) -> str:
        return "".join(self)

    def to_string(self) -> str:
        """Convert the streamed string to a string."""
        pass

    def truncate(self, length: int) -> str:
        """Truncate the streamed string to the specified length."""
        pass


class AsyncStreamedStr(AsyncIterable[str]):
    """Async version of `StreamedStr`."""

    def __init__(self, chunks: AsyncIterable[str]):
        self._chunks = CachedAsyncIterable(chunks)

    async def __aiter__(self) -> AsyncIterator[str]:
        async for chunk in self._chunks:
            yield chunk

    async def to_string(self) -> str:
        """Convert the streamed string to a string."""
        pass

    async def truncate(self, length: int) -> str:
        """Truncate the streamed string to the specified length."""
        pass
