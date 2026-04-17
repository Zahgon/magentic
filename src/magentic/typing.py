import inspect
import types
from collections.abc import Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

T_co = TypeVar("T_co", covariant=True)

if TYPE_CHECKING:
    # Cannot be defined at runtime because Protocol cannot inherit from non-Protocol
    class NonStringSequence(Sequence[T_co], Protocol[T_co]):  # type: ignore[misc]
        """Protocol that matches Sequences except for `str`."""

        # HACK: Works because `__contains__` method of `str` does not match `Sequence`
        # See: https://github.com/python/typing/issues/256#issuecomment-1442633430


def is_union_type(type_: type) -> bool:
    """Return True if the type is a union type."""
    pass


TypeT = TypeVar("TypeT", bound=type)


def split_union_type(type_: TypeT) -> Sequence[TypeT]:
    """Split a union type into its constituent types."""
    pass


def is_origin_abstract(type_: type) -> bool:
    """Return true if the unsubscripted type is an abstract base class (ABC)."""
    pass


def is_origin_subclass(
    type_: type, cls_or_tuple: TypeT | tuple[TypeT, ...]
) -> TypeGuard[TypeT]:
    """Check if the unsubscripted type is a subclass of the given class(es)."""
    pass


def is_any_origin_subclass(
    types: Iterable[type], cls_or_tuple: TypeT | tuple[TypeT, ...]
) -> bool:
    """Check if any of the unsubscripted types is a subclass of the given class(es)."""
    pass


def name_type(type_: type) -> str:
    """Generate a name for the given type.

    e.g. `list[str]` -> `"list_of_str"`
    """
    pass
