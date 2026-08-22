"""Field roles — who owns which part of the output.

This is the idea the rest of the library is built around. In a generation task
with real domain logic, some fields are *decisions your code already knows how
to make* and some are *the reason you reached for a model in the first place*.
Mixing them into one prompt and hoping the model respects the first kind is the
default approach, and it is why these features are flaky.

So each field declares a role:

``computed``   your code fills it, before the model is called; the model never
               sees it as a choice, only as a stated fact.
``chosen``     the model picks, but from a closed set you supply at runtime.
``generated``  the model is free.

The practical consequence is that the schema sent to the model contains only
the ``chosen`` and ``generated`` fields. A field the model cannot see is a
field it cannot get wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = ["Role", "FieldSpec", "computed", "chosen", "generated"]


class Role(StrEnum):
    COMPUTED = "computed"
    CHOSEN = "chosen"
    GENERATED = "generated"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Metadata attached to a field through ``Annotated``."""

    role: Role
    resolver: Callable[[Any], Any] | None = None
    source: str | None = None
    key: str | None = None
    describe: str | None = None
    max_items: int | None = None

    @property
    def model_facing(self) -> bool:
        """Does this field appear in the schema handed to the model?"""
        return self.role is not Role.COMPUTED

    def allowed_values(self, ctx: Any) -> tuple[Any, ...] | None:
        """The closed set for a ``chosen`` field, read off the context.

        Returned as a tuple so callers can put it in a prompt, hand it to a
        grammar backend, or check membership — the three things anyone wants to
        do with a closed set.
        """
        if self.role is not Role.CHOSEN or self.source is None:
            return None
        pool = _read_source(ctx, self.source)
        if self.key is None:
            return tuple(pool)
        return tuple(_read_source(item, self.key) for item in pool)

    def resolve(self, ctx: Any) -> Any:
        if self.resolver is None:
            raise ValueError("computed field has no resolver")
        return self.resolver(ctx)


def _read_source(obj: Any, path: str) -> Any:
    """Read ``path`` off a context object, tolerating dicts and attributes.

    Contexts in the wild are ORM rows, dataclasses, Pydantic models and plain
    dicts in roughly equal measure; refusing three of those would be a silly
    reason for a library to be unusable.
    """
    current = obj
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(f"context has no key {path!r}")
            current = current[part]
        else:
            if not hasattr(current, part):
                raise AttributeError(f"context has no attribute {path!r}")
            current = getattr(current, part)
    return current


def computed(resolver: Callable[[Any], Any], *, describe: str | None = None) -> FieldSpec:
    """Your code owns this field.

    ``resolver`` is called with the context before the model runs. The value is
    stated to the model as a fact so it can write around it, but the model is
    never asked to produce it.
    """
    return FieldSpec(role=Role.COMPUTED, resolver=resolver, describe=describe)


def chosen(
    *,
    source: str,
    key: str | None = None,
    describe: str | None = None,
    max_items: int | None = None,
) -> FieldSpec:
    """The model picks from a closed set.

    ``source`` is a dotted path into the context holding the allowed pool.
    ``key`` is for the common case where the field is a list of objects and it
    is one attribute of each — an id — that must come from the pool, while the
    rest of the object is generated.
    """
    return FieldSpec(
        role=Role.CHOSEN,
        source=source,
        key=key,
        describe=describe,
        max_items=max_items,
    )


def generated(*, describe: str | None = None, max_items: int | None = None) -> FieldSpec:
    """The model is free within the type."""
    return FieldSpec(role=Role.GENERATED, describe=describe, max_items=max_items)
