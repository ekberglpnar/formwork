"""Deterministic repairs — fixing a violation without paying for a model call.

Not every failure needs the model. If it produced eleven exercises where ten
were allowed, dropping the eleventh is a decision your code can make, and
asking a language model to make it instead costs a round trip and may break
something else on the way. Rules that admit a mechanical fix declare one; the
engine tries it before it considers going back to the model.

A strategy returns the patched instance, or ``None`` to say "not my problem" —
in which case the violation survives and becomes part of the repair prompt.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from formwork.fields import Role, _read_source
from formwork.rules import RepairFn, Violation

__all__ = ["drop_invalid", "truncate", "clamp", "dedupe", "chain", "custom"]


def drop_invalid(field: str, *, source: str | None = None, key: str | None = None) -> RepairFn:
    """Remove list items whose value is outside the field's closed set.

    With no ``source`` the allowed pool is read from the field's own ``chosen``
    declaration, which is the case worth optimising for — you already said
    where the pool lives, saying it twice would be a papercut.
    """

    def repair(instance: Any, violation: Violation, ctx: Any) -> Any | None:
        spec = type(instance).spec_of(field)
        if source is not None:
            allowed = set(_read_source(ctx, source))
            probe = key
        else:
            if spec.role is not Role.CHOSEN:
                raise ValueError(
                    f"drop_invalid({field!r}) needs source=... "
                    "because the field is not declared with chosen()"
                )
            allowed = set(spec.allowed_values(ctx) or ())
            probe = key or spec.key

        current = getattr(instance, field)
        kept = [
            item for item in current if (_read_source(item, probe) if probe else item) in allowed
        ]
        if len(kept) == len(current):
            return None
        if not kept:
            # Emptying the field is not a repair, it is a different failure.
            return None
        return instance.patched(**{field: kept})

    return repair


def truncate(field: str, max_items: int | Callable[[Any], int]) -> RepairFn:
    """Cap a list at ``max_items``, keeping the head."""

    def repair(instance: Any, violation: Violation, ctx: Any) -> Any | None:
        limit = max_items(ctx) if callable(max_items) else max_items
        current = getattr(instance, field)
        if len(current) <= limit:
            return None
        return instance.patched(**{field: list(current)[:limit]})

    return repair


def clamp(
    field: str,
    low: float | Callable[[Any], float] | None = None,
    high: float | Callable[[Any], float] | None = None,
    *,
    item_field: str | None = None,
) -> RepairFn:
    """Pull a number into range.

    ``item_field`` clamps one attribute of every element of a list field, which
    is how it usually comes up: ``clamp("exercises", 1, 6, item_field="sets")``.
    """

    def repair(instance: Any, violation: Violation, ctx: Any) -> Any | None:
        lo = low(ctx) if callable(low) else low
        hi = high(ctx) if callable(high) else high

        def squeeze(value: Any) -> Any:
            if lo is not None and value < lo:
                return type(value)(lo)
            if hi is not None and value > hi:
                return type(value)(hi)
            return value

        current = getattr(instance, field)

        if item_field is None:
            fixed = squeeze(current)
            return None if fixed == current else instance.patched(**{field: fixed})

        changed = False
        items = []
        for item in current:
            value = getattr(item, item_field)
            fixed = squeeze(value)
            if fixed != value:
                changed = True
                item = item.model_copy(update={item_field: fixed})
            items.append(item)
        return instance.patched(**{field: items}) if changed else None

    return repair


def dedupe(field: str, *, key: str | None = None) -> RepairFn:
    """Drop repeated entries, keeping first occurrence."""

    def repair(instance: Any, violation: Violation, ctx: Any) -> Any | None:
        current = getattr(instance, field)
        seen: set[Any] = set()
        kept = []
        for item in current:
            probe = _read_source(item, key) if key else item
            if probe in seen:
                continue
            seen.add(probe)
            kept.append(item)
        if len(kept) == len(current):
            return None
        return instance.patched(**{field: kept})

    return repair


def chain(*strategies: RepairFn) -> RepairFn:
    """Apply strategies in order; succeeds if any of them changed something."""

    def repair(instance: Any, violation: Violation, ctx: Any) -> Any | None:
        current = instance
        touched = False
        for strategy in strategies:
            result = strategy(current, violation, ctx)
            if result is not None:
                current = result
                touched = True
        return current if touched else None

    return repair


def custom(fn: Callable[[Any, Violation, Any], Any | None]) -> RepairFn:
    """Escape hatch for a repair the built-ins do not cover."""
    return fn


def apply(
    instance: Any,
    violations: Sequence[Violation],
    ctx: Any,
    strategies: dict[str, RepairFn],
) -> tuple[Any, list[str]]:
    """Run whatever repairs the violated rules declared.

    Returns the (possibly) patched instance and the names of the rules whose
    repair actually did something. Rules are not re-checked here — the engine
    does that, once, after all repairs have been applied, because two repairs
    can interact and re-checking in between would report noise.
    """
    current = instance
    fired: list[str] = []

    for violation in violations:
        strategy = strategies.get(violation.rule)
        if strategy is None:
            continue
        patched = strategy(current, violation, ctx)
        if patched is not None:
            current = patched
            fired.append(violation.rule)

    return current, fired
