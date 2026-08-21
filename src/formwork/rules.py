"""Semantic rules — the constraints a JSON schema cannot express.

A grammar can guarantee that ``sets`` is an integer. It cannot guarantee that
the sets across a week sum into the range your programming logic decided on, or
that every ``exercise_id`` exists in the 200-row library you passed in at
runtime. Those constraints are relational and context-dependent, so they live
here as ordinary Python predicates attached to the spec.

Violation messages are written to be read *by the model*: they go straight into
the repair prompt, so they should say what is wrong in concrete terms
("ex_42 is not in the library") rather than in schema terms ("invalid value").
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

__all__ = ["Violation", "Rule", "Objective", "rule", "soft", "RepairFn"]

# A repair takes the assembled instance, the violation it should fix, and the
# context; it returns a patched instance, or None if it cannot help.
RepairFn = Callable[[Any, "Violation", Any], Any | None]

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True, slots=True)
class Violation:
    """One broken rule.

    ``fields`` is what makes targeted repair possible: it names the fields
    implicated in the failure so the engine can freeze everything else and ask
    the model for a minimal edit instead of a fresh plan.
    """

    rule: str
    message: str
    fields: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"[{self.rule}] {self.message}"


@dataclass(frozen=True, slots=True)
class Rule:
    """A hard constraint. If it does not hold, the run does not succeed."""

    name: str
    fn: Callable[..., Any]
    message: str | None
    fields: tuple[str, ...]
    repair: RepairFn | None

    def check(self, instance: Any, ctx: Any) -> list[Violation]:
        """Run the predicate and normalise whatever it returned into violations.

        Accepted return shapes, in rough order of how often they get used:
        ``None``/``True`` (passed), ``False`` (failed, use the declared
        message), a ``str`` (failed with that message), a ``Violation``, or an
        iterable of ``str``/``Violation`` for rules that report several
        problems at once.
        """
        result = self.fn(instance, ctx)
        return list(self._normalise(result))

    def _normalise(self, result: Any) -> Iterator[Violation]:
        if result is None or result is True:
            return
        if result is False:
            yield self._violation(self.message or f"rule {self.name!r} failed")
            return
        if isinstance(result, Violation):
            yield self._with_defaults(result)
            return
        if isinstance(result, str):
            yield self._violation(result)
            return
        if isinstance(result, Iterable):
            for item in result:
                yield from self._normalise(item)
            return
        raise TypeError(
            f"rule {self.name!r} returned {type(result).__name__}; "
            "expected None, bool, str, Violation, or an iterable of those"
        )

    def _violation(self, message: str) -> Violation:
        return Violation(rule=self.name, message=message, fields=self.fields)

    def _with_defaults(self, violation: Violation) -> Violation:
        # A rule may build its own Violation but leave the bookkeeping to us.
        return Violation(
            rule=violation.rule or self.name,
            message=violation.message,
            fields=violation.fields or self.fields,
        )


@dataclass(frozen=True, slots=True)
class Objective:
    """A soft constraint: not a pass/fail gate, a number to push down.

    Used only when generating more than one candidate — the engine keeps the
    rule-valid candidate with the lowest weighted score.
    """

    name: str
    fn: Callable[..., float]
    weight: float

    def score(self, instance: Any, ctx: Any) -> float:
        return self.weight * float(self.fn(instance, ctx))


@dataclass(slots=True)
class _RuleMarker:
    message: str | None
    fields: tuple[str, ...]
    repair: RepairFn | None
    name: str | None


@dataclass(slots=True)
class _ObjectiveMarker:
    weight: float
    name: str | None


def rule(
    message: str | None = None,
    *,
    fields: Iterable[str] = (),
    repair: RepairFn | None = None,
    name: str | None = None,
) -> Callable[[F], F]:
    """Mark a method as a hard constraint.

    ``fields`` should name the fields the model would have to change to fix a
    failure. Getting this right is what turns a blind retry into a targeted
    one, so it is worth being accurate; if you leave it empty the engine falls
    back to regenerating every model-owned field.

    ``repair`` lets you fix the failure without calling the model at all.
    """

    def decorator(fn: F) -> F:
        fn.__formwork_rule__ = _RuleMarker(  # type: ignore[attr-defined]
            message=message,
            fields=tuple(fields),
            repair=repair,
            name=name,
        )
        return fn

    return decorator


def soft(
    *,
    weight: float = 1.0,
    name: str | None = None,
) -> Callable[[F], F]:
    """Mark a method as a soft objective returning a score to minimise."""

    def decorator(fn: F) -> F:
        fn.__formwork_objective__ = _ObjectiveMarker(  # type: ignore[attr-defined]
            weight=weight,
            name=name,
        )
        return fn

    return decorator


def collect(namespace: type) -> tuple[list[Rule], list[Objective]]:
    """Pull the decorated methods off a class, walking bases so rules inherit."""
    rules: dict[str, Rule] = {}
    objectives: dict[str, Objective] = {}

    for klass in reversed(namespace.__mro__):
        for attr, value in vars(klass).items():
            if marker := getattr(value, "__formwork_rule__", None):
                key = marker.name or attr
                rules[key] = Rule(
                    name=key,
                    fn=value,
                    message=marker.message,
                    fields=marker.fields,
                    repair=marker.repair,
                )
            if obj_marker := getattr(value, "__formwork_objective__", None):
                key = obj_marker.name or attr
                objectives[key] = Objective(name=key, fn=value, weight=obj_marker.weight)

    return list(rules.values()), list(objectives.values())


@dataclass(slots=True)
class RuleReport:
    """Result of running every rule once."""

    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def implicated_fields(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for violation in self.violations:
            for name in violation.fields:
                seen[name] = None
        return tuple(seen)
