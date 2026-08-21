"""Model doubles, so an LLM feature can be unit-tested like ordinary code.

The claim this library makes is "you will never be handed an object that
breaks your rules". A claim like that is worth as much as the adversary you
tested it against, so the adversary ships with the library.

``Chaos`` is the interesting one: give it a baseline response and it will keep
returning plausible-but-wrong variations of it — unknown ids, counts over the
limit, numbers out of range, duplicated entries. Point your test at it, assert
that every run either returns a rule-valid object or raises, and you have
covered the failure mode that actually bites in production, which is not "the
model returned garbage" but "the model returned something that looked fine".
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from formwork.providers.base import Model, ModelRequest
from formwork.report import Usage

__all__ = ["Scripted", "Always", "Chaos", "Recording", "AsyncAdapter", "Exhausted"]


class Exhausted(RuntimeError):
    """A scripted double ran out of responses.

    Usually means the engine retried more times than the test expected, which
    is a finding rather than a nuisance — so this is loud.
    """


@dataclass(slots=True)
class Scripted:
    """Returns the given responses in order, one per call."""

    responses: Sequence[dict[str, Any]]
    usage: Usage = field(default_factory=lambda: Usage(100, 50, 10))
    _calls: int = 0

    def generate_structured(self, request: ModelRequest) -> tuple[dict[str, Any], Usage]:
        if self._calls >= len(self.responses):
            raise Exhausted(
                f"Scripted has {len(self.responses)} response(s) but the engine "
                f"asked for {self._calls + 1}"
            )
        response = self.responses[self._calls]
        self._calls += 1
        return dict(response), self.usage

    @property
    def call_count(self) -> int:
        return self._calls


@dataclass(slots=True)
class Always:
    """Returns the same response forever. For testing the give-up path."""

    response: dict[str, Any]
    usage: Usage = field(default_factory=lambda: Usage(100, 50, 10))
    _calls: int = 0

    def generate_structured(self, request: ModelRequest) -> tuple[dict[str, Any], Usage]:
        self._calls += 1
        return dict(self.response), self.usage

    @property
    def call_count(self) -> int:
        return self._calls


@dataclass(slots=True)
class Chaos:
    """Plausible-but-wrong variations on a baseline response.

    ``aggression`` is the probability that any given leaf gets perturbed.
    ``repairable`` limits mutations to ones the engine could in principle fix,
    which is what you want when the assertion under test is "it converges";
    turn it off when the assertion is "it gives up cleanly".
    """

    baseline: dict[str, Any]
    seed: int = 0
    aggression: float = 0.3
    repairable: bool = True
    unknown_id: str = "does-not-exist"
    usage: Usage = field(default_factory=lambda: Usage(100, 50, 10))
    _rng: random.Random = field(init=False, repr=False)
    _calls: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def generate_structured(self, request: ModelRequest) -> tuple[dict[str, Any], Usage]:
        self._calls += 1
        # A repair asks for a subset of fields; honour that or the engine's own
        # assembly would fail for reasons unrelated to what we are testing.
        source = self.baseline
        if request.fields:
            source = {k: v for k, v in self.baseline.items() if k in request.fields}
        return self._mutate(source), self.usage

    @property
    def call_count(self) -> int:
        return self._calls

    def _mutate(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._mutate(v) for k, v in value.items()}
        if isinstance(value, list):
            return self._mutate_list(value)
        return self._mutate_leaf(value)

    def _mutate_list(self, value: list[Any]) -> list[Any]:
        items = [self._mutate(item) for item in value]
        if not self._fires() or not items:
            return items
        choice = self._rng.choice(["duplicate", "extend", "drop"])
        if choice == "duplicate":
            items.append(dict(items[0]) if isinstance(items[0], dict) else items[0])
        elif choice == "extend":
            template = items[0]
            items.extend(
                dict(template) if isinstance(template, dict) else template for _ in range(3)
            )
        elif choice == "drop" and len(items) > 1:
            items.pop()
        return items

    def _mutate_leaf(self, value: Any) -> Any:
        if not self._fires():
            return value
        if isinstance(value, bool):
            return not value
        if isinstance(value, int):
            return self._rng.choice([value * 10, -abs(value) - 1, 0])
        if isinstance(value, float):
            return self._rng.choice([value * 10.0, -abs(value) - 1.0, 0.0])
        if isinstance(value, str):
            if self.repairable:
                return self.unknown_id
            return self._rng.choice([self.unknown_id, ""])
        return value

    def _fires(self) -> bool:
        return self._rng.random() < self.aggression


@dataclass(slots=True)
class Recording:
    """Wraps a model and keeps every request, for assertions about the loop.

    Mostly used to check the thing that justifies targeted repair existing:
    that the second call asked for two fields and not for the whole object.
    """

    inner: Model
    requests: list[ModelRequest] = field(default_factory=list)

    def generate_structured(self, request: ModelRequest) -> tuple[dict[str, Any], Usage]:
        self.requests.append(request)
        return self.inner.generate_structured(request)

    @property
    def call_count(self) -> int:
        return len(self.requests)

    @property
    def requested_fields(self) -> list[tuple[str, ...]]:
        return [r.fields for r in self.requests]

    def __iter__(self) -> Iterator[ModelRequest]:
        return iter(self.requests)


@dataclass(slots=True)
class AsyncAdapter:
    """Presents a synchronous model through the async interface."""

    inner: Model

    async def generate_structured(self, request: ModelRequest) -> tuple[dict[str, Any], Usage]:
        return self.inner.generate_structured(request)
