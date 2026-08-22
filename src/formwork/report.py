"""Provenance for a single generation run.

Every ``generate`` call returns one of these alongside the object. The point is
auditability: in a regulated domain "the model said so" is not an answer, but
"rule ``weekly_volume`` was checked, failed once, and was repaired
deterministically by ``drop_excess``" is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from formwork.rules import Violation

__all__ = ["Usage", "Attempt", "Report"]


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one model call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            latency_ms=self.latency_ms + other.latency_ms,
        )


@dataclass(slots=True)
class Attempt:
    """One trip to the model.

    ``kind`` distinguishes the initial generation from the targeted repairs that
    follow it, because the whole efficiency claim of this library rests on the
    repairs being cheaper than the thing they replace (a full regeneration).
    """

    index: int
    kind: str  # "initial" | "repair"
    usage: Usage = field(default_factory=Usage)
    structural_errors: tuple[str, ...] = ()
    violations: tuple[Violation, ...] = ()
    repaired_by: tuple[str, ...] = ()
    targeted_fields: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Did the model's output need nothing at all?

        A deterministic repair counts against this: it rescued the run, but the
        output that arrived was still wrong. ``valid_first_try`` is the number
        the benchmark reports, so it must not be flattered here.
        """
        return not (self.structural_errors or self.violations or self.repaired_by)


@dataclass(slots=True)
class Report:
    """Everything that happened, in order."""

    attempts: list[Attempt] = field(default_factory=list)
    soft_scores: dict[str, float] = field(default_factory=dict)
    candidates_considered: int = 1

    @property
    def usage(self) -> Usage:
        total = Usage()
        for attempt in self.attempts:
            total = total + attempt.usage
        return total

    @property
    def model_calls(self) -> int:
        return len(self.attempts)

    @property
    def valid_first_try(self) -> bool:
        return bool(self.attempts) and self.attempts[0].ok

    @property
    def deterministic_repairs(self) -> tuple[str, ...]:
        return tuple(name for a in self.attempts for name in a.repaired_by)

    def summary(self) -> str:
        bits = [
            f"{self.model_calls} model call(s)",
            f"{self.usage.total_tokens} tokens",
        ]
        if repairs := self.deterministic_repairs:
            bits.append(f"repaired by {', '.join(repairs)}")
        if self.candidates_considered > 1:
            bits.append(f"best of {self.candidates_considered}")
        return "; ".join(bits)
