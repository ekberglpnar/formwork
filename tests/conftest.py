"""A small but realistic spec, shared by the tests.

Modelled on the case the library was extracted from: a training plan where the
split and the volume range are decided by a rule engine, the exercises must
come from a filtered library, and the prose is the only thing the model is
genuinely being asked for.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

# The benchmark lives at the repo root, outside the installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pydantic import BaseModel, Field

from formwork import Spec, chosen, computed, generated, repair, rule, soft

# ── context ──────────────────────────────────────────────────────────────


@dataclass
class Ctx:
    days: int = 3
    experience: str = "intermediate"
    max_per_day: int = 4
    library: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"id": "squat", "muscle": "legs"},
            {"id": "bench", "muscle": "chest"},
            {"id": "row", "muscle": "back"},
            {"id": "curl", "muscle": "arms"},
            {"id": "press", "muscle": "shoulders"},
        ]
    )

    @property
    def library_ids(self) -> set[str]:
        return {e["id"] for e in self.library}

    @property
    def muscle_of(self) -> dict[str, str]:
        return {e["id"]: e["muscle"] for e in self.library}


def split_for(ctx: Ctx) -> str:
    return {2: "full-body", 3: "push-pull-legs"}.get(ctx.days, "upper-lower")


def weekly_sets(ctx: Ctx) -> tuple[int, int]:
    return (10, 20) if ctx.experience == "intermediate" else (8, 14)


# ── spec ─────────────────────────────────────────────────────────────────


class PlanExercise(BaseModel):
    id: str
    sets: Annotated[int, Field(ge=1, le=10)]


class WorkoutPlan(Spec):
    """A week of training for one person."""

    split: Annotated[str, computed(split_for, describe="chosen by the rule engine")]
    set_range: Annotated[tuple[int, int], computed(weekly_sets)]

    exercises: Annotated[
        list[PlanExercise],
        chosen(source="library", key="id", describe="one entry per movement"),
    ]
    rationale: Annotated[str, generated(describe="Two sentences, plain language.")]

    def sets_total(self) -> int:
        return sum(e.sets for e in self.exercises)

    @rule(
        "Every exercise must come from the supplied library.",
        fields=["exercises"],
        repair=repair.drop_invalid("exercises"),
    )
    def known_exercises(self, ctx: Ctx) -> list[str]:
        return [
            f"{e.id} is not in the library"
            for e in self.exercises
            if e.id not in ctx.library_ids
        ]

    @rule("At most max_per_day exercises.", fields=["exercises"])
    def not_too_many(self, ctx: Ctx) -> str | None:
        if len(self.exercises) > ctx.max_per_day:
            return f"{len(self.exercises)} exercises, allowed at most {ctx.max_per_day}"
        return None

    @rule("Total weekly sets must sit inside the prescribed range.", fields=["exercises"])
    def volume(self, ctx: Ctx) -> str | None:
        low, high = self.set_range
        total = self.sets_total()
        if not low <= total <= high:
            return f"{total} total sets, allowed {low}-{high}"
        return None

    @soft(weight=1.0)
    def prefer_variety(self, ctx: Ctx) -> float:
        """Fewer repeated muscle groups is better."""
        muscles = [ctx.muscle_of.get(e.id, "?") for e in self.exercises]
        return float(len(muscles) - len(set(muscles)))


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def ctx() -> Ctx:
    return Ctx()


@pytest.fixture
def valid_response() -> dict[str, Any]:
    return {
        "exercises": [
            {"id": "squat", "sets": 4},
            {"id": "bench", "sets": 4},
            {"id": "row", "sets": 4},
        ],
        "rationale": "Balanced across the week. Compound movements first.",
    }
