"""A worked example, runnable with no API key:

    python examples/workout.py

It uses a scripted stand-in for the model so the output is identical every
time. Swap ``Scripted`` for a real adapter and nothing else changes.

The domain is the one formwork was extracted from. The split and the volume
range are decisions a rule engine makes from the user's profile; the exercise
library is a filtered, closed set; the only thing genuinely being asked of a
language model is which movements to pick and how to explain them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import BaseModel, Field

from formwork import Session, Spec, chosen, computed, generate, generated, repair, rule, soft
from formwork.providers import Scripted

# ── the context your application already has ─────────────────────────────


@dataclass
class Profile:
    days_per_week: int
    experience: str
    max_per_day: int = 4
    library: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"id": "squat", "name": "Back Squat", "muscle": "legs"},
            {"id": "deadlift", "name": "Deadlift", "muscle": "back"},
            {"id": "bench", "name": "Bench Press", "muscle": "chest"},
            {"id": "row", "name": "Barbell Row", "muscle": "back"},
            {"id": "press", "name": "Overhead Press", "muscle": "shoulders"},
            {"id": "curl", "name": "Biceps Curl", "muscle": "arms"},
        ]
    )

    @property
    def library_ids(self) -> set[str]:
        return {e["id"] for e in self.library}

    @property
    def muscle_of(self) -> dict[str, str]:
        return {e["id"]: e["muscle"] for e in self.library}


# ── the deterministic half: ordinary functions, no model involved ────────


def split_for(profile: Profile) -> str:
    return {2: "full-body", 3: "push-pull-legs", 4: "upper-lower"}.get(
        profile.days_per_week, "full-body"
    )


def weekly_set_range(profile: Profile) -> tuple[int, int]:
    return {"beginner": (8, 14), "intermediate": (10, 20), "advanced": (14, 26)}[
        profile.experience
    ]


# ── the spec ─────────────────────────────────────────────────────────────


class Movement(BaseModel):
    id: str
    sets: Annotated[int, Field(ge=1, le=10)]
    reps: Annotated[int, Field(ge=1, le=30)]


class WeeklyPlan(Spec):
    """A week of training for one person."""

    __formwork_instructions__ = (
        "Design one week of resistance training. Pick movements that cover the "
        "week without repeating the same muscle group more than necessary."
    )

    split: Annotated[str, computed(split_for, describe="decided from days per week")]
    set_range: Annotated[
        tuple[int, int], computed(weekly_set_range, describe="total sets across the week")
    ]
    # A live run against Gemini returned five movements where four were
    # allowed, and it was right to: the rule text said "at most the configured
    # number" without ever saying what the number was. If a limit matters to
    # the model, it has to reach the model as a value, and a computed field is
    # how values get there. Rule prose alone is not a constraint.
    max_movements: Annotated[
        int,
        computed(lambda p: p.max_per_day, describe="hard ceiling on the movement count"),
    ]

    movements: Annotated[
        list[Movement],
        chosen(source="library", key="id", describe="one entry per movement"),
    ]
    rationale: Annotated[
        str, generated(describe="Two sentences for the user, plain language, no jargon.")
    ]

    def total_sets(self) -> int:
        return sum(m.sets for m in self.movements)

    # An invented id is fixable without asking the model again.
    @rule(
        "Every movement id must come from the supplied library.",
        fields=["movements"],
        repair=repair.drop_invalid("movements"),
    )
    def known_movements(self, profile: Profile) -> list[str]:
        return [
            f"{m.id!r} is not in the library"
            for m in self.movements
            if m.id not in profile.library_ids
        ]

    # So is a list that ran long.
    @rule(
        "Return no more movements than max_movements, stated above.",
        fields=["movements"],
        repair=repair.truncate("movements", lambda p: p.max_per_day),
    )
    def not_too_many(self, profile: Profile) -> str | None:
        if len(self.movements) > self.max_movements:
            return f"{len(self.movements)} movements, allowed {self.max_movements}"
        return None

    # This one is a judgement call, so it goes back to the model — but only
    # this field goes back.
    @rule("Total weekly sets must sit inside the prescribed range.", fields=["movements"])
    def volume(self, profile: Profile) -> str | None:
        low, high = self.set_range
        total = self.total_sets()
        if not low <= total <= high:
            return f"{total} total sets, allowed {low}-{high}"
        return None

    @soft(weight=1.0)
    def prefer_variety(self, profile: Profile) -> float:
        """Repeating a muscle group is allowed but not preferred."""
        muscles = [profile.muscle_of.get(m.id, "?") for m in self.movements]
        return float(len(muscles) - len(set(muscles)))


# ── a stand-in model that gets it wrong twice, the way they do ───────────

FIRST_TRY = {
    # Three things wrong at once, which is realistic: 'hack-squat' is not in
    # the library, that is five movements where four are allowed, and once the
    # invented one is dropped the week still adds up to 22 sets against a
    # ceiling of 20. The first two are mechanical. The third is a judgement
    # call, so only the third costs a second call.
    "movements": [
        {"id": "squat", "sets": 4, "reps": 8},
        {"id": "bench", "sets": 4, "reps": 8},
        {"id": "row", "sets": 4, "reps": 10},
        {"id": "hack-squat", "sets": 3, "reps": 12},
        {"id": "curl", "sets": 10, "reps": 12},
    ],
    "rationale": "A balanced week built around the main compound lifts.",
}

SECOND_TRY = {
    # Only 'movements' is asked for this time; the rationale above is kept.
    "movements": [
        {"id": "squat", "sets": 4, "reps": 8},
        {"id": "bench", "sets": 4, "reps": 8},
        {"id": "row", "sets": 4, "reps": 10},
        {"id": "press", "sets": 3, "reps": 10},
    ],
}


def main() -> None:
    profile = Profile(days_per_week=3, experience="intermediate")

    print("=" * 72)
    print("PROMPT THE MODEL RECEIVES")
    print("=" * 72)
    print(Session(WeeklyPlan, profile).next_request().prompt)

    print()
    print("=" * 72)
    print("RUN")
    print("=" * 72)

    plan, report = generate(WeeklyPlan, profile, Scripted([FIRST_TRY, SECOND_TRY]))

    print(f"split       : {plan.split}")
    print(f"set range   : {plan.set_range[0]}-{plan.set_range[1]}")
    print(f"movements   : {', '.join(m.id for m in plan.movements)}")
    print(f"total sets  : {plan.total_sets()}")
    print(f"rationale   : {plan.rationale}")
    print()
    print(f"report      : {report.summary()}")
    print(f"clean first : {report.valid_first_try}")

    for attempt in report.attempts:
        fixed = ", ".join(attempt.repaired_by) or "-"
        broke = ", ".join(v.rule for v in attempt.violations) or "-"
        print(
            f"  attempt {attempt.index}  {attempt.kind:<7} "
            f"asked for {attempt.targeted_fields}  repaired: {fixed}  unresolved: {broke}"
        )

    print()
    print("The rationale came from the first call and was never regenerated:")
    print(f"  {plan.rationale!r}")


if __name__ == "__main__":
    main()
