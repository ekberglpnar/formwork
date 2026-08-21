"""Three tasks, three shapes of constraint.

Domain variety is not the point — constraint variety is. A library that only
helps with closed-set membership would be a narrow thing, so the tasks cover
membership, tolerance arithmetic, exclusion, and combinatorial coverage.

Two conventions hold everywhere and both exist for fairness:

**Rules read the context, never the object's own computed fields.** Otherwise
an arm where the model supplies its own limits could satisfy a rule by
inventing a generous one, and would score better for cheating. ``truth(ctx)``
is the single source for every threshold.

**Every task carries a ``restates_constraints`` rule.** It is trivially true
when the constraints are computed by the rule engine, and can fail when the
model was asked to restate them. That difference is precisely the field
ownership claim, measured rather than asserted.

Tuples are avoided deliberately: Pydantic renders them as ``prefixItems``,
which Gemini's schema subset does not accept.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import BaseModel, Field

from formwork import Spec, chosen, computed, generated, repair, rule, soft

DIFFICULTIES = ("easy", "medium", "hard")


# ══════════════════════════════════════════════════════════════════════════
# 1. Workout — closed-set membership + a sum inside a range + a count cap
# ══════════════════════════════════════════════════════════════════════════

_MUSCLES = ["legs", "chest", "back", "shoulders", "arms", "core"]


@dataclass
class WorkoutCtx:
    difficulty: str
    library: list[dict[str, Any]]
    min_sets: int
    max_sets: int
    max_movements: int
    split: str

    @property
    def library_ids(self) -> set[str]:
        return {e["id"] for e in self.library}

    @property
    def muscle_of(self) -> dict[str, str]:
        return {e["id"]: e["muscle"] for e in self.library}


def workout_ctx(difficulty: str) -> WorkoutCtx:
    rng = random.Random(f"workout-{difficulty}")
    size = {"easy": 6, "medium": 18, "hard": 40}[difficulty]
    library = [
        {"id": f"ex{i:03d}", "name": f"Movement {i}", "muscle": _MUSCLES[i % len(_MUSCLES)]}
        for i in range(size)
    ]
    rng.shuffle(library)
    # The window narrows as difficulty rises: on hard there is exactly one
    # total that works, so the model has to do arithmetic rather than guess.
    window = {"easy": (10, 24), "medium": (14, 18), "hard": (15, 15)}[difficulty]
    return WorkoutCtx(
        difficulty=difficulty,
        library=library,
        min_sets=window[0],
        max_sets=window[1],
        max_movements={"easy": 5, "medium": 4, "hard": 3}[difficulty],
        split={"easy": "full-body", "medium": "push-pull-legs", "hard": "upper-lower"}[difficulty],
    )


def workout_truth(ctx: WorkoutCtx) -> dict[str, Any]:
    return {
        "split": ctx.split,
        "min_sets": ctx.min_sets,
        "max_sets": ctx.max_sets,
        "max_movements": ctx.max_movements,
    }


class Movement(BaseModel):
    id: str
    sets: Annotated[int, Field(ge=1, le=12)]
    reps: Annotated[int, Field(ge=1, le=30)]


class WorkoutPlan(Spec):
    """Design one week of resistance training."""

    split: Annotated[str, computed(lambda c: c.split, describe="training split to use")]
    min_sets: Annotated[int, computed(lambda c: c.min_sets, describe="minimum total sets")]
    max_sets: Annotated[int, computed(lambda c: c.max_sets, describe="maximum total sets")]
    max_movements: Annotated[
        int, computed(lambda c: c.max_movements, describe="hard ceiling on movement count")
    ]

    movements: Annotated[list[Movement], chosen(source="library", key="id")]
    rationale: Annotated[str, generated(describe="Two sentences for the user.")]

    def total_sets(self) -> int:
        return sum(m.sets for m in self.movements)

    @rule(
        "Every movement id must come from the library listed above.",
        fields=["movements"],
        repair=repair.drop_invalid("movements"),
    )
    def known_movements(self, ctx: WorkoutCtx) -> list[str]:
        return [
            f"{m.id!r} is not in the library"
            for m in self.movements
            if m.id not in ctx.library_ids
        ]

    @rule(
        "Return no more movements than max_movements.",
        fields=["movements"],
        repair=repair.truncate("movements", lambda c: c.max_movements),
    )
    def movement_count(self, ctx: WorkoutCtx) -> str | None:
        if len(self.movements) > ctx.max_movements:
            return f"{len(self.movements)} movements, allowed at most {ctx.max_movements}"
        return None

    @rule("Total sets across all movements must fall inside the range.", fields=["movements"])
    def volume(self, ctx: WorkoutCtx) -> str | None:
        total = self.total_sets()
        if not ctx.min_sets <= total <= ctx.max_sets:
            return f"{total} total sets, allowed {ctx.min_sets}-{ctx.max_sets}"
        return None

    @rule("The constraints stated in the prompt must be echoed back unchanged.")
    def restates_constraints(self, ctx: WorkoutCtx) -> list[str]:
        return _mismatches(self, workout_truth(ctx))

    @soft(weight=1.0)
    def variety(self, ctx: WorkoutCtx) -> float:
        muscles = [ctx.muscle_of.get(m.id, "?") for m in self.movements]
        return float(len(muscles) - len(set(muscles)))


# ══════════════════════════════════════════════════════════════════════════
# 2. Nutrition — tolerance arithmetic + a floor + an exclusion list
# ══════════════════════════════════════════════════════════════════════════

_FOODS = [
    ("chicken", 165, 31.0, []),
    ("rice", 130, 2.7, []),
    ("oats", 389, 16.9, ["gluten"]),
    ("milk", 61, 3.2, ["dairy"]),
    ("yoghurt", 59, 10.0, ["dairy"]),
    ("egg", 155, 13.0, ["egg"]),
    ("almond", 579, 21.0, ["nuts"]),
    ("salmon", 208, 20.0, ["fish"]),
    ("lentil", 116, 9.0, []),
    ("bread", 265, 9.0, ["gluten"]),
    ("beef", 250, 26.0, []),
    ("tofu", 76, 8.0, ["soy"]),
    ("banana", 89, 1.1, []),
    ("cheese", 402, 25.0, ["dairy"]),
    ("potato", 77, 2.0, []),
]


@dataclass
class NutritionCtx:
    difficulty: str
    catalogue: list[dict[str, Any]]
    kcal_target: int
    tolerance_pct: int
    protein_min: int
    allergens: list[str] = field(default_factory=list)

    @property
    def catalogue_ids(self) -> set[str]:
        return {f["id"] for f in self.catalogue}

    @property
    def by_id(self) -> dict[str, dict[str, Any]]:
        return {f["id"]: f for f in self.catalogue}

    def kcal_of(self, pick: Any) -> float:
        food = self.by_id.get(pick.id)
        return 0.0 if food is None else food["kcal_per_100g"] * pick.grams / 100.0

    def protein_of(self, pick: Any) -> float:
        food = self.by_id.get(pick.id)
        return 0.0 if food is None else food["protein_per_100g"] * pick.grams / 100.0


def nutrition_ctx(difficulty: str) -> NutritionCtx:
    catalogue = [
        {"id": name, "kcal_per_100g": kcal, "protein_per_100g": protein, "allergens": allergens}
        for name, kcal, protein, allergens in _FOODS
    ]
    tolerance = {"easy": 15, "medium": 7, "hard": 3}[difficulty]
    allergens = {"easy": [], "medium": ["dairy"], "hard": ["dairy", "gluten", "nuts"]}[difficulty]
    return NutritionCtx(
        difficulty=difficulty,
        catalogue=catalogue,
        kcal_target={"easy": 2000, "medium": 2200, "hard": 1850}[difficulty],
        tolerance_pct=tolerance,
        protein_min={"easy": 90, "medium": 120, "hard": 140}[difficulty],
        allergens=allergens,
    )


def nutrition_truth(ctx: NutritionCtx) -> dict[str, Any]:
    return {
        "kcal_target": ctx.kcal_target,
        "tolerance_pct": ctx.tolerance_pct,
        "protein_min": ctx.protein_min,
    }


class FoodPick(BaseModel):
    id: str
    grams: Annotated[int, Field(ge=10, le=1000)]


class DayPlan(Spec):
    """Plan one day of eating."""

    kcal_target: Annotated[int, computed(lambda c: c.kcal_target, describe="target calories")]
    tolerance_pct: Annotated[
        int, computed(lambda c: c.tolerance_pct, describe="allowed deviation, percent")
    ]
    protein_min: Annotated[
        int, computed(lambda c: c.protein_min, describe="minimum grams of protein")
    ]

    foods: Annotated[list[FoodPick], chosen(source="catalogue", key="id")]
    note: Annotated[str, generated(describe="One sentence for the user.")]

    def kcal(self, ctx: NutritionCtx) -> float:
        return sum(ctx.kcal_of(f) for f in self.foods)

    def protein(self, ctx: NutritionCtx) -> float:
        return sum(ctx.protein_of(f) for f in self.foods)

    @rule(
        "Every food id must come from the catalogue listed above.",
        fields=["foods"],
        repair=repair.drop_invalid("foods"),
    )
    def known_foods(self, ctx: NutritionCtx) -> list[str]:
        return [
            f"{f.id!r} is not in the catalogue"
            for f in self.foods
            if f.id not in ctx.catalogue_ids
        ]

    @rule("Total calories must land within the stated tolerance of the target.", fields=["foods"])
    def calories(self, ctx: NutritionCtx) -> str | None:
        slack = ctx.kcal_target * ctx.tolerance_pct / 100.0
        total = self.kcal(ctx)
        if abs(total - ctx.kcal_target) > slack:
            low = ctx.kcal_target - slack
            high = ctx.kcal_target + slack
            return (
                f"{total:.0f} kcal, target {ctx.kcal_target} "
                f"+/-{ctx.tolerance_pct}% ({low:.0f}-{high:.0f})"
            )
        return None

    @rule("Protein must meet the floor.", fields=["foods"])
    def protein_floor(self, ctx: NutritionCtx) -> str | None:
        total = self.protein(ctx)
        if total < ctx.protein_min:
            return f"{total:.0f}g protein, need at least {ctx.protein_min}g"
        return None

    @rule(
        "No food may contain an allergen the user cannot eat.",
        fields=["foods"],
        repair=repair.custom(
            lambda instance, violation, ctx: _drop_allergens(instance, ctx)
        ),
    )
    def allergens(self, ctx: NutritionCtx) -> list[str]:
        bad = []
        for pick in self.foods:
            food = ctx.by_id.get(pick.id)
            if food is None:
                continue
            hit = set(food["allergens"]) & set(ctx.allergens)
            if hit:
                bad.append(f"{pick.id!r} contains {', '.join(sorted(hit))}")
        return bad

    @rule("The constraints stated in the prompt must be echoed back unchanged.")
    def restates_constraints(self, ctx: NutritionCtx) -> list[str]:
        return _mismatches(self, nutrition_truth(ctx))

    @soft(weight=1.0)
    def few_items(self, ctx: NutritionCtx) -> float:
        """Shorter shopping lists are nicer."""
        return float(max(0, len(self.foods) - 5))


def _drop_allergens(instance: Any, ctx: NutritionCtx) -> Any | None:
    kept = [
        pick
        for pick in instance.foods
        if not (set(ctx.by_id.get(pick.id, {}).get("allergens", [])) & set(ctx.allergens))
    ]
    if len(kept) == len(instance.foods) or not kept:
        return None
    return instance.patched(foods=kept)


# ══════════════════════════════════════════════════════════════════════════
# 3. Shifts — combinatorial coverage, the shape the other two do not cover
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class ShiftCtx:
    difficulty: str
    roster: list[dict[str, Any]]
    days: int
    staff_per_day: int
    max_shifts_per_person: int

    @property
    def roster_ids(self) -> set[str]:
        return {e["id"] for e in self.roster}


def shift_ctx(difficulty: str) -> ShiftCtx:
    size = {"easy": 8, "medium": 6, "hard": 5}[difficulty]
    roster = [{"id": f"emp{i:02d}", "name": f"Employee {i}"} for i in range(size)]
    days = {"easy": 3, "medium": 5, "hard": 5}[difficulty]
    staff = {"easy": 2, "medium": 2, "hard": 3}[difficulty]
    # On hard, 5 days x 3 staff = 15 slots across 5 people at 3 each: the only
    # solution is everyone working exactly their maximum.
    cap = {"easy": 3, "medium": 2, "hard": 3}[difficulty]
    return ShiftCtx(
        difficulty=difficulty,
        roster=roster,
        days=days,
        staff_per_day=staff,
        max_shifts_per_person=cap,
    )


def shift_truth(ctx: ShiftCtx) -> dict[str, Any]:
    return {
        "days": ctx.days,
        "staff_per_day": ctx.staff_per_day,
        "max_shifts_per_person": ctx.max_shifts_per_person,
    }


class Assignment(BaseModel):
    day: Annotated[int, Field(ge=1, le=7)]
    employee_id: str


class Roster(Spec):
    """Build the week's shift rota."""

    days: Annotated[int, computed(lambda c: c.days, describe="number of days to cover")]
    staff_per_day: Annotated[
        int, computed(lambda c: c.staff_per_day, describe="people required each day")
    ]
    max_shifts_per_person: Annotated[
        int, computed(lambda c: c.max_shifts_per_person, describe="cap per person for the week")
    ]

    assignments: Annotated[list[Assignment], chosen(source="roster", key="id")]
    note: Annotated[str, generated(describe="One sentence for the manager.")]

    @rule(
        "Every employee_id must come from the roster listed above.",
        fields=["assignments"],
        repair=repair.custom(
            lambda instance, violation, ctx: _drop_unknown_staff(instance, ctx)
        ),
    )
    def known_staff(self, ctx: ShiftCtx) -> list[str]:
        return [
            f"{a.employee_id!r} is not on the roster"
            for a in self.assignments
            if a.employee_id not in ctx.roster_ids
        ]

    @rule(
        "Nobody may be assigned twice on the same day.",
        fields=["assignments"],
        repair=repair.custom(lambda instance, violation, ctx: _dedupe_day_staff(instance)),
    )
    def no_double_booking(self, ctx: ShiftCtx) -> list[str]:
        seen: set[tuple[int, str]] = set()
        clashes = []
        for a in self.assignments:
            key = (a.day, a.employee_id)
            if key in seen:
                clashes.append(f"{a.employee_id} is on day {a.day} more than once")
            seen.add(key)
        return clashes

    @rule("Each day must have exactly the required number of staff.", fields=["assignments"])
    def coverage(self, ctx: ShiftCtx) -> list[str]:
        counts = {day: 0 for day in range(1, ctx.days + 1)}
        problems = []
        for a in self.assignments:
            if a.day in counts:
                counts[a.day] += 1
            else:
                problems.append(f"day {a.day} is outside 1-{ctx.days}")
        for day, count in counts.items():
            if count != ctx.staff_per_day:
                problems.append(f"day {day} has {count} staff, needs exactly {ctx.staff_per_day}")
        return problems

    @rule("Nobody may exceed the weekly cap.", fields=["assignments"])
    def weekly_cap(self, ctx: ShiftCtx) -> list[str]:
        counts: dict[str, int] = {}
        for a in self.assignments:
            counts[a.employee_id] = counts.get(a.employee_id, 0) + 1
        return [
            f"{who} works {n} shifts, cap is {ctx.max_shifts_per_person}"
            for who, n in sorted(counts.items())
            if n > ctx.max_shifts_per_person
        ]

    @rule("The constraints stated in the prompt must be echoed back unchanged.")
    def restates_constraints(self, ctx: ShiftCtx) -> list[str]:
        return _mismatches(self, shift_truth(ctx))

    @soft(weight=1.0)
    def fairness(self, ctx: ShiftCtx) -> float:
        """An even spread of shifts across people is preferable."""
        counts: dict[str, int] = {}
        for a in self.assignments:
            counts[a.employee_id] = counts.get(a.employee_id, 0) + 1
        if not counts:
            return 0.0
        return float(max(counts.values()) - min(counts.values()))


def _drop_unknown_staff(instance: Any, ctx: ShiftCtx) -> Any | None:
    kept = [a for a in instance.assignments if a.employee_id in ctx.roster_ids]
    if len(kept) == len(instance.assignments) or not kept:
        return None
    return instance.patched(assignments=kept)


def _dedupe_day_staff(instance: Any) -> Any | None:
    seen: set[tuple[int, str]] = set()
    kept = []
    for a in instance.assignments:
        key = (a.day, a.employee_id)
        if key in seen:
            continue
        seen.add(key)
        kept.append(a)
    if len(kept) == len(instance.assignments):
        return None
    return instance.patched(assignments=kept)


# ══════════════════════════════════════════════════════════════════════════


def _mismatches(instance: Any, truth: dict[str, Any]) -> list[str]:
    """Which stated constraints did the object come back with wrong?

    Always empty when the fields are computed — the rule engine wrote them.
    Can fail when the model was asked to restate them, which is the whole
    point of measuring it.
    """
    return [
        f"{name} came back as {getattr(instance, name)!r}, should be {expected!r}"
        for name, expected in truth.items()
        if getattr(instance, name, None) != expected
    ]


@dataclass(frozen=True)
class Task:
    name: str
    spec: type[Spec]
    make_ctx: Any
    truth: Any


TASKS: dict[str, Task] = {
    "workout": Task("workout", WorkoutPlan, workout_ctx, workout_truth),
    "nutrition": Task("nutrition", DayPlan, nutrition_ctx, nutrition_truth),
    "shifts": Task("shifts", Roster, shift_ctx, shift_truth),
}
