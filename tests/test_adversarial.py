"""The guarantee, tested the only way a guarantee can be: adversarially.

Everything else in the suite checks a path someone thought of. This file checks
the property that the library exists to provide — across hundreds of hostile
responses, a caller is never handed an object that breaks a declared rule.
"""

from __future__ import annotations

import contextlib

from conftest import WorkoutPlan
from formwork import FormworkError, generate
from formwork.providers import Chaos, Recording, Scripted

BASELINE = {
    "exercises": [
        {"id": "squat", "sets": 4},
        {"id": "bench", "sets": 4},
        {"id": "row", "sets": 4},
    ],
    "rationale": "Balanced across the week.",
}

SEEDS = range(200)


def test_never_returns_an_object_that_breaks_a_rule(ctx):
    """Either a valid plan or an exception. There is no third outcome."""
    returned = 0

    for seed in SEEDS:
        model = Chaos(baseline=BASELINE, seed=seed, aggression=0.3)
        try:
            plan, _ = generate(WorkoutPlan, ctx, model, max_attempts=3)
        except FormworkError:
            continue
        returned += 1
        assert plan.check(ctx).ok, f"seed {seed} produced an invalid plan"

    # If nothing ever succeeded the assertion above would be vacuous.
    assert returned > 0


def test_a_mild_adversary_mostly_converges(ctx):
    """Sanity check on the loop, not a performance claim.

    Guaranteeing validity is easy if you always raise; this pins down that the
    repair path actually rescues runs rather than the engine simply refusing.
    """
    succeeded = sum(
        _survives(ctx, Chaos(baseline=BASELINE, seed=seed, aggression=0.1))
        for seed in SEEDS
    )
    assert succeeded > len(SEEDS) // 2


def test_a_type_breaking_adversary_is_survived_too(ctx):
    """``repairable=False`` also emits values that miss the schema outright."""
    for seed in SEEDS:
        model = Chaos(baseline=BASELINE, seed=seed, aggression=0.5, repairable=False)
        try:
            plan, _ = generate(WorkoutPlan, ctx, model, max_attempts=3)
        except FormworkError:
            continue
        assert plan.check(ctx).ok


def test_chaos_respects_a_narrowed_repair_request(ctx):
    """A double that ignored request.fields would fail the run for the wrong
    reason, and the suite would be testing the double instead of the engine."""
    over = {
        "exercises": [
            {"id": "squat", "sets": 9},
            {"id": "bench", "sets": 9},
            {"id": "row", "sets": 9},
        ],
        "rationale": "keep me",
    }
    model = Recording(Chaos(baseline=over, seed=1, aggression=0.0))
    with contextlib.suppress(FormworkError):
        generate(WorkoutPlan, ctx, model, max_attempts=2)

    assert model.requests[1].fields == ("exercises",)


def test_chaos_is_deterministic_for_a_seed(ctx):
    first = Chaos(baseline=BASELINE, seed=7).generate_structured(_request(ctx))
    second = Chaos(baseline=BASELINE, seed=7).generate_structured(_request(ctx))
    assert first == second


def test_zero_aggression_is_a_passthrough(ctx):
    plan, report = generate(WorkoutPlan, ctx, Chaos(baseline=BASELINE, aggression=0.0))
    assert report.valid_first_try
    assert plan.sets_total() == 12


def test_scripted_running_dry_is_loud(ctx):
    """Silence here would look like a hang; an exhausted script is a finding."""
    import pytest

    from formwork.providers import Exhausted

    with pytest.raises(Exhausted, match="asked for 2"):
        generate(WorkoutPlan, ctx, Scripted([{"exercises": [{"id": "ghost", "sets": 1}]}]))


def _survives(ctx, model) -> bool:
    try:
        plan, _ = generate(WorkoutPlan, ctx, model, max_attempts=3)
    except FormworkError:
        return False
    assert plan.check(ctx).ok
    return True


def _request(ctx):
    from formwork import Session

    return Session(WorkoutPlan, ctx).next_request()
