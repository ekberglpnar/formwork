"""The generation loop."""

from __future__ import annotations

from typing import Annotated

import pytest

from conftest import WorkoutPlan
from formwork import (
    ConstraintError,
    Session,
    Spec,
    StructuralError,
    agenerate,
    generate,
    generated,
    rule,
)
from formwork.providers import Always, AsyncAdapter, Recording, Scripted

VALID = {
    "exercises": [
        {"id": "squat", "sets": 4},
        {"id": "bench", "sets": 4},
        {"id": "row", "sets": 4},
    ],
    "rationale": "Balanced across the week.",
}

# Valid ids, sane count, but 27 sets against a 10-20 range. No rule declares a
# deterministic repair for volume, so this is the case that must go back to the
# model — and it is the case targeted repair exists for.
OVER_VOLUME = {
    "exercises": [
        {"id": "squat", "sets": 9},
        {"id": "bench", "sets": 9},
        {"id": "row", "sets": 9},
    ],
    "rationale": "Too much work.",
}

# One invented id among three real ones: fixable without a model call.
UNKNOWN_ID = {
    "exercises": [
        {"id": "squat", "sets": 4},
        {"id": "bench", "sets": 4},
        {"id": "row", "sets": 4},
        {"id": "ghost", "sets": 4},
    ],
    "rationale": "Contains a movement that does not exist.",
}


# ── happy path ───────────────────────────────────────────────────────────


def test_clean_output_costs_one_call(ctx):
    model = Scripted([VALID])
    plan, report = generate(WorkoutPlan, ctx, model)

    assert plan.split == "push-pull-legs"
    assert plan.sets_total() == 12
    assert report.model_calls == 1
    assert report.valid_first_try


def test_computed_fields_are_filled_from_the_rule_engine(ctx):
    plan, _ = generate(WorkoutPlan, ctx, Scripted([VALID]))
    assert plan.set_range == (10, 20)


def test_usage_is_accumulated(ctx):
    model = Scripted([OVER_VOLUME, {"exercises": VALID["exercises"]}])
    _, report = generate(WorkoutPlan, ctx, model)
    assert report.usage.total_tokens == 300
    assert report.model_calls == 2


# ── deterministic repair ─────────────────────────────────────────────────


def test_deterministic_repair_avoids_a_second_call(ctx):
    model = Scripted([UNKNOWN_ID])
    plan, report = generate(WorkoutPlan, ctx, model)

    assert [e.id for e in plan.exercises] == ["squat", "bench", "row"]
    assert model.call_count == 1
    assert report.deterministic_repairs == ("known_exercises",)


def test_a_repaired_run_does_not_count_as_valid_first_try(ctx):
    _, report = generate(WorkoutPlan, ctx, Scripted([UNKNOWN_ID]))
    assert not report.valid_first_try


# ── targeted repair ──────────────────────────────────────────────────────


def test_repair_asks_only_for_the_implicated_fields(ctx):
    """The whole efficiency argument, asserted."""
    model = Recording(Scripted([OVER_VOLUME, {"exercises": VALID["exercises"]}]))
    plan, report = generate(WorkoutPlan, ctx, model)

    assert model.requested_fields == [
        ("exercises", "rationale"),
        ("exercises",),
    ]
    assert report.attempts[1].kind == "repair"
    assert plan.sets_total() == 12


def test_frozen_fields_survive_the_repair(ctx):
    """The rationale was fine, so it must come back unchanged and unpaid-for."""
    model = Scripted([OVER_VOLUME, {"exercises": VALID["exercises"]}])
    plan, _ = generate(WorkoutPlan, ctx, model)
    assert plan.rationale == "Too much work."


def test_repair_prompt_names_the_violation(ctx):
    model = Recording(Scripted([OVER_VOLUME, {"exercises": VALID["exercises"]}]))
    generate(WorkoutPlan, ctx, model)
    assert "27 total sets, allowed 10-20" in model.requests[1].prompt


def test_rules_without_declared_fields_fall_back_to_everything(ctx):
    class Vague(Spec):
        text: Annotated[str, generated()]

        @rule("must not be empty")
        def non_empty(self, ctx):
            return len(self.text) > 0

    model = Recording(Scripted([{"text": ""}, {"text": "ok"}]))
    generate(Vague, None, model)
    assert model.requested_fields == [("text",), ("text",)]


# ── giving up ────────────────────────────────────────────────────────────


def test_gives_up_rather_than_returning_an_invalid_object(ctx):
    model = Always(OVER_VOLUME)
    with pytest.raises(ConstraintError) as caught:
        generate(WorkoutPlan, ctx, model, max_attempts=3)

    assert model.call_count == 3
    assert "27 total sets" in str(caught.value)
    assert caught.value.report.model_calls == 3


def test_max_attempts_of_one_means_no_repair(ctx):
    model = Always(OVER_VOLUME)
    with pytest.raises(ConstraintError):
        generate(WorkoutPlan, ctx, model, max_attempts=1)
    assert model.call_count == 1


def test_max_attempts_must_be_positive(ctx):
    with pytest.raises(ValueError, match="at least 1"):
        Session(WorkoutPlan, ctx, max_attempts=0)


# ── structural failures ──────────────────────────────────────────────────


def test_schema_miss_is_retried_with_the_errors_quoted(ctx):
    model = Recording(Scripted([{"rationale": "no exercises key"}, VALID]))
    plan, report = generate(WorkoutPlan, ctx, model)

    assert report.attempts[0].structural_errors
    assert "exercises" in model.requests[1].prompt
    # A structural miss leaves no instance to patch, so the retry is a full one.
    assert model.requests[1].kind == "initial"
    assert plan.sets_total() == 12


def test_persistent_schema_miss_raises_structural_error(ctx):
    with pytest.raises(StructuralError):
        generate(WorkoutPlan, ctx, Always({"rationale": "still wrong"}), max_attempts=2)


def test_pydantic_constraints_are_enforced_on_the_model(ctx):
    """sets=99 breaks Field(le=10), which is a structural miss, not a rule."""
    bad = {"exercises": [{"id": "squat", "sets": 99}], "rationale": "x"}
    model = Scripted([bad, VALID])
    plan, report = generate(WorkoutPlan, ctx, model)
    assert report.attempts[0].structural_errors
    assert plan.sets_total() == 12


# ── soft objectives ──────────────────────────────────────────────────────


def test_candidates_keeps_the_best_scoring_valid_result(ctx):
    repetitive = {
        "exercises": [
            {"id": "squat", "sets": 4},
            {"id": "squat", "sets": 4},
            {"id": "squat", "sets": 4},
        ],
        "rationale": "All legs.",
    }
    model = Scripted([repetitive, VALID])
    plan, report = generate(WorkoutPlan, ctx, model, candidates=2)

    assert [e.id for e in plan.exercises] == ["squat", "bench", "row"]
    assert report.candidates_considered == 2
    assert report.soft_scores == {"prefer_variety": 0.0}


def test_candidate_cost_is_reported_in_full(ctx):
    model = Scripted([VALID, VALID])
    _, report = generate(WorkoutPlan, ctx, model, candidates=2)
    assert report.model_calls == 2
    assert report.usage.total_tokens == 300


def test_failed_candidates_still_report_their_cost(ctx):
    with pytest.raises(ConstraintError) as caught:
        generate(WorkoutPlan, ctx, Always(OVER_VOLUME), max_attempts=2, candidates=2)
    assert caught.value.report.model_calls == 4


def test_objectives_do_not_gate_a_single_candidate_run(ctx):
    """A bad soft score is not a failure — that is what makes it soft."""
    repetitive = {
        "exercises": [{"id": "squat", "sets": 6}, {"id": "squat", "sets": 6}],
        "rationale": "All legs.",
    }
    plan, report = generate(WorkoutPlan, ctx, Scripted([repetitive]))
    assert report.soft_scores == {"prefer_variety": 1.0}


# ── driving the session yourself ─────────────────────────────────────────


def test_session_can_be_driven_by_hand(ctx):
    session = Session(WorkoutPlan, ctx)

    first = session.next_request()
    assert first is not None
    assert set(first.schema.model_fields) == {"exercises", "rationale"}
    assert "push-pull-legs" in first.prompt

    session.feed(VALID)
    assert session.done
    assert session.next_request() is None

    plan, report = session.finish()
    assert plan.sets_total() == 12
    assert report.model_calls == 1


def test_feeding_without_a_request_is_an_error(ctx):
    session = Session(WorkoutPlan, ctx)
    with pytest.raises(RuntimeError, match="no outstanding request"):
        session.feed(VALID)


def test_next_request_is_idempotent_until_fed(ctx):
    session = Session(WorkoutPlan, ctx)
    assert session.next_request() is session.next_request()


# ── async ────────────────────────────────────────────────────────────────


async def test_async_driver_matches_the_sync_one(ctx):
    model = AsyncAdapter(Scripted([OVER_VOLUME, {"exercises": VALID["exercises"]}]))
    plan, report = await agenerate(WorkoutPlan, ctx, model)
    assert plan.sets_total() == 12
    assert report.model_calls == 2


async def test_async_driver_raises_the_same_error(ctx):
    with pytest.raises(ConstraintError):
        await agenerate(WorkoutPlan, ctx, AsyncAdapter(Always(OVER_VOLUME)), max_attempts=2)


# ── prompt content ───────────────────────────────────────────────────────


def test_prompt_states_computed_fields_as_decisions(ctx):
    session = Session(WorkoutPlan, ctx)
    prompt = session.next_request().prompt
    assert "## Already decided" in prompt
    assert "do not return them" in prompt
    assert "chosen by the rule engine" in prompt


def test_prompt_lists_the_closed_set(ctx):
    prompt = Session(WorkoutPlan, ctx).next_request().prompt
    assert "squat, bench, row, curl, press" in prompt
    assert "will be rejected" in prompt


def test_prompt_states_the_rules(ctx):
    prompt = Session(WorkoutPlan, ctx).next_request().prompt
    assert "Every exercise must come from the supplied library." in prompt
    assert "Total weekly sets must sit inside the prescribed range." in prompt
