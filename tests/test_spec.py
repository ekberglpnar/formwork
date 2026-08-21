"""Field roles, and the schema split that follows from them."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from formwork import Role
from conftest import PlanExercise, WorkoutPlan


def test_roles_are_read_off_the_annotations():
    assert WorkoutPlan.spec_of("split").role is Role.COMPUTED
    assert WorkoutPlan.spec_of("set_range").role is Role.COMPUTED
    assert WorkoutPlan.spec_of("exercises").role is Role.CHOSEN
    assert WorkoutPlan.spec_of("rationale").role is Role.GENERATED


def test_computed_fields_are_absent_from_the_model_schema():
    """The central claim: what the model cannot see, it cannot get wrong."""
    schema = WorkoutPlan.model_facing_schema()
    assert set(schema.model_fields) == {"exercises", "rationale"}


def test_pydantic_constraints_survive_into_the_model_schema():
    schema = WorkoutPlan.model_facing_schema()
    json_schema = schema.model_json_schema()
    item = json_schema["$defs"]["PlanExercise"]["properties"]["sets"]
    assert item["minimum"] == 1
    assert item["maximum"] == 10


def test_narrowed_schema_for_targeted_repair():
    schema = WorkoutPlan.model_facing_schema(only=("exercises",))
    assert set(schema.model_fields) == {"exercises"}


def test_narrowed_schema_refuses_computed_fields():
    with pytest.raises(ValueError, match="computed"):
        WorkoutPlan.model_facing_schema(only=("split",))


def test_closed_set_is_read_from_the_context(ctx):
    allowed = WorkoutPlan.spec_of("exercises").allowed_values(ctx)
    assert allowed == ("squat", "bench", "row", "curl", "press")


def test_resolve_computed_runs_the_rule_engine(ctx):
    values = WorkoutPlan.resolve_computed(ctx)
    assert values == {"split": "push-pull-legs", "set_range": (10, 20)}


def test_assemble_joins_both_halves(ctx, valid_response):
    partial = WorkoutPlan.model_facing_schema().model_validate(valid_response)
    plan = WorkoutPlan.assemble(WorkoutPlan.resolve_computed(ctx), partial)
    assert plan.split == "push-pull-legs"
    assert plan.sets_total() == 12


def test_patched_revalidates(ctx, valid_response):
    plan = _plan(ctx, valid_response)
    smaller = plan.patched(exercises=plan.exercises[:1])
    assert smaller.sets_total() == 4
    with pytest.raises(ValidationError):
        plan.patched(exercises=[PlanExercise(id="squat", sets=99)])


def test_extra_fields_are_rejected(ctx, valid_response):
    schema = WorkoutPlan.model_facing_schema()
    with pytest.raises(ValidationError):
        schema.model_validate({**valid_response, "sneaky": 1})


def _plan(ctx, response):
    partial = WorkoutPlan.model_facing_schema().model_validate(response)
    return WorkoutPlan.assemble(WorkoutPlan.resolve_computed(ctx), partial)
