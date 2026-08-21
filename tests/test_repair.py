"""Deterministic repair strategies, exercised directly."""

from __future__ import annotations

from formwork import Violation, repair
from conftest import WorkoutPlan

VIOLATION = Violation(rule="whatever", message="...", fields=("exercises",))


def plan(ctx, exercises, rationale="ok"):
    partial = WorkoutPlan.model_facing_schema().model_validate(
        {"exercises": exercises, "rationale": rationale}
    )
    return WorkoutPlan.assemble(WorkoutPlan.resolve_computed(ctx), partial)


def ids(instance):
    return [e.id for e in instance.exercises]


# ── drop_invalid ─────────────────────────────────────────────────────────


def test_drop_invalid_removes_unknown_ids(ctx):
    before = plan(ctx, [{"id": "squat", "sets": 4}, {"id": "ghost", "sets": 4}])
    after = repair.drop_invalid("exercises")(before, VIOLATION, ctx)
    assert ids(after) == ["squat"]


def test_drop_invalid_reads_the_pool_from_the_field_declaration(ctx):
    """No second source= argument: the field already said where the set lives."""
    before = plan(ctx, [{"id": "bench", "sets": 3}])
    assert repair.drop_invalid("exercises")(before, VIOLATION, ctx) is None


def test_drop_invalid_declines_to_empty_the_field(ctx):
    """Returning an empty plan is a different failure, not a repair."""
    before = plan(ctx, [{"id": "ghost", "sets": 4}])
    assert repair.drop_invalid("exercises")(before, VIOLATION, ctx) is None


def test_drop_invalid_with_an_explicit_source(ctx):
    ctx.allowed_now = ["squat"]
    before = plan(ctx, [{"id": "squat", "sets": 4}, {"id": "bench", "sets": 4}])
    after = repair.drop_invalid("exercises", source="allowed_now", key="id")(
        before, VIOLATION, ctx
    )
    assert ids(after) == ["squat"]


# ── truncate ─────────────────────────────────────────────────────────────


def test_truncate_caps_a_list(ctx):
    before = plan(ctx, [{"id": "squat", "sets": 2}] * 6)
    after = repair.truncate("exercises", 4)(before, VIOLATION, ctx)
    assert len(after.exercises) == 4


def test_truncate_accepts_a_callable_limit(ctx):
    before = plan(ctx, [{"id": "squat", "sets": 2}] * 6)
    after = repair.truncate("exercises", lambda c: c.max_per_day)(before, VIOLATION, ctx)
    assert len(after.exercises) == ctx.max_per_day


def test_truncate_is_a_no_op_when_already_short(ctx):
    before = plan(ctx, [{"id": "squat", "sets": 2}])
    assert repair.truncate("exercises", 4)(before, VIOLATION, ctx) is None


# ── clamp ────────────────────────────────────────────────────────────────


def test_clamp_squeezes_an_item_field(ctx):
    before = plan(ctx, [{"id": "squat", "sets": 9}, {"id": "bench", "sets": 2}])
    after = repair.clamp("exercises", 3, 5, item_field="sets")(before, VIOLATION, ctx)
    assert [e.sets for e in after.exercises] == [5, 3]


def test_clamp_is_a_no_op_inside_the_range(ctx):
    before = plan(ctx, [{"id": "squat", "sets": 4}])
    assert repair.clamp("exercises", 3, 5, item_field="sets")(before, VIOLATION, ctx) is None


# ── dedupe ───────────────────────────────────────────────────────────────


def test_dedupe_keeps_the_first_occurrence(ctx):
    before = plan(
        ctx,
        [
            {"id": "squat", "sets": 4},
            {"id": "squat", "sets": 2},
            {"id": "bench", "sets": 3},
        ],
    )
    after = repair.dedupe("exercises", key="id")(before, VIOLATION, ctx)
    assert ids(after) == ["squat", "bench"]
    assert after.exercises[0].sets == 4


# ── chain ────────────────────────────────────────────────────────────────


def test_chain_applies_every_strategy(ctx):
    before = plan(
        ctx,
        [{"id": "ghost", "sets": 9}] + [{"id": "squat", "sets": 9}] * 6,
    )
    strategy = repair.chain(
        repair.drop_invalid("exercises"),
        repair.truncate("exercises", 3),
        repair.clamp("exercises", 1, 4, item_field="sets"),
    )
    after = strategy(before, VIOLATION, ctx)
    assert ids(after) == ["squat", "squat", "squat"]
    assert all(e.sets == 4 for e in after.exercises)


def test_chain_reports_nothing_when_no_link_fires(ctx):
    before = plan(ctx, [{"id": "squat", "sets": 4}])
    strategy = repair.chain(
        repair.drop_invalid("exercises"),
        repair.truncate("exercises", 10),
    )
    assert strategy(before, VIOLATION, ctx) is None


# ── the dispatcher ───────────────────────────────────────────────────────


def test_apply_only_runs_strategies_for_violated_rules(ctx):
    before = plan(ctx, [{"id": "squat", "sets": 4}, {"id": "ghost", "sets": 4}])
    violations = before.check(ctx).violations
    patched, fired = repair.apply(
        before,
        violations,
        ctx,
        {"known_exercises": repair.drop_invalid("exercises")},
    )
    assert fired == ["known_exercises"]
    assert ids(patched) == ["squat"]


def test_apply_reports_nothing_when_a_strategy_declines(ctx):
    before = plan(ctx, [{"id": "squat", "sets": 9}, {"id": "bench", "sets": 9}])
    violations = before.check(ctx).violations
    _, fired = repair.apply(
        before,
        violations,
        ctx,
        {"known_exercises": repair.drop_invalid("exercises")},
    )
    assert fired == []
