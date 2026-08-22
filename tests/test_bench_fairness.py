"""The benchmark's fairness invariants, enforced rather than promised.

A benchmark written by the author of the thing being measured is worth exactly
as much as its controls, so the controls are tests. Each one corresponds to a
way the result could be tilted without anyone noticing.
"""

from __future__ import annotations

import pytest
from bench.arms import ARMS, RestatePromptBuilder, RunOutcome, _validate
from bench.report import PAIRS, balance_check, render
from bench.tasks import DIFFICULTIES, TASKS

from formwork.fields import Role


@pytest.mark.parametrize("task_name", list(TASKS))
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_computed_values_match_the_declared_truth(task_name, difficulty):
    """The rule engine and the scorer must agree on what the constraints are.

    If they drifted, the ownership arms would be scored against different
    numbers from the restating arms.
    """
    task = TASKS[task_name]
    ctx = task.make_ctx(difficulty)
    assert task.spec.resolve_computed(ctx) == task.truth(ctx)


@pytest.mark.parametrize("task_name", list(TASKS))
def test_rules_read_the_context_not_the_object(task_name):
    """A model that invents a generous limit must not thereby satisfy the rule.

    Constructed directly with absurd computed values: the rules should still
    fail against the real context, and restates_constraints should catch the
    fabrication.
    """
    task = TASKS[task_name]
    ctx = task.make_ctx("hard")
    truth = task.truth(ctx)

    inflated = {name: _inflate(value) for name, value in truth.items()}
    owned = {
        name: [] if name != _prose_field(task) else "x"
        for name in task.spec.model_owned_fields()
    }
    instance = task.spec(**inflated, **owned)

    violations = {v.rule for v in instance.check(ctx).violations}
    assert "restates_constraints" in violations


@pytest.mark.parametrize("task_name", list(TASKS))
def test_both_prompt_styles_state_the_same_rules_and_choices(task_name):
    """The restating arms differ in one section only.

    A baseline with worse prose would measure the prompt rather than the
    mechanism, so everything except the constraint-delivery section must match
    byte for byte.
    """
    from formwork.prompt import PromptBuilder

    task = TASKS[task_name]
    ctx = task.make_ctx("medium")
    computed = task.spec.resolve_computed(ctx)

    owned_style = PromptBuilder()
    restate_style = RestatePromptBuilder()

    assert owned_style._task(task.spec) == restate_style._task(task.spec)
    assert owned_style._choices(task.spec, ctx) == restate_style._choices(task.spec, ctx)
    assert owned_style._requirements(task.spec) == restate_style._requirements(task.spec)

    # ...and the one section that is meant to differ, does.
    assert owned_style._facts(task.spec, computed) != restate_style._facts(task.spec, computed)


@pytest.mark.parametrize("task_name", list(TASKS))
def test_restating_prompt_states_every_constraint_value(task_name):
    """The restating arms are not being starved of information."""
    task = TASKS[task_name]
    ctx = task.make_ctx("medium")
    computed = task.spec.resolve_computed(ctx)
    facts = RestatePromptBuilder()._facts(task.spec, computed)

    for name, value in computed.items():
        assert name in facts
        assert repr(value) in facts


@pytest.mark.parametrize("task_name", list(TASKS))
def test_ownership_arms_hide_computed_fields_and_others_do_not(task_name):
    task = TASKS[task_name]
    facing = set(task.spec.model_facing_schema().model_fields)
    full = set(task.spec.model_fields)
    computed = set(task.spec.fields_with_role(Role.COMPUTED))

    assert computed, "task declares no computed fields; it cannot test ownership"
    assert not (facing & computed)
    assert computed <= full


def test_every_reported_comparison_differs_in_exactly_one_mechanism():
    """The load-bearing control.

    If a pair differs in two factors, the row attributing its gain to one of
    them is wrong. The subtle case is declared repairs: they fire *within* an
    attempt, so they are live even in a one-shot arm.
    """
    factors = ("ownership", "single", "targeted", "deterministic")

    def diff(a: str, b: str) -> set[str]:
        return {f for f in factors if ARMS[a][f] != ARMS[b][f]}

    for label, before, after in PAIRS:
        if label.startswith("Headline"):
            continue  # deliberately end-to-end, not an attribution
        assert len(diff(before, after)) == 1, (
            f"{label}: {before} → {after} differs in {sorted(diff(before, after))}"
        )

    assert diff("single", "own-only") == {"ownership"}
    assert diff("naive-retry", "own-retry") == {"ownership"}
    assert diff("own-retry", "own-targeted") == {"targeted"}
    assert diff("own-targeted", "formwork") == {"deterministic"}


def test_only_the_full_arm_uses_declared_repairs():
    """Any other arm with repairs on would blur the last row."""
    with_repairs = [name for name, cfg in ARMS.items() if cfg["deterministic"]]
    assert with_repairs == ["formwork"]


@pytest.mark.parametrize("task_name", list(TASKS))
def test_the_scorer_is_the_same_function_for_every_arm(task_name):
    """_validate is what the restating arms use; check() is what formwork uses.
    They must agree, since the report compares the two directly."""
    task = TASKS[task_name]
    ctx = task.make_ctx("medium")
    truth = task.truth(ctx)
    owned = {
        name: [] if name != _prose_field(task) else "x"
        for name in task.spec.model_owned_fields()
    }
    raw = {**truth, **owned}

    problems, instance = _validate(task.spec, ctx, raw)
    assert instance is not None
    direct = instance.check(ctx)

    assert bool(problems) == (not direct.ok)
    if problems:
        assert set(problems["rules"]) == {v.rule for v in direct.violations}


def test_the_report_refuses_data_with_lopsided_dropouts():
    """10% errors on the one-shot arm against 35% on the retrying one is the
    shape a rate limit produces, and rendered as a table it would show formwork
    winning for a reason that has nothing to do with formwork.
    """
    outcomes = [
        *_runs("single", n=20, errors=2),
        *_runs("formwork", n=20, errors=7),
    ]
    usable, detail = balance_check(outcomes)

    assert not usable
    assert "35%" in detail
    assert "NO RESULT" in render(outcomes, {"model": "x"})


def test_the_report_renders_when_dropouts_are_rare_and_even():
    outcomes = [
        *_runs("single", n=40, errors=1),
        *_runs("formwork", n=40, errors=1),
    ]
    usable, _ = balance_check(outcomes)

    assert usable
    assert "NO RESULT" not in render(outcomes, {"model": "x"})


def test_a_clean_sweep_is_reportable():
    outcomes = [*_runs("single", n=10, errors=0), *_runs("formwork", n=10, errors=0)]
    assert balance_check(outcomes)[0]


def _runs(arm: str, *, n: int, errors: int) -> list[RunOutcome]:
    made = []
    for i in range(n):
        failed = i < errors
        made.append(
            RunOutcome(
                arm=arm,
                task="workout",
                difficulty="easy",
                run=i,
                ok=not failed,
                valid_first_try=not failed,
                model_calls=1,
                prompt_tokens=100,
                completion_tokens=50,
                error="provider: throttled" if failed else None,
            )
        )
    return made


def _inflate(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 999
    if isinstance(value, str):
        return value + "-wrong"
    return value


def _prose_field(task) -> str:
    """The free-text field: the one generated field that is a plain string."""
    for name in task.spec.model_owned_fields():
        if task.spec.model_fields[name].annotation is str:
            return name
    raise AssertionError("task has no prose field")
