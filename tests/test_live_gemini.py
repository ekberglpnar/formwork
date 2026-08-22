"""Live tests against Gemini. Skipped unless GEMINI_API_KEY is set.

These cost money, so there are few of them and each earns its place by
checking something the offline suite structurally cannot. Run them with::

    .venv/bin/python -m pytest tests/test_live_gemini.py -q -s

The offline suite proves the loop is correct given a response. These prove a
real provider will play along — above all that it accepts the schema formwork
invents on the fly for a targeted repair, which is the one design decision
that could have failed in production and nowhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import Ctx, WorkoutPlan
from formwork import ConstraintError, Session
from formwork import generate as run

pytestmark = pytest.mark.live

try:  # optional extra
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover
    pass

pytest.importorskip("google.genai", reason="needs the gemini extra")

if not os.environ.get("GEMINI_API_KEY"):
    pytest.skip("GEMINI_API_KEY not set", allow_module_level=True)

from formwork.providers.gemini import Gemini  # noqa: E402

# 30 sets against a 10-20 ceiling. Fed by hand so a repair is guaranteed
# rather than hoped for.
OVER_VOLUME = {
    "exercises": [
        {"id": "squat", "sets": 10},
        {"id": "bench", "sets": 10},
        {"id": "row", "sets": 10},
    ],
    "rationale": "Deliberately too much volume.",
}


@pytest.fixture(scope="module")
def model() -> Gemini:
    return Gemini()


def test_a_normal_run_produces_a_valid_plan(model, ctx: Ctx):
    plan, report = run(WorkoutPlan, ctx, model, max_attempts=3)

    assert plan.check(ctx).ok
    assert report.usage.total_tokens > 0
    assert report.model_calls >= 1


def test_the_narrowed_repair_schema_is_accepted(model, ctx: Ctx):
    """The load-bearing one.

    Each repair hands the provider a Pydantic model built seconds earlier
    containing a subset of the fields. If Gemini rejected those, targeted
    repair would be a local-only trick and the design would need rethinking.
    """
    session: Session[WorkoutPlan] = Session(WorkoutPlan, ctx, max_attempts=2)

    assert session.next_request() is not None
    session.feed(OVER_VOLUME)

    repair_request = session.next_request()
    assert repair_request is not None, "a 30-set plan should have triggered a repair"
    assert repair_request.kind == "repair"
    assert repair_request.fields == ("exercises",)

    raw, usage = model.generate_structured(repair_request)

    assert set(raw) == {"exercises"}, f"model answered outside the narrowed schema: {sorted(raw)}"
    session.feed(raw, usage)

    plan, _ = session.finish()
    assert plan.check(ctx).ok
    # The frozen field must survive untouched, not be regenerated.
    assert plan.rationale == OVER_VOLUME["rationale"]


def test_computed_fields_are_never_returned(model, ctx: Ctx):
    """extra="forbid" would catch this, but the point is that it never fires:
    the model is not asked for these fields, so it cannot supply them."""
    request = Session(WorkoutPlan, ctx).next_request()
    assert request is not None

    raw, _ = model.generate_structured(request)

    assert "split" not in raw
    assert "set_range" not in raw


def test_the_model_is_told_the_closed_set_and_stays_inside_it(model, ctx: Ctx):
    """Not a guarantee — just evidence the prompt is doing its job. If this
    starts failing often, the prompt needs work, not the engine."""
    try:
        plan, _ = run(WorkoutPlan, ctx, model, max_attempts=2)
    except ConstraintError:
        pytest.skip("model could not satisfy the rules this run")

    assert all(e.id in ctx.library_ids for e in plan.exercises)
