"""Live smoke test against Gemini.

    .venv/bin/python scripts/live_smoke.py

Not a benchmark — it makes a handful of calls and answers the questions the
offline suite structurally cannot:

  A. Does a normal run complete against a real provider?
  B. **Does the provider accept the narrowed repair schema?** This is the one
     real architectural risk. Every repair builds a fresh Pydantic model with a
     subset of the fields, so formwork asks a provider to honour a schema it
     has never seen before, mid-conversation. If that is rejected, targeted
     repair does not work in production and the design needs rethinking.

B is forced rather than hoped for: the session is driven by hand and fed a
known-bad first response, so the model is guaranteed to receive a repair
request. Costs two calls.

Everything is recorded to bench/recordings/ so the responses become offline
fixtures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

load_dotenv(ROOT / ".env")

from workout import Profile, WeeklyPlan  # noqa: E402

from formwork import ConstraintError, Session, generate  # noqa: E402
from formwork.providers import JsonlRecorder, ProviderError  # noqa: E402
from formwork.providers.gemini import Gemini  # noqa: E402

RECORDINGS = ROOT / "bench" / "recordings"

# Valid ids and a plausible shape, but 30 sets against a 10-20 ceiling. The
# engine will have to go back to the model, with a schema containing only
# 'movements'.
DELIBERATELY_OVER_VOLUME = {
    "movements": [
        {"id": "squat", "sets": 10, "reps": 8},
        {"id": "bench", "sets": 10, "reps": 8},
        {"id": "row", "sets": 10, "reps": 10},
    ],
    "rationale": "Deliberately too much volume, to force a repair.",
}


def part_a(profile: Profile, model_name: str) -> None:
    print("=" * 72)
    print("A. Normal run")
    print("=" * 72)

    model = JsonlRecorder(
        Gemini(model=model_name),
        RECORDINGS / "smoke_normal.jsonl",
    )

    try:
        plan, report = generate(WeeklyPlan, profile, model, max_attempts=3)
    except ConstraintError as error:
        print(f"gave up: {error}")
        print(f"cost   : {error.report.summary()}")
        return

    print(f"split      : {plan.split}")
    print(f"movements  : {', '.join(m.id for m in plan.movements)}")
    print(f"total sets : {plan.total_sets()}  (allowed {plan.set_range[0]}-{plan.set_range[1]})")
    print(f"rationale  : {plan.rationale}")
    print()
    print(f"report     : {report.summary()}")
    print(f"valid@1    : {report.valid_first_try}")
    for attempt in report.attempts:
        print(
            f"  attempt {attempt.index} {attempt.kind:<7} "
            f"asked={attempt.targeted_fields} "
            f"repaired={attempt.repaired_by or '-'} "
            f"unresolved={tuple(v.rule for v in attempt.violations) or '-'}"
        )


def part_b(profile: Profile, model_name: str) -> None:
    print()
    print("=" * 72)
    print("B. Forced repair — does Gemini accept the narrowed schema?")
    print("=" * 72)

    model = JsonlRecorder(
        Gemini(model=model_name),
        RECORDINGS / "smoke_repair.jsonl",
    )
    session: Session[WeeklyPlan] = Session(WeeklyPlan, profile, max_attempts=2)

    # Attempt 0: skip the model, feed it something wrong on purpose.
    first = session.next_request()
    assert first is not None
    print(f"attempt 0  fields={first.fields}  (fed a bad response by hand)")
    session.feed(DELIBERATELY_OVER_VOLUME)

    # Attempt 1: this one is real, and it carries the narrowed schema.
    second = session.next_request()
    if second is None:
        print("no repair was requested — the bad response was not bad enough")
        return

    print(f"attempt 1  kind={second.kind}  fields={second.fields}")
    print(f"           schema={second.schema.__name__} "
          f"props={list(second.schema.model_fields)}")
    print()
    print("--- repair prompt sent ---")
    print(second.prompt)
    print("--- end ---")
    print()

    try:
        raw, usage = model.generate_structured(second)
    except ProviderError as error:
        print(f"PROVIDER REJECTED THE NARROWED SCHEMA: {error}")
        print("This is the finding the whole script exists for.")
        return

    print(f"accepted. returned keys: {sorted(raw)}")
    session.feed(raw, usage)

    try:
        plan, report = session.finish()
    except ConstraintError as error:
        print(f"still invalid after repair: {error}")
        return

    print(f"movements  : {', '.join(m.id for m in plan.movements)}")
    print(f"total sets : {plan.total_sets()}")
    print(f"rationale  : {plan.rationale!r}")
    print()
    print("The rationale above came from the hand-fed first response. If it is")
    print("unchanged, field freezing works end to end against a real provider.")
    print(f"report     : {report.summary()}")


def main() -> int:
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is empty — fill in .env first.")
        return 1

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    print(f"model: {model_name}")
    print(f"recordings: {RECORDINGS}")
    print()

    profile = Profile(days_per_week=3, experience="intermediate")

    try:
        part_a(profile, model_name)
        part_b(profile, model_name)
    except ProviderError as error:
        print(f"provider error: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
