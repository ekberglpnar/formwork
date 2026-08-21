"""The arms under comparison, and the one loop that is not formwork's.

Six arms forming a chain in which **each consecutive pair differs by exactly
one mechanism**, so a measured difference can be attributed to that mechanism
rather than to "the library":

    single        own off · one attempt   · —          · no repairs
    own-only      own ON  · one attempt   · —          · no repairs   → ownership
    naive-retry   own off · regenerate    · —          · no repairs   (hand-rolled)
    own-retry     own ON  · regenerate    · —          · no repairs   → ownership
    own-targeted  own ON  · targeted      · —          · no repairs   → targeting
    formwork      own ON  · targeted      · —          · repairs ON   → repairs

An earlier version had ``own-only`` with declared repairs enabled, which made
the ownership row differ in two factors at once: a declared repair fires
*within* an attempt, so it is live even at one attempt. The fairness tests
caught it. Both single-attempt arms now have repairs off.

The two ownership-off arms share a prompt with the rest, differing only in the
section that delivers the constraints. That is deliberate. A baseline with
deliberately worse prose would measure my writing, not the mechanism.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from formwork import ConstraintError, Session, StructuralError
from formwork.prompt import PromptBuilder
from formwork.providers.base import Model, ModelRequest, ProviderError
from formwork.report import Usage
from formwork.spec import Spec
from bench.errors import QuotaWall
from bench.tasks import Task


@dataclass
class RunOutcome:
    """One arm, one task, one difficulty, one repetition."""

    arm: str
    task: str
    difficulty: str
    run: int

    ok: bool = False
    valid_first_try: bool = False
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    deterministic_repairs: int = 0
    soft_score: float | None = None
    violated_rules: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class RestatePromptBuilder(PromptBuilder):
    """Prompt for the arms where the model must supply the constraints itself.

    Same rules, same closed sets, same task text as every other arm. The only
    change is that the constraint values are handed over as something to copy
    into the output rather than as something already settled — which is exactly
    the difference field ownership makes, and the only thing this arm is here
    to measure.
    """

    def _facts(self, spec: type[Any], computed_values: dict[str, Any]) -> str:
        if not computed_values:
            return ""
        lines = [
            "## Constraint values",
            "Copy each of these into the matching field of your output, unchanged.",
        ]
        for name, value in computed_values.items():
            note = spec.spec_of(name).describe
            label = f"{name} ({note})" if note else name
            lines.append(f"- {label}: {value!r}")
        return "\n".join(lines)


def run_arm(
    arm: str,
    task: Task,
    difficulty: str,
    run_index: int,
    model: Model,
    *,
    max_attempts: int = 3,
) -> RunOutcome:
    ctx = task.make_ctx(difficulty)
    outcome = RunOutcome(arm=arm, task=task.name, difficulty=difficulty, run=run_index)

    config = ARMS[arm]
    try:
        if config["ownership"]:
            _run_formwork(task, ctx, model, outcome, config, max_attempts)
        else:
            _run_restating(task, ctx, model, outcome, config, max_attempts)
    except QuotaWall:
        # The one failure that must not become a data point. Let it escape and
        # stop the sweep; see bench/errors.py for why.
        raise
    except ProviderError as error:
        outcome.error = f"provider: {error}"
    except Exception as error:  # noqa: BLE001 - one bad run must not stop the sweep
        outcome.error = f"{type(error).__name__}: {error}"

    return outcome


# ── formwork arms ────────────────────────────────────────────────────────


def _run_formwork(
    task: Task,
    ctx: Any,
    model: Model,
    outcome: RunOutcome,
    config: dict[str, Any],
    max_attempts: int,
) -> None:
    attempts = 1 if config["single"] else max_attempts
    session: Session[Spec] = Session(
        task.spec,
        ctx,
        max_attempts=attempts,
        targeted_repair=config["targeted"],
        use_declared_repairs=config["deterministic"],
    )

    while (request := session.next_request()) is not None:
        raw, usage = model.generate_structured(request)
        session.feed(raw, usage)

    try:
        instance, report = session.finish()
    except (ConstraintError, StructuralError) as error:
        _record(outcome, error.report)
        outcome.violated_rules = sorted(
            {v.rule for v in getattr(error, "violations", [])}
        )
        outcome.ok = False
        return

    _record(outcome, report)
    outcome.ok = True
    outcome.soft_score = sum(report.soft_scores.values())


def _record(outcome: RunOutcome, report: Any) -> None:
    outcome.model_calls = report.model_calls
    outcome.prompt_tokens = report.usage.prompt_tokens
    outcome.completion_tokens = report.usage.completion_tokens
    outcome.latency_ms = report.usage.latency_ms
    outcome.valid_first_try = report.valid_first_try
    outcome.deterministic_repairs = len(report.deterministic_repairs)


# ── the hand-rolled arms ─────────────────────────────────────────────────


def _run_restating(
    task: Task,
    ctx: Any,
    model: Model,
    outcome: RunOutcome,
    config: dict[str, Any],
    max_attempts: int,
) -> None:
    """What a competent person writes without a library.

    Ask for the whole object, validate, paste the errors into the prompt, ask
    again for the whole object. No field is ever frozen and no failure is ever
    fixed locally.
    """
    spec = task.spec
    attempts = 1 if config["single"] else max_attempts

    prompts = RestatePromptBuilder()
    computed_values = spec.resolve_computed(ctx)
    base_prompt = prompts.initial(spec, ctx, computed_values)
    system = prompts.system(spec)
    schema = _full_schema(spec)

    prompt = base_prompt
    usage_total = Usage()
    first = True

    for _ in range(attempts):
        request = ModelRequest(
            prompt=prompt,
            schema=schema,
            system=system,
            kind="initial",
            fields=tuple(spec.model_fields),
        )
        started = time.perf_counter()
        raw, usage = model.generate_structured(request)
        usage_total = usage_total + usage
        outcome.model_calls += 1

        problems, instance = _validate(spec, ctx, raw)

        if not problems:
            outcome.ok = True
            outcome.valid_first_try = first
            outcome.soft_score = sum(instance.score(ctx)[1].values()) if instance else None
            outcome.violated_rules = []
            break

        first = False
        outcome.violated_rules = problems["rules"]
        prompt = (
            f"{base_prompt}\n\n## Your last answer was rejected\n"
            + "\n".join(f"- {line}" for line in problems["messages"])
            + "\nReturn the whole object again, corrected."
        )
        _ = time.perf_counter() - started

    outcome.prompt_tokens = usage_total.prompt_tokens
    outcome.completion_tokens = usage_total.completion_tokens
    outcome.latency_ms = usage_total.latency_ms


def _validate(spec: type[Spec], ctx: Any, raw: dict[str, Any]):
    """Same validator as every other arm, so 'valid' means one thing."""
    try:
        instance = spec.model_validate(raw)
    except ValidationError as error:
        messages = [
            f"{'.'.join(str(p) for p in item['loc']) or '<root>'}: {item['msg']}"
            for item in error.errors()
        ]
        return {"rules": ["<schema>"], "messages": messages}, None

    report = instance.check(ctx)
    if report.ok:
        return None, instance
    return (
        {
            "rules": sorted({v.rule for v in report.violations}),
            "messages": [v.message for v in report.violations],
        },
        instance,
    )


def _full_schema(spec: type[Spec]) -> type[BaseModel]:
    """The whole object, computed fields included — the model produces all of it."""
    return spec


# ── registry ─────────────────────────────────────────────────────────────

ARMS: dict[str, dict[str, Any]] = {
    #                ownership  one-shot  targeted  declared repairs
    "single":       {"ownership": False, "single": True,  "targeted": False, "deterministic": False},  # noqa: E501
    "own-only":     {"ownership": True,  "single": True,  "targeted": False, "deterministic": False},  # noqa: E501
    "naive-retry":  {"ownership": False, "single": False, "targeted": False, "deterministic": False},  # noqa: E501
    "own-retry":    {"ownership": True,  "single": False, "targeted": False, "deterministic": False},  # noqa: E501
    "own-targeted": {"ownership": True,  "single": False, "targeted": True,  "deterministic": False},  # noqa: E501
    "formwork":     {"ownership": True,  "single": False, "targeted": True,  "deterministic": True},   # noqa: E501
}

ARM_ORDER = list(ARMS)
