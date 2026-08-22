"""The generation loop, written sans-IO.

``Session`` is a state machine: it hands you a request, you get it answered
however you like, you feed the answer back. The synchronous and asynchronous
drivers below are each about six lines on top of it, the test doubles drive it
without a network, and anyone with an unusual setup — a queue, a batch API, a
human in the loop — can drive it themselves. One copy of the logic.

The loop itself:

    computed fields → prompt → model → schema check → rule check
                                            ↓ fails
                              deterministic repair, if a rule declared one
                                            ↓ still fails
                              targeted repair: re-ask for the implicated
                              fields only, freeze the rest
                                            ↓ out of attempts
                                      ConstraintError

The step worth defending is the targeted one. The obvious implementation is to
regenerate everything with the errors appended to the prompt, which is what
most hand-rolled versions do; it throws away correct work, costs a full
completion, and gives the model fresh opportunities to break a rule it had
satisfied. Re-asking for two fields out of nine is cheaper and converges more
often, and the price is that rules have to say which fields they implicate.
"""

from __future__ import annotations

import math
from typing import Any, Generic, TypeVar

from pydantic import ValidationError

from formwork import repair as repair_module
from formwork.errors import ConstraintError, FormworkError, StructuralError
from formwork.prompt import PromptBuilder
from formwork.providers.base import AsyncModel, Model, ModelRequest
from formwork.report import Attempt, Report, Usage
from formwork.rules import Violation
from formwork.spec import Spec

__all__ = ["Session", "generate", "agenerate"]

S = TypeVar("S", bound=Spec)

DEFAULT_MAX_ATTEMPTS = 3


class Session(Generic[S]):
    """One candidate's worth of generation, driven by the caller."""

    def __init__(
        self,
        spec: type[S],
        ctx: Any,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        prompt_builder: PromptBuilder | None = None,
        system: str | None = None,
        targeted_repair: bool = True,
        use_declared_repairs: bool = True,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self.spec = spec
        self.ctx = ctx
        self.max_attempts = max_attempts
        # Both default on. They exist to be switched off by the benchmark, so
        # that a measured gain can be attributed to one mechanism rather than
        # to "the library"; turning them off in production only makes the loop
        # more expensive.
        self.targeted_repair = targeted_repair
        self.use_declared_repairs = use_declared_repairs
        self.prompts = prompt_builder or PromptBuilder()
        self.system = system if system is not None else self.prompts.system(spec)

        self.computed_values = spec.resolve_computed(ctx)
        self.report = Report()

        self._instance: S | None = None
        self._violations: list[Violation] = []
        self._structural: list[str] = []
        self._pending: ModelRequest | None = None
        self._result: S | None = None

    # ── driving ──────────────────────────────────────────────────────────

    def next_request(self) -> ModelRequest | None:
        """The next call to make, or None when there is nothing left to try."""
        if self._result is not None:
            return None
        if len(self.report.attempts) >= self.max_attempts:
            return None
        if self._pending is not None:
            return self._pending

        self._pending = self._build_request()
        return self._pending

    def feed(self, raw: dict[str, Any], usage: Usage | None = None) -> None:
        """Hand back what the model returned for the outstanding request."""
        request = self._pending
        if request is None:
            raise RuntimeError("feed() called with no outstanding request")
        self._pending = None

        attempt = Attempt(
            index=len(self.report.attempts),
            kind=request.kind,
            usage=usage or Usage(),
            targeted_fields=request.fields,
        )
        self.report.attempts.append(attempt)

        candidate = self._parse(request, raw, attempt)
        if candidate is None:
            return

        self._structural = []
        self._evaluate(candidate, attempt)

    def finish(self) -> tuple[S, Report]:
        """The object, or the reason there isn't one."""
        if self._result is not None:
            total, breakdown = self._result.score(self.ctx)
            self.report.soft_scores = breakdown
            return self._result, self.report
        if self._violations:
            raise ConstraintError(self._violations, self.report)
        raise StructuralError(self._structural or ["model was never called"], self.report)

    @property
    def done(self) -> bool:
        return self._result is not None

    # ── internals ────────────────────────────────────────────────────────

    def _build_request(self) -> ModelRequest:
        if self._instance is None:
            schema = self.spec.model_facing_schema()
            if self._structural:
                prompt = self.prompts.structural_retry(
                    self.spec, self.ctx, self.computed_values, self._structural
                )
            else:
                prompt = self.prompts.initial(self.spec, self.ctx, self.computed_values)
            return ModelRequest(
                prompt=prompt,
                schema=schema,
                system=self.system,
                kind="initial",
                fields=self.spec.model_owned_fields(),
            )

        targeted = self._targeted_fields()
        return ModelRequest(
            prompt=self.prompts.repair(
                self.spec, self.ctx, self._instance, self._violations, targeted
            ),
            schema=self.spec.model_facing_schema(only=targeted),
            system=self.system,
            kind="repair",
            fields=targeted,
        )

    def _targeted_fields(self) -> tuple[str, ...]:
        """Fields the violated rules pointed at, falling back to everything.

        The fallback is deliberate rather than an error: a rule that does not
        declare its fields still works, it just costs a full regeneration. The
        library should not refuse to run because someone was in a hurry.
        """
        if not self.targeted_repair:
            return self.spec.model_owned_fields()

        owned = set(self.spec.model_owned_fields())
        implicated = tuple(
            dict.fromkeys(
                name
                for violation in self._violations
                for name in violation.fields
                if name in owned
            )
        )
        return implicated or self.spec.model_owned_fields()

    def _parse(
        self, request: ModelRequest, raw: dict[str, Any], attempt: Attempt
    ) -> S | None:
        try:
            parsed = request.schema.model_validate(raw)
        except ValidationError as error:
            attempt.structural_errors = tuple(_format(error))
            self._structural = list(attempt.structural_errors)
            return None

        try:
            if request.is_repair:
                assert self._instance is not None
                return self._instance.patched(**parsed.model_dump())
            return self.spec.assemble(self.computed_values, parsed)
        except ValidationError as error:
            # The partial matched its own schema but the whole object does not
            # hold together — a cross-field constraint expressed in Pydantic.
            attempt.structural_errors = tuple(_format(error))
            self._structural = list(attempt.structural_errors)
            return None

    def _evaluate(self, candidate: S, attempt: Attempt) -> None:
        result = candidate.check(self.ctx)
        if result.ok:
            self._settle(candidate)
            return

        strategies = (
            {
                rule.name: rule.repair
                for rule in type(candidate).__formwork_rules__
                if rule.repair is not None
            }
            if self.use_declared_repairs
            else {}
        )
        patched, fired = repair_module.apply(candidate, result.violations, self.ctx, strategies)

        if fired:
            attempt.repaired_by = tuple(fired)
            candidate = patched
            result = candidate.check(self.ctx)
            if result.ok:
                self._settle(candidate)
                return

        attempt.violations = tuple(result.violations)
        self._instance = candidate
        self._violations = list(result.violations)

    def _settle(self, candidate: S) -> None:
        self._instance = candidate
        self._result = candidate
        self._violations = []


def _format(error: ValidationError) -> list[str]:
    """Pydantic errors as one readable line each, for the retry prompt."""
    lines = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        lines.append(f"{location}: {item['msg']}")
    return lines


# ── drivers ──────────────────────────────────────────────────────────────


def generate(
    spec: type[S],
    ctx: Any,
    model: Model,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    candidates: int = 1,
    prompt_builder: PromptBuilder | None = None,
    system: str | None = None,
    targeted_repair: bool = True,
    use_declared_repairs: bool = True,
) -> tuple[S, Report]:
    """Produce an object satisfying every rule, or raise.

    ``candidates`` above 1 runs the loop that many times and keeps the
    rule-valid result with the lowest soft score. It multiplies cost, so it is
    off by default and only earns its keep when the spec declares objectives.
    """
    picker: _Picker[S] = _Picker(ctx, candidates)

    for _ in range(candidates):
        session = Session(
            spec,
            ctx,
            max_attempts=max_attempts,
            prompt_builder=prompt_builder,
            system=system,
            targeted_repair=targeted_repair,
            use_declared_repairs=use_declared_repairs,
        )
        while (request := session.next_request()) is not None:
            raw, usage = model.generate_structured(request)
            session.feed(raw, usage)
        picker.offer(session)

    return picker.best()


async def agenerate(
    spec: type[S],
    ctx: Any,
    model: AsyncModel,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    candidates: int = 1,
    prompt_builder: PromptBuilder | None = None,
    system: str | None = None,
    targeted_repair: bool = True,
    use_declared_repairs: bool = True,
) -> tuple[S, Report]:
    """``generate``, awaited. Candidates run sequentially, not concurrently:
    a later candidate is only worth paying for if the earlier ones were poor,
    and firing them all at once would remove that option."""
    picker: _Picker[S] = _Picker(ctx, candidates)

    for _ in range(candidates):
        session = Session(
            spec,
            ctx,
            max_attempts=max_attempts,
            prompt_builder=prompt_builder,
            system=system,
            targeted_repair=targeted_repair,
            use_declared_repairs=use_declared_repairs,
        )
        while (request := session.next_request()) is not None:
            raw, usage = await model.generate_structured(request)
            session.feed(raw, usage)
        picker.offer(session)

    return picker.best()


class _Picker(Generic[S]):
    """Keeps the best candidate and the full cost of finding it."""

    def __init__(self, ctx: Any, candidates: int) -> None:
        self.ctx = ctx
        self.candidates = candidates
        self.attempts: list[Attempt] = []
        self.best_result: tuple[S, Report] | None = None
        self.best_score = math.inf
        self.last_error: FormworkError | None = None

    def offer(self, session: Session[S]) -> None:
        try:
            instance, report = session.finish()
        except FormworkError as error:
            self.last_error = error
            self.attempts.extend(session.report.attempts)
            return

        score, _ = instance.score(self.ctx)
        self.attempts.extend(report.attempts)
        if score < self.best_score:
            self.best_score = score
            self.best_result = (instance, report)

    def best(self) -> tuple[S, Report]:
        if self.best_result is None:
            assert self.last_error is not None
            # Re-raise with the cost of every candidate, not just the last.
            self.last_error.report.attempts = self.attempts
            self.last_error.report.candidates_considered = self.candidates
            raise self.last_error

        instance, report = self.best_result
        report.attempts = self.attempts
        report.candidates_considered = self.candidates
        return instance, report
