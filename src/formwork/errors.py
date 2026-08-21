"""Failures the caller is expected to handle."""

from __future__ import annotations

from typing import TYPE_CHECKING

from formwork.rules import Violation

if TYPE_CHECKING:
    from formwork.report import Report

__all__ = ["FormworkError", "ConstraintError", "StructuralError"]


class FormworkError(Exception):
    """Base class.

    Every failure carries the report, so a caller that only catches the base
    class can still bill the tokens the attempt cost.
    """

    report: Report


class ConstraintError(FormworkError):
    """The engine could not produce an object satisfying every rule.

    This is the load-bearing guarantee: the alternative to raising here would
    be returning something that breaks a rule you declared, and a library that
    does that is worse than no library, because you would have stopped
    checking.
    """

    def __init__(self, violations: list[Violation], report: Report) -> None:
        self.violations = violations
        self.report = report
        detail = "; ".join(str(v) for v in violations) or "unknown"
        super().__init__(
            f"gave up after {report.model_calls} model call(s): {detail}"
        )


class StructuralError(FormworkError):
    """The model never produced output matching the schema."""

    def __init__(self, errors: list[str], report: Report) -> None:
        self.errors = errors
        self.report = report
        detail = "; ".join(errors) or "unknown"
        super().__init__(
            f"schema never satisfied after {report.model_calls} model call(s): {detail}"
        )
