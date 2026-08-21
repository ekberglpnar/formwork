"""formwork — declare who owns which field, then guarantee the domain rules.

    from typing import Annotated
    from formwork import Spec, computed, chosen, generated, rule, generate

    class Plan(Spec):
        "A weekly training plan."
        split:     Annotated[str, computed(rules.split_for)]
        exercises: Annotated[list[Exercise], chosen(source="library", key="id")]
        rationale: Annotated[str, generated(describe="Two sentences, plain language.")]

        @rule("Weekly sets per muscle must stay inside the prescribed range.",
              fields=["exercises"], repair=repair.truncate("exercises", 10))
        def volume(self, ctx): ...

    plan, report = generate(Plan, ctx, model)
"""

from formwork.engine import Session, agenerate, generate
from formwork.errors import FormworkError, ConstraintError, StructuralError
from formwork.fields import FieldSpec, Role, chosen, computed, generated
from formwork.prompt import PromptBuilder
from formwork.providers.base import AsyncModel, Model, ModelRequest
from formwork.report import Attempt, Report, Usage
from formwork.rules import Objective, Rule, Violation, rule, soft
from formwork.spec import Spec

__version__ = "0.1.0"

__all__ = [
    # spec
    "Spec",
    "computed",
    "chosen",
    "generated",
    "FieldSpec",
    "Role",
    # rules
    "rule",
    "soft",
    "Rule",
    "Objective",
    "Violation",
    # running
    "generate",
    "agenerate",
    "Session",
    "PromptBuilder",
    # providers
    "Model",
    "AsyncModel",
    "ModelRequest",
    # results
    "Report",
    "Attempt",
    "Usage",
    # errors
    "FormworkError",
    "ConstraintError",
    "StructuralError",
    "__version__",
]
