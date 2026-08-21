"""Turning a spec into the text the model reads.

Two prompts matter. The first states the computed fields as *decisions already
made* rather than as requests — the difference between "use a 4-day upper/lower
split" and "please pick a sensible split" is most of the reliability gap. The
second is the repair prompt, and its job is to be small: here is what you
returned, here is precisely what is wrong, change only these fields.

Replace this wholesale by passing your own builder to the engine; nothing else
in the library reads these strings.
"""

from __future__ import annotations

import json
from typing import Any

from formwork.fields import Role
from formwork.rules import Violation

__all__ = ["PromptBuilder"]

_MAX_LISTED_CHOICES = 60

DEFAULT_SYSTEM = (
    "You produce structured output for a system that will validate it against "
    "hard domain rules before use. Facts stated in the prompt are already "
    "decided: honour them exactly, do not restate or adjust them. Prefer "
    "returning fewer, correct items over filling space."
)


class PromptBuilder:
    """Default prompt strategy."""

    def system(self, spec: type[Any]) -> str:
        return DEFAULT_SYSTEM

    # ── initial ──────────────────────────────────────────────────────────

    def initial(
        self,
        spec: type[Any],
        ctx: Any,
        computed_values: dict[str, Any],
    ) -> str:
        sections = [
            self._task(spec),
            self._facts(spec, computed_values),
            self._choices(spec, ctx),
            self._free(spec),
            self._requirements(spec),
        ]
        return "\n\n".join(s for s in sections if s)

    def _task(self, spec: type[Any]) -> str:
        instructions = getattr(spec, "__formwork_instructions__", None)
        if instructions:
            return str(instructions).strip()
        doc = (spec.__doc__ or "").strip()
        return doc or f"Produce a {spec.__name__}."

    def _facts(self, spec: type[Any], computed_values: dict[str, Any]) -> str:
        if not computed_values:
            return ""
        lines = ["## Already decided", "These are fixed. Work within them; do not return them."]
        for name, value in computed_values.items():
            note = spec.spec_of(name).describe
            label = f"{name} ({note})" if note else name
            lines.append(f"- {label}: {_render(value)}")
        return "\n".join(lines)

    def _choices(self, spec: type[Any], ctx: Any) -> str:
        chosen = spec.fields_with_role(Role.CHOSEN)
        if not chosen:
            return ""
        lines = ["## Closed sets", "Values outside these lists will be rejected."]
        for name in chosen:
            field_spec = spec.spec_of(name)
            allowed = field_spec.allowed_values(ctx) or ()
            lines.append(f"- {name}: {_render_choices(allowed)}")
            if field_spec.describe:
                lines.append(f"  {field_spec.describe}")
            if field_spec.max_items is not None:
                lines.append(f"  at most {field_spec.max_items} item(s)")
        return "\n".join(lines)

    def _free(self, spec: type[Any]) -> str:
        free = [
            name
            for name in spec.fields_with_role(Role.GENERATED)
            if spec.spec_of(name).describe
        ]
        if not free:
            return ""
        lines = ["## Your judgement"]
        for name in free:
            lines.append(f"- {name}: {spec.spec_of(name).describe}")
        return "\n".join(lines)

    def _requirements(self, spec: type[Any]) -> str:
        rules = getattr(spec, "__formwork_rules__", [])
        stated = [(r.name, _describe_rule(r)) for r in rules]
        stated = [(n, d) for n, d in stated if d]
        if not stated:
            return ""
        lines = ["## Rules your output must satisfy"]
        lines.extend(f"- {desc}" for _, desc in stated)
        return "\n".join(lines)

    # ── repair ───────────────────────────────────────────────────────────

    def repair(
        self,
        spec: type[Any],
        ctx: Any,
        instance: Any,
        violations: list[Violation],
        targeted: tuple[str, ...],
    ) -> str:
        current = {name: getattr(instance, name) for name in targeted}
        lines = [
            "Your previous output broke rules that are not negotiable.",
            "",
            "## What you returned for the fields in question",
            _render(current),
            "",
            "## What is wrong",
        ]
        lines.extend(f"- {v.message}" for v in violations)
        lines += [
            "",
            "## What to do",
            f"Return only these fields: {', '.join(targeted)}.",
            "Change as little as possible. Everything else is already correct "
            "and will be kept as it is.",
        ]
        closed = self._choices(spec, ctx)
        if closed:
            lines += ["", closed]
        return "\n".join(lines)

    def structural_retry(
        self,
        spec: type[Any],
        ctx: Any,
        computed_values: dict[str, Any],
        errors: list[str],
    ) -> str:
        base = self.initial(spec, ctx, computed_values)
        problems = "\n".join(f"- {e}" for e in errors)
        return (
            f"{base}\n\n## Your last attempt did not match the schema\n{problems}\n"
            "Return valid JSON for the schema this time."
        )


def _describe_rule(rule: Any) -> str:
    if rule.message:
        return str(rule.message)
    doc = (rule.fn.__doc__ or "").strip()
    return doc.split("\n")[0] if doc else ""


def _render(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=_fallback)
    except (TypeError, ValueError):
        return str(value)


def _fallback(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def _render_choices(allowed: tuple[Any, ...]) -> str:
    if len(allowed) <= _MAX_LISTED_CHOICES:
        return ", ".join(str(a) for a in allowed)
    head = ", ".join(str(a) for a in allowed[:_MAX_LISTED_CHOICES])
    return f"{head}, ... ({len(allowed)} total; anything not listed is still rejected)"
