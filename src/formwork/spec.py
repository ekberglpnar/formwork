"""The ``Spec`` base class: one schema that carries both halves of the contract.

A spec says what the output looks like (it is a Pydantic model), who fills each
part (``fields.py``), and what must be true of the result (``rules.py``). The
engine reads all three off the same class, which is the whole ergonomic point —
today those three things live in a schema file, a service, and a validator that
have to be kept in agreement by hand.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Self, get_args, get_origin

from pydantic import BaseModel, ConfigDict, create_model

from formwork.fields import FieldSpec, Role, generated
from formwork.rules import Objective, Rule, RuleReport, collect

__all__ = ["Spec"]


class Spec(BaseModel):
    """Base class for a constrained generation target."""

    model_config = ConfigDict(extra="forbid")

    # Populated per-subclass by __pydantic_init_subclass__.
    __formwork_fields__: ClassVar[dict[str, FieldSpec]] = {}
    __formwork_rules__: ClassVar[list[Rule]] = []
    __formwork_objectives__: ClassVar[list[Objective]] = []
    __formwork_model_schema__: ClassVar[type[BaseModel] | None] = None

    # Instructions for the model that apply to the spec as a whole.
    __formwork_instructions__: ClassVar[str | None] = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        cls.__formwork_fields__ = _read_field_specs(cls)
        cls.__formwork_rules__, cls.__formwork_objectives__ = collect(cls)
        # Built lazily: subclasses of subclasses would inherit a stale one.
        cls.__formwork_model_schema__ = None

    # ── introspection ────────────────────────────────────────────────────

    @classmethod
    def spec_of(cls, name: str) -> FieldSpec:
        return cls.__formwork_fields__[name]

    @classmethod
    def fields_with_role(cls, role: Role) -> tuple[str, ...]:
        return tuple(n for n, s in cls.__formwork_fields__.items() if s.role is role)

    @classmethod
    def model_owned_fields(cls) -> tuple[str, ...]:
        return tuple(n for n, s in cls.__formwork_fields__.items() if s.model_facing)

    # ── the schema the model actually sees ───────────────────────────────

    @classmethod
    def model_facing_schema(cls, only: tuple[str, ...] | None = None) -> type[BaseModel]:
        """Build the Pydantic model the LLM is asked to fill.

        Computed fields are absent by construction. ``only`` narrows it further
        for a targeted repair, so a second call can ask for two fields instead
        of the entire object.
        """
        if only is None and cls.__formwork_model_schema__ is not None:
            return cls.__formwork_model_schema__

        wanted = only if only is not None else cls.model_owned_fields()
        definitions: dict[str, Any] = {}

        for name in wanted:
            spec = cls.__formwork_fields__.get(name)
            if spec is not None and not spec.model_facing:
                raise ValueError(
                    f"{name!r} is computed; it cannot be requested from the model"
                )
            definitions[name] = _redeclare(cls, name, spec)

        suffix = "Partial" if only is not None else "Output"
        built = create_model(
            f"{cls.__name__}{suffix}",
            __config__=ConfigDict(extra="forbid"),
            **definitions,
        )

        if only is None:
            cls.__formwork_model_schema__ = built
        return built  # type: ignore[no-any-return]

    # ── assembly and checking ────────────────────────────────────────────

    @classmethod
    def resolve_computed(cls, ctx: Any) -> dict[str, Any]:
        """Run every computed field's resolver against the context."""
        return {
            name: spec.resolve(ctx)
            for name, spec in cls.__formwork_fields__.items()
            if spec.role is Role.COMPUTED
        }

    @classmethod
    def assemble(cls, computed_values: dict[str, Any], model_output: BaseModel) -> Self:
        """Glue the two halves into the real object."""
        return cls(**computed_values, **model_output.model_dump())

    def patched(self, **changes: Any) -> Self:
        """A copy with ``changes`` applied, re-validated.

        Repairs go through here so a strategy cannot quietly produce an object
        that would not have passed the schema.
        """
        data = self.model_dump()
        data.update(changes)
        return type(self)(**data)

    def check(self, ctx: Any) -> RuleReport:
        """Run every hard rule. Order is declaration order, which makes the
        first reported violation stable across runs and therefore testable."""
        report = RuleReport()
        for rule in type(self).__formwork_rules__:
            report.violations.extend(rule.check(self, ctx))
        return report

    def score(self, ctx: Any) -> tuple[float, dict[str, float]]:
        """Total weighted soft score (lower is better) and the breakdown."""
        breakdown = {o.name: o.score(self, ctx) for o in type(self).__formwork_objectives__}
        return sum(breakdown.values()), breakdown


def _read_field_specs(cls: type[BaseModel]) -> dict[str, FieldSpec]:
    """Pull the FieldSpec out of each field's Annotated metadata.

    Fields without one default to ``generated``: the permissive reading is the
    right one here, since a user who has not thought about a field has not
    thought about handing it to their rule engine either.
    """
    specs: dict[str, FieldSpec] = {}
    for name, info in cls.model_fields.items():
        found = [m for m in info.metadata if isinstance(m, FieldSpec)]
        if len(found) > 1:
            raise TypeError(f"field {name!r} declares {len(found)} roles; expected one")
        specs[name] = found[0] if found else generated()
    return specs


def _redeclare(cls: type[BaseModel], name: str, spec: FieldSpec | None) -> tuple[Any, Any]:
    """Rebuild one field for the model-facing schema.

    Our own FieldSpec is stripped from the annotation — it is instruction for
    the engine, not for the model — while any Pydantic constraints the user
    attached (``Field(ge=1)`` and friends) are preserved, because those are
    exactly the cheap structural guards we want the model held to.
    """
    info = cls.model_fields[name]
    annotation: Any = info.annotation
    carried = [m for m in info.metadata if not isinstance(m, FieldSpec)]

    if carried:
        annotation = Annotated[tuple([annotation, *carried])]

    description = spec.describe if spec else None
    if description and not info.description:
        rebuilt = info.__class__(default=info.default, description=description)
    else:
        rebuilt = info

    return (annotation, rebuilt)


def item_type_of(annotation: Any) -> Any:
    """Element type of a list annotation, or None. Used by repair strategies."""
    if get_origin(annotation) in (list, tuple):
        args = get_args(annotation)
        return args[0] if args else None
    return None
